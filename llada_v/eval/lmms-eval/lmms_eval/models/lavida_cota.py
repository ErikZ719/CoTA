"""lmms-eval model wrapper for LaViDa-LLaDA with the dLLM-Cache hook and the CoTA/CoTA++ components (2026-09-24).

Same decoding as experiments/repeat_eval/scripts/run_repeat_eval_lavida.py (Table V): LaViDa's own generate()
re-implemented step by step without the prefix cache, temperature 0, low-confidence remasking, one token per step,
semi-AR blocks. The backend and the components come from the environment, exactly like the LLaDA-V wrapper:
  COTA_BACKEND      none | dllm_cache
  COTA_CACHE_PI/GI/TR   25 / 7 / 0.10 (LaViDa's Table V setting)
  COTA_CTAE=reroute_q COTA_CTAR_THETA COTA_DAR_R COTA_DAR_MODE COTA_CTEV_MODE COTA_CTEV_LAMBDA
Generation length per task comes from the task's max_new_tokens (gen_kwargs) or GEN_L; the block is
min(128, L) (LaViDa's released lmms-eval rule) unless GEN_BLOCK is set. Registered as "lavida_cota".
"""
import atexit, copy, glob, os, sys, warnings
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

warnings.filterwarnings("ignore")
Z = os.environ.get("COTA_Z", "/data/zhaoqiyan")
REPO = f"{Z}/autodl-tmp/LaViDa"
DLLM = f"{Z}/autodl-tmp/dLLM-cache"
SCRIPTS = f"{Z}/autodl-tmp/experiments/repeat_eval/scripts"
for _p in (SCRIPTS, DLLM, REPO):                       # LaViDa's own `llava` must shadow the LLaDA-V one in site-packages
    if _p not in sys.path:
        sys.path.insert(0, _p)
import mmada_cotapp as MC                              # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402  (LaViDa)
from llava.mm_utils import process_images, tokenizer_image_token  # noqa: E402
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN  # noqa: E402
from llava.conversation import conv_templates  # noqa: E402

CKPT = glob.glob(f"{Z}/autodl-tmp/hf_cache/hub/models--jacklishufan--lavida-llada-v1.0-instruct/snapshots/*")[0]
MASK_ID = 126336


def _env_components():
    E = os.environ.get
    return dict(ctar=E("COTA_CTAE", "off") == "reroute_q", lo=int(E("COTA_STITCH_LO", 24)), hi=int(E("COTA_STITCH_HI", 31)),
                theta=int(E("COTA_CTAR_THETA", 1)), w=int(E("COTA_CTAR_W", 5)),
                dar_r=int(E("COTA_DAR_R", 0)), dar_mode=E("COTA_DAR_MODE", "legacy"),
                ctae_mode="off", ctae_sigma=5.0, ctae_gamma=0.5,
                ctev_mode=E("COTA_CTEV_MODE", "off"), ctev_lambda=float(E("COTA_CTEV_LAMBDA", 0.25)),
                ctev_window=int(E("COTA_CTEV_WINDOW", 5)), ctev_norm=int(E("COTA_CTEV_NORM", 1)))


@register_model("lavida_cota")
class LaViDaCoTA(lmms):
    def __init__(self, pretrained: str = CKPT, device: str = "cuda", batch_size: int = 1, **kwargs) -> None:
        super().__init__()
        from accelerate import Accelerator
        acc = Accelerator()
        if acc.num_processes > 1:
            self._device = torch.device(f"cuda:{acc.local_process_index}")
            self._rank, self._world_size = acc.local_process_index, acc.num_processes
        else:
            self._device = torch.device(device)
            self._rank, self._world_size = 0, 1
        self.accelerator = acc
        vision_kwargs = dict(mm_vision_tower="google/siglip-so400m-patch14-384", mm_resampler_type=None,
                             mm_projector_type="mlp2x_gelu", mm_hidden_size=1152, use_mm_proj=True)
        self._tokenizer, self._model, self._image_processor, _ = load_pretrained_model(
            pretrained, None, "llava_llada", device_map=str(self._device), vision_kwargs=vision_kwargs, torch_dtype="bfloat16")
        self._model.eval(); self._model.tie_weights(); self._model.to(torch.bfloat16)
        self.llm = self._model.get_model()
        self.llm.set_activation_checkpointing(None)
        tk = self._tokenizer
        self.terms = {i for i in (tk.eos_token_id, tk.convert_tokens_to_ids("<|eot_id|>"), tk.convert_tokens_to_ids("<|endoftext|>"))
                      if isinstance(i, int) and i >= 0}
        self.batch_size_per_gpu = 1
        self._cota_setup()

    # ---------------------------------------------------------------- backend + components (environment driven)
    def _cota_setup(self):
        E = os.environ.get
        self.backend = E("COTA_BACKEND", "none")
        self.comp = _env_components()
        MC.configure(**self.comp)
        self.use_port = MC.any_on()
        if self.backend == "dllm_cache":
            from dataclasses import asdict
            from dllm_cache.cache import dLLMCache, dLLMCacheConfig
            self.cache_cfg = dict(prompt_interval_steps=int(E("COTA_CACHE_PI", 25)), gen_interval_steps=int(E("COTA_CACHE_GI", 7)),
                                  transfer_ratio=float(E("COTA_CACHE_TR", 0.10)))
            dLLMCache.new_instance(**asdict(dLLMCacheConfig(**self.cache_cfg)))
            H = MC.build_dllm_hook()                    # with no component this is the original dLLM-Cache hook
            H.register_cache_MMaDA(self.llm, "transformer.blocks")
            print(f"[lavida_cota] dLLM-Cache {self.cache_cfg} components={self.comp if self.use_port else None}", flush=True)
        elif self.backend == "slowfast":
            # 2026-09-30, SlowFast rows of the general-benchmark table: the sampler, the evolved cache and the
            # embedding shim of run_repeat_eval_lavida.py --mode slowfast. One sampler per (length, block).
            self.cache_cfg = dict(prompt_interval_steps=int(E("COTA_CACHE_PI", 25)), gen_interval_steps=int(E("COTA_CACHE_GI", 7)),
                                  transfer_ratio=float(E("COTA_CACHE_TR", 0.10)))
            self._SF = MC.build_sf_stack(self.llm) if self.use_port else __import__("sf_mmada_stack")
            self._SF.SFFeatureCache.new_instance(prompt_interval_steps=self.cache_cfg["prompt_interval_steps"],
                                                 gen_interval_steps=self.cache_cfg["gen_interval_steps"],
                                                 cfg_interval_steps=1,
                                                 transfer_ratio=self.cache_cfg["transfer_ratio"])
            self._SF.register_sf_cache_MMaDA(self.llm, "transformer.blocks")
            _comp, _use_port = self.comp, self.use_port

            class _LaViDaShim:
                """The sampler calls model(x, attention_mask=...); LaViDa needs the prefix as embeddings."""
                def __init__(self, inner):
                    self._llm = inner
                    self.device = next(inner.parameters()).device
                    self.prefix = None

                def __call__(self, x, attention_mask=None):
                    need = _comp["ctev_mode"] != "off"
                    emb = self._llm.transformer.wte(x)
                    emb[:, :self.prefix.shape[1]] = self.prefix
                    out = self._llm(None, input_embeddings=emb, output_hidden_states=need)
                    if _use_port:
                        MC.S["hid"] = (out.hidden_states, x.shape[1]) if need else None
                    return out

            self._sf_shim = _LaViDaShim(self.llm)
            self._sf_samplers = {}
            print(f"[lavida_cota] SlowFast {self.cache_cfg} components={self.comp if self.use_port else None}", flush=True)
        elif self.backend == "none":
            assert not self.use_port, "components need a cache backend"
            print("[lavida_cota] baseline (no cache)", flush=True)
        else:
            raise ValueError(f"COTA_BACKEND={self.backend} not supported here (SlowFast rows are not part of Table VII)")

        def _coverage():
            print(f"[lavida_cota] STATS {dict(MC.STATS)} comp={self.comp} backend={self.backend}", flush=True)
        atexit.register(_coverage)

    # ---------------------------------------------------------------- lmms plumbing
    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def eot_token_id(self):
        return self._tokenizer.eos_token_id

    def tok_encode(self, string, left_truncate_len=None, add_special_tokens=None):
        return self._tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self._tokenizer.decode(tokens)

    def loglikelihood(self, requests):
        raise NotImplementedError("lavida_cota: generate_until only")

    def flatten(self, x):
        return [item for sub in x for item in sub]

    # ---------------------------------------------------------------- decoding (== run_repeat_eval_lavida.py::generate)
    @torch.no_grad()
    def _generate(self, inputs_embeds, L, block, steps):
        llm, comp, device = self.llm, self.comp, self._device
        P = inputs_embeds.shape[1]
        MC.reset(P, L)
        x = torch.full((1, P + L), MASK_ID, dtype=torch.long, device=device)
        x[:, :P] = 0                                 # placeholder ids; the prefix lives in inputs_embeds
        assert L % block == 0
        nblocks = L // block
        assert steps % nblocks == 0
        spb = steps // nblocks
        need_h = comp["ctev_mode"] != "off"
        for nb in range(nblocks):
            bm = (x[:, P + nb * block: P + (nb + 1) * block] == MASK_ID)
            ntt = MC.get_num_transfer_tokens(bm, spb)
            for i in range(spb):
                mask_index = (x == MASK_ID)
                if not mask_index[:, P + nb * block: P + (nb + 1) * block].any():
                    continue
                emb = llm.transformer.wte(x)
                emb[:, :P] = inputs_embeds
                out = llm(None, input_embeddings=emb, output_hidden_states=need_h)
                logits = out.logits
                x0 = torch.argmax(logits, dim=-1)
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                x0_p[:, P + (nb + 1) * block:] = -np.inf
                x0 = torch.where(mask_index, x0, x)
                conf = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=device, dtype=x0_p.dtype))
                if comp["dar_r"] > 0 and comp["dar_mode"] == "legacy":
                    MC.publish_scores(conf[0, P:P + L])
                if need_h:
                    pen = MC.penalty(llm, out.hidden_states, x[0, P:P + L], P, L, MASK_ID)
                    conf[:, P:P + L] = conf[:, P:P + L] - pen.to(conf.dtype)
                ti = torch.zeros_like(x0, dtype=torch.bool)
                _, sel = torch.topk(conf[0], k=int(ntt[0, i]))
                ti[0, sel] = True
                x[ti] = x0[ti]
                if comp["dar_r"] > 0 and comp["dar_mode"] in ("masked", "score"):
                    sv = conf[0, P:P + L].detach().clone()
                    if comp["dar_mode"] == "masked":
                        sv[ti[0, P:P + L]] = -float("inf")
                    MC.publish_scores(sv)
                MC.publish_mask(x[0, P:P + L] == MASK_ID)
                MC.STATS["steps"] += 1
        return x[:, P:]

    def _trim(self, ids):
        out = []
        for t in ids:
            if t in self.terms:
                break
            out.append(t)
        return out

    # ---------------------------------------------------------------- lmms-eval entry point
    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        for req in requests:
            context, gen_kwargs, doc_to_visual, doc_id, task, split = req.args
            gen_kwargs = dict(gen_kwargs)
            until = gen_kwargs.pop("until", None)
            if isinstance(until, str):
                until = [until]
            visuals = doc_to_visual(self.task_dict[task][split][doc_id])
            visuals = self.flatten([visuals]) if not isinstance(visuals, list) else visuals
            L = int(os.environ.get("GEN_L", gen_kwargs.get("max_new_tokens", 256)))
            block = int(os.environ.get("GEN_BLOCK", min(128, L)))
            steps = int(os.environ.get("GEN_STEPS", L))          # one token per step (LaViDa's released rule)
            if visuals:
                img = visuals[0].convert("RGB") if hasattr(visuals[0], "convert") else visuals[0]
                image_tensor = process_images([img], self._image_processor, self._model.config)
                image_tensor = [t.to(dtype=torch.bfloat16, device=self._device) for t in image_tensor]
                question = context if DEFAULT_IMAGE_TOKEN in context else DEFAULT_IMAGE_TOKEN + "\n" + context
                image_sizes = [img.size]
            else:
                image_tensor, question, image_sizes = None, context, None
            conv = copy.deepcopy(conv_templates["llada"])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            input_ids = tokenizer_image_token(conv.get_prompt(), self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self._device)
            with torch.no_grad():
                if image_tensor is not None:
                    (_, _, _, _, inputs_embeds, _) = self._model.prepare_inputs_labels_for_multimodal(
                        input_ids, None, None, None, None, image_tensor, ["image"], image_sizes=image_sizes)
                else:
                    inputs_embeds = self.llm.transformer.wte(input_ids)
                if self.backend == "dllm_cache":
                    from dllm_cache.cache import dLLMCache
                    dLLMCache().reset_cache(prompt_length=inputs_embeds.shape[1])
                if self.backend == "slowfast":
                    if (L, block) not in self._sf_samplers:
                        self._sf_samplers[(L, block)] = self._SF.SlowFastSampler(
                            self._sf_shim, {"gen_length": L, "block_length": block}, mask_id=MASK_ID)
                    self._sf_shim.prefix = inputs_embeds.to(torch.bfloat16)
                    MC.reset(inputs_embeds.shape[1], L)
                    gen = self._sf_samplers[(L, block)].generate(
                        torch.zeros((1, inputs_embeds.shape[1]), dtype=torch.long, device=self._device), None)
                else:
                    gen = self._generate(inputs_embeds.to(torch.bfloat16), L, block, steps)
            text = self._tokenizer.decode(self._trim(gen[0].tolist()), skip_special_tokens=True).strip()
            if until:
                for u in until:
                    if u and u in text:
                        text = text.split(u)[0]
            res.append(text)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), [text])
            pbar.update(1)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests):
        raise NotImplementedError
