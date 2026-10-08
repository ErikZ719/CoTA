"""lmms-eval model wrapper for MMaDA-8B-Base with the dLLM-Cache hook and the CoTA/CoTA++ components (2026-09-24).

Same pipeline as experiments/repeat_eval/scripts/run_repeat_eval_mmada.py (Table V): official MMU input format
([<|mmu|>][<|soi|>][MAGVIT-v2 codes][<|eoi|>][chat-templated text]), 256x256 image, temperature 0, low-confidence
remasking, block-wise semi-AR decoding through dLLM-cache's demo_MMada_mmu_cache.mmu_generate_with_cache (or the
component port mmada_cotapp.mmu_generate_cotapp when a component is on). Backend and components from the environment
(COTA_BACKEND none|dllm_cache, COTA_CACHE_PI/GI/TR 20/10/0.10, COTA_CTAE, COTA_CTAR_THETA, COTA_DAR_R, COTA_DAR_MODE,
COTA_CTEV_MODE, COTA_CTEV_LAMBDA). Generation settings: GEN_L / GEN_STEPS / GEN_BLOCK (max_new_tokens, steps, block).
Registered as "mmada_cota".
"""
import atexit, os, sys, warnings
from typing import List

import torch
from torchvision import transforms
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

warnings.filterwarnings("ignore")
Z = os.environ.get("COTA_Z", "/data/zhaoqiyan")
DLLM = f"{Z}/autodl-tmp/dLLM-cache"
SCRIPTS = f"{Z}/autodl-tmp/experiments/repeat_eval/scripts"
for _p in (SCRIPTS, DLLM):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(DLLM)                                          # repo-local packages (mmada_models, mmada_training, dllm_cache)
import mmada_cotapp as MC                               # noqa: E402
from transformers import AutoTokenizer                  # noqa: E402
from mmada_models import MMadaModelLM, MAGVITv2         # noqa: E402
from mmada_training.prompting_utils import UniversalPrompting  # noqa: E402
from demo_MMada_mmu_cache import mmu_generate_with_cache  # noqa: E402

MASK_ID = 126336
CHAT_TEMPLATE = (
    "{% set loop_messages = messages %}{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n'"
    "+ message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
    "{{ content }}{% endfor %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n' }}")


def _env_components():
    E = os.environ.get
    return dict(ctar=E("COTA_CTAE", "off") == "reroute_q", lo=int(E("COTA_STITCH_LO", 24)), hi=int(E("COTA_STITCH_HI", 31)),
                theta=int(E("COTA_CTAR_THETA", 1)), w=int(E("COTA_CTAR_W", 5)),
                dar_r=int(E("COTA_DAR_R", 0)), dar_mode=E("COTA_DAR_MODE", "legacy"),
                ctae_mode="off", ctae_sigma=5.0, ctae_gamma=0.5,
                ctev_mode=E("COTA_CTEV_MODE", "off"), ctev_lambda=float(E("COTA_CTEV_LAMBDA", 0.25)),
                ctev_window=int(E("COTA_CTEV_WINDOW", 5)), ctev_norm=int(E("COTA_CTEV_NORM", 1)))


@register_model("mmada_cota")
class MMaDACoTA(lmms):
    def __init__(self, pretrained: str = os.environ.get("MMADA_CKPT", "Gen-Verse/MMaDA-8B-Base"), device: str = "cuda", batch_size: int = 1, resolution: int = 256, **kwargs) -> None:
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
        self._model = MMadaModelLM.from_pretrained(pretrained, trust_remote_code=True, torch_dtype=torch.bfloat16).to(self._device).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        self.vq_model = MAGVITv2().from_pretrained("showlab/magvitv2").to(self._device)
        self.uni_prompting = UniversalPrompting(
            self._tokenizer, max_text_len=512,
            special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
            ignore_id=-100, cond_dropout_prob=0.1, use_reserved_token=True)
        self._tokenizer.chat_template = CHAT_TEMPLATE
        tk = self._tokenizer
        self.terms = {i for i in (tk.eos_token_id, tk.convert_tokens_to_ids("<|eot_id|>")) if isinstance(i, int) and i >= 0}
        self.resolution = int(os.environ.get("MMADA_RES", resolution))   # 256 = the Base checkpoint default, 512 = the official VLMEvalKit recipe
        self.tfm = transforms.Compose([
            transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop((self.resolution, self.resolution)),
            transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        self.batch_size_per_gpu = 1
        self._cota_setup()

    def _cota_setup(self):
        E = os.environ.get
        self.backend = E("COTA_BACKEND", "none")
        self.comp = _env_components()
        self.gen_fn = None
        if self.backend == "dllm_cache":
            from dataclasses import asdict
            from dllm_cache.cache import dLLMCache, dLLMCacheConfig
            from dllm_cache import register_cache_MMaDA
            self.cache_cfg = dict(prompt_interval_steps=int(E("COTA_CACHE_PI", 20)), gen_interval_steps=int(E("COTA_CACHE_GI", 10)),
                                  transfer_ratio=float(E("COTA_CACHE_TR", 0.10)))
            dLLMCache.new_instance(**asdict(dLLMCacheConfig(**self.cache_cfg)))
            MC.configure(**self.comp)
            self.use_port = MC.any_on()
            if self.use_port:
                H = MC.build_dllm_hook()
                H.register_cache_MMaDA(self._model, "model.transformer.blocks")
                self.gen_fn = MC.mmu_generate_cotapp
                print(f"[mmada_cota] dLLM-Cache {self.cache_cfg} + components {self.comp}", flush=True)
            else:
                register_cache_MMaDA(self._model, "model.transformer.blocks")
                print(f"[mmada_cota] dLLM-Cache {self.cache_cfg}", flush=True)
        elif self.backend == "slowfast":
            # 2026-09-30, SlowFast rows of the general-benchmark table: the sampler and the evolved cache of
            # run_repeat_eval_mmada.py --mode slowfast. One sampler per (length, block).
            self.cache_cfg = dict(prompt_interval_steps=int(E("COTA_CACHE_PI", 20)), gen_interval_steps=int(E("COTA_CACHE_GI", 10)),
                                  transfer_ratio=float(E("COTA_CACHE_TR", 0.10)))
            MC.configure(**self.comp)
            self.use_port = MC.any_on()
            if self.use_port:
                self._SF = MC.build_sf_stack(self._model)
                self._sf_shim = MC.CotaShim(self._model)
            else:
                import sf_mmada_stack as _SFM
                self._SF = _SFM

                class _ModelShim:
                    """MMaDA's forward takes no attention_mask; the sampler passes one."""
                    def __init__(self, m):
                        self._m = m
                        self.device = next(m.parameters()).device

                    def __call__(self, x, attention_mask=None):
                        return self._m(x)

                self._sf_shim = _ModelShim(self._model)
            self._SF.SFFeatureCache.new_instance(prompt_interval_steps=self.cache_cfg["prompt_interval_steps"],
                                                 gen_interval_steps=self.cache_cfg["gen_interval_steps"],
                                                 cfg_interval_steps=1,
                                                 transfer_ratio=self.cache_cfg["transfer_ratio"])
            self._SF.register_sf_cache_MMaDA(self._model, "model.transformer.blocks")
            self._sf_samplers = {}
            print(f"[mmada_cota] SlowFast {self.cache_cfg} components={self.comp if self.use_port else None}", flush=True)
        elif self.backend == "none":
            MC.configure(**self.comp)
            assert not MC.any_on(), "components need a cache backend"
            self.use_port = False
            print("[mmada_cota] baseline (no hook registered)", flush=True)
        else:
            raise ValueError(f"COTA_BACKEND={self.backend} not supported here")

        def _coverage():
            print(f"[mmada_cota] STATS {dict(MC.STATS)} comp={self.comp} backend={self.backend}", flush=True)
        atexit.register(_coverage)

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
        raise NotImplementedError("mmada_cota: generate_until only")

    def _trim(self, ids):
        out = []
        for t in ids:
            if t in self.terms:
                break
            out.append(t)
        return out

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        sp = self.uni_prompting.sptids_dict
        for req in requests:
            context, gen_kwargs, doc_to_visual, doc_id, task, split = req.args
            gen_kwargs = dict(gen_kwargs)
            until = gen_kwargs.pop("until", None)
            if isinstance(until, str):
                until = [until]
            visuals = doc_to_visual(self.task_dict[task][split][doc_id])
            visuals = visuals if isinstance(visuals, list) else [visuals]
            L = int(os.environ.get("GEN_L", gen_kwargs.get("max_new_tokens", 128)))
            steps = int(os.environ.get("GEN_STEPS", L))
            block = int(os.environ.get("GEN_BLOCK", L))
            text_prompt = context.replace("<image>", "").strip()     # the image precedes the text in MMaDA's MMU format
            with torch.no_grad():
                text_ids = self._tokenizer.apply_chat_template([{"role": "user", "content": text_prompt}], tokenize=True,
                                                               add_generation_prompt=True, return_tensors="pt").to(self._device)
                if visuals:
                    img = visuals[0].convert("RGB")
                    pixel = self.tfm(img).unsqueeze(0).to(self._device)
                    image_tokens = self.vq_model.get_code(pixel) + len(self._tokenizer)
                    input_ids = torch.cat([sp["<|mmu|>"].to(self._device).unsqueeze(0), sp["<|soi|>"].to(self._device).unsqueeze(0),
                                           image_tokens, sp["<|eoi|>"].to(self._device).unsqueeze(0), text_ids], dim=1).long()
                else:
                    input_ids = text_ids.long()
                attention_mask = torch.ones_like(input_ids)
                if self.backend == "dllm_cache":
                    from dllm_cache.cache import dLLMCache
                    dLLMCache().reset_cache(prompt_length=input_ids.shape[1])
                if self.backend == "slowfast":
                    if (L, block) not in self._sf_samplers:
                        self._sf_samplers[(L, block)] = self._SF.SlowFastSampler(
                            self._sf_shim, {"gen_length": L, "block_length": block})
                    output = torch.cat([input_ids, self._sf_samplers[(L, block)].generate(input_ids, attention_mask)], dim=1)
                elif self.gen_fn is not None:
                    output = self.gen_fn(self._model, input_ids=input_ids, max_new_tokens=L, steps=steps, block_length=block,
                                         mask_id=MASK_ID, attention_mask=attention_mask)
                else:
                    output = mmu_generate_with_cache(self._model, input_ids=input_ids, max_new_tokens=L, steps=steps, block_length=block,
                                                     temperature=0, remasking="low_confidence", mask_id=MASK_ID, attention_mask=attention_mask)
                gen_ids = output[:, input_ids.shape[1]:]
            text = self._tokenizer.decode(self._trim(gen_ids[0].tolist()), skip_special_tokens=True).strip()
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
