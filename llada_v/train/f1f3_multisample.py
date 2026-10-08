# Multi-sample validation: F1 regional effect + F3 independence.
# Records per image/mode, no large tensors:
#   entropy_bits [steps, 33, GEN]  per-layer logit-lens entropy
#   delta        [steps, GEN]      staleness per suffix position (cache mode)
#   gen_ids      [GEN]             final decoded ids -> repeat positions
# Delta follows the paper definition: steps since a position was last FRESHLY
# RECOMPUTED, counting BOTH the periodic full-suffix refresh (every
# gen_interval steps, which bypasses refresh_index) and the similarity refresh.
import os, sys, json, math, time, copy, random
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
os.environ.setdefault('HF_ENDPOINT','https://hf-mirror.com')
os.environ.setdefault('HF_HOME','/data/zhaoqiyan/autodl-tmp/hf_cache')
os.environ.setdefault('HUGGINGFACE_HUB_CACHE','/data/zhaoqiyan/autodl-tmp/hf_cache/hub')
import numpy as np, torch
from dataclasses import asdict
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.cache import dLLMCache, dLLMCacheConfig
from llava.hooks import register_cache_LLaDA_V
from llava.hooks import cache_hook_LLaDA_V

MODE=os.environ.get('F3_MODE','dllm_cache')
assert MODE in ('baseline','dllm_cache')
GEN=int(os.environ.get('F1F3_GEN','128')); STEPS=GEN; BLOCK=GEN
OUT=os.environ.get('F1F3_OUT','/data/zhaoqiyan/autodl-tmp/information_flow/multisample')
QUESTION='Please describe the image in detail.'
PI,GI,TR=25,7,0.25
os.makedirs(OUT,exist_ok=True)

_im=os.environ.get('F1F3_IMAGES','').strip()
if _im:
    IMAGES=[x for x in _im.split(',') if x]
else:
    root='/data/zhaoqiyan/autodl-tmp/coco2014/val2014'
    fs=sorted(f for f in os.listdir(root) if f.lower().endswith(('.jpg','.jpeg','.png')))
    random.seed(int(os.environ.get('F1F3_SEED','0')))
    IMAGES=[os.path.join(root,f) for f in random.sample(fs,int(os.environ.get('F1F3_N','60')))]

device='cuda:0'
tokenizer,model,image_processor,_=load_pretrained_model(
    'GSAI-ML/LLaDA-V',None,'llava_llada',attn_implementation='sdpa',device_map=device)
model.eval()

class DeltaRec:
    def __init__(self,g): self.g=g; self.reset()
    def reset(self): self.delta=np.zeros(self.g,dtype=np.int32); self.rows=[]
    def step_start(self): self.delta+=1
    def snapshot(self): self.rows.append(self.delta.copy())
    def mark(self,idx):
        idx=np.asarray(idx).reshape(-1); idx=idx[(idx>=0)&(idx<self.g)]
        self.delta[idx]=0
    def mark_all(self): self.delta[:]=0
REC=DeltaRec(GEN)
CACHE_INST=None
if MODE=='dllm_cache':
    dLLMCache.new_instance(**asdict(dLLMCacheConfig(
        prompt_interval_steps=PI,gen_interval_steps=GI,transfer_ratio=TR)))
    register_cache_LLaDA_V(model,'model.layers')
    CACHE_INST=dLLMCache()
    _orig=cache_hook_LLaDA_V.refresh_index
    def _wrapped(new_features,cached_features=None,transfer_ratio=0.5,layer_id=0):
        out=_orig(new_features,cached_features,transfer_ratio,layer_id)
        if out is not None and out.numel()>0:   # union over layers, matching attention_recorder
            REC.mark(out.detach().cpu().numpy())
        return out
    cache_hook_LLaDA_V.refresh_index=_wrapped
    print('[ms] dLLM-Cache ON (%d,%d,%s) + delta recorder'%(PI,GI,TR))
else:
    print('[ms] baseline (no cache)')

core=model.model; layers=core.layers; fnorm=core.norm; head=model.lm_head
N=len(layers); LOG2E=1.0/math.log(2.0)
step_rows=[]; mask_rows=[]; _cur={}
MASK_ID=126336
_masked_emb=model.model.embed_tokens(torch.tensor([MASK_ID],device=device)).detach()
def ent_bits(h):
    with torch.no_grad():
        z=head(fnorm(h[:,-GEN:,:])).float()
        lp=torch.log_softmax(z,dim=-1)
        e=-(lp.exp()*lp).sum(-1)*LOG2E
    return e[0].to(torch.float16).cpu().numpy()
def pre_hook(m,a,k=None):
    h=a[0] if a else k['hidden_states']
    if MODE=='dllm_cache': REC.step_start()
    with torch.no_grad():
        gm=(torch.abs(h[:,-GEN:,:]-_masked_emb)<1e-5).all(-1)[0]
    mask_rows.append(gm.cpu().numpy())
    _cur[0]=ent_bits(h)
def mk(d):
    def hk(m,i,o):
        h=o[0] if isinstance(o,(tuple,list)) else o
        if d==1 and MODE=='dllm_cache' and CACHE_INST.refresh_gen(0):
            REC.mark_all()
        _cur[d]=ent_bits(h)
        if d==N:
            step_rows.append(np.stack([_cur[x] for x in range(N+1)])); _cur.clear()
            if MODE=='dllm_cache': REC.snapshot()
    return hk
hs=[layers[0].register_forward_pre_hook(pre_hook)]+[b.register_forward_hook(mk(i+1)) for i,b in enumerate(layers)]

recs=[]
for n,p in enumerate(IMAGES,1):
    p=p.strip(); stem=os.path.splitext(os.path.basename(p))[0]
    dst=os.path.join(OUT,'%s_G%d_%s.npz'%(MODE,GEN,stem))
    if os.path.exists(dst):
        print('[%d/%d] skip %s'%(n,len(IMAGES),stem),flush=True); continue
    step_rows.clear(); mask_rows.clear(); _cur.clear(); REC.reset()
    im=Image.open(p).convert('RGB')
    it=[t.to(dtype=torch.float16,device=device) for t in process_images([im],image_processor,model.config)]
    cv=copy.deepcopy(conv_templates['llava_llada'])
    cv.append_message(cv.roles[0],DEFAULT_IMAGE_TOKEN+'\n'+QUESTION); cv.append_message(cv.roles[1],None)
    ids=tokenizer_image_token(cv.get_prompt(),tokenizer,IMAGE_TOKEN_INDEX,return_tensors='pt').unsqueeze(0).to(device)
    t0=time.time()
    try:
        cont=model.generate(ids,images=it,image_sizes=[im.size],steps=STEPS,gen_length=GEN,
            block_length=BLOCK,tokenizer=tokenizer,stopping_criteria=['<|eot_id|>'],
            prefix_refresh_interval=32,threshold=1)
    except Exception as e:
        print('  FAIL %s: %s'%(stem,e),flush=True); continue
    dt=time.time()-t0
    E=np.stack(step_rows) if step_rows else np.zeros((0,N+1,GEN),dtype=np.float16)
    D=np.stack(REC.rows) if REC.rows else np.zeros((0,GEN),dtype=np.int32)
    gid=cont[0,-GEN:].cpu().numpy()
    M=np.stack(mask_rows) if mask_rows else np.zeros((0,GEN),dtype=bool)
    np.savez_compressed(dst,entropy_bits=E.astype(np.float16),delta=D.astype(np.int16),
                        is_masked=M.astype(bool),gen_ids=gid)
    txt=tokenizer.batch_decode(cont,skip_special_tokens=True)[0]
    recs.append({'image':p,'mode':MODE,'sec':round(dt,1),'E':list(E.shape),'D':list(D.shape),'text':txt[:150]})
    print('[%d/%d] %s %.1fs E%s D%s'%(n,len(IMAGES),stem,dt,E.shape,D.shape),flush=True)
    with open(os.path.join(OUT,'%s_meta.json'%MODE),'w') as f: json.dump(recs,f,indent=1)
for h in hs: h.remove()
print('[ms] DONE %d'%len(recs))
