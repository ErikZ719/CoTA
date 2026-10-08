# Multi-sample attention collection for the F1 regional analysis.
# Reuses the validated attention_recorder (buffer-accumulating: freshly computed
# query rows overwrite, unrefreshed rows carry over the cached values), so the
# saved matrices are the EFFECTIVE attention the model actually decodes against,
# and the format matches the reference-sample attn_npz exactly.
#
# Env: F1A_MODE=baseline|dllm_cache  F1A_N=20  F1A_SEED=0  F1A_OUT=<dir>
import os, sys, json, time, copy, random, shutil
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
from llava.model.language_model.utils.attention_recorder import (
    attach_attention_recorder, detach_attention_recorder)

MODE=os.environ.get('F1A_MODE','dllm_cache'); assert MODE in ('baseline','dllm_cache')
TAG=os.environ.get('F1A_TAG',MODE)                 # output subdir; lets components coexist
DAR_R=int(os.environ.get('F1A_DAR_R','0'))
DAR_W=int(os.environ.get('F1A_DAR_W','0'))   # also refresh the +-w neighbours of each imminent position
CTAE=os.environ.get('F1A_CTAE','off')
BAND_LO=int(os.environ.get('F1A_BAND_LO','24')); BAND_HI=int(os.environ.get('F1A_BAND_HI','31'))
THETA=int(os.environ.get('F1A_CTAR_THETA','0'))
GEN=128; STEPS=128; BLOCK=128; LASTQ=128; LASTK=128
OUT=os.environ.get('F1A_OUT','/data/zhaoqiyan/autodl-tmp/information_flow/attn_multi')
QUESTION='Please describe the image in detail.'
PI,GI,TR=25,int(os.environ.get('F1A_GI','7')),0.25
os.makedirs(OUT,exist_ok=True)

root='/data/zhaoqiyan/autodl-tmp/coco2014/val2014'
fs=sorted(f for f in os.listdir(root) if f.lower().endswith(('.jpg','.jpeg','.png')))
random.seed(int(os.environ.get('F1A_SEED','0')))
# same seed/order as the multisample run, so images line up with the entropy data
_ex=os.environ.get('F1A_IMAGES','').strip()
if _ex:
    IMAGES=[x for x in _ex.split(',') if x]
else:
    picks=random.sample(fs,60)[:int(os.environ.get('F1A_N','20'))]
    IMAGES=[os.path.join(root,f) for f in picks]

device='cuda:0'
tokenizer,model,image_processor,_=load_pretrained_model(
    'GSAI-ML/LLaDA-V',None,'llava_llada',attn_implementation='sdpa',device_map=device)
model.eval()
if MODE=='dllm_cache':
    dLLMCache.new_instance(**asdict(dLLMCacheConfig(
        prompt_interval_steps=PI,gen_interval_steps=GI,transfer_ratio=TR)))
    register_cache_LLaDA_V(model,'model.layers')
    print('[f1a] dLLM-Cache ON (%d,%d,%s)'%(PI,GI,TR))
    from llava.hooks import cache_hook_LLaDA_V as _ch
    from llava.model.language_model.utils import attention_recorder as _rec
    _ch.set_ctae(mode=CTAE)
    if DAR_R>0: _ch.set_dar(True,r=DAR_R,w=DAR_W)
    if CTAE.startswith(('stitch','reroute')):
        _ch.set_stitch(True,M=GEN,lo=BAND_LO,hi=BAND_HI)
        _ch.set_ctarx(scope='all',theta=THETA)
    _rec.set_recorder_components(dar=DAR_R>0, ctar_theta=THETA>0,
                                 ctar_A=CTAE.startswith(('stitch','reroute')))
    print('[f1a] components: dar_r=%d dar_w=%d ctae=%s band=[%d,%d] theta=%d gi=%d'%(DAR_R,DAR_W,CTAE,BAND_LO,BAND_HI,THETA,GI))
else:
    print('[f1a] baseline (no cache)')

recs=[]
for n,p in enumerate(IMAGES,1):
    stem=os.path.splitext(os.path.basename(p))[0]
    dst=os.path.join(OUT,TAG,stem)
    done=os.path.join(dst,'step_%d.npz'%(STEPS-1))
    if os.path.exists(done):
        print('[%d/%d] skip %s'%(n,len(IMAGES),stem),flush=True); continue
    if os.path.isdir(dst): shutil.rmtree(dst)          # clear partial runs
    attach_attention_recorder(model,dst,save_last_q=LASTQ,save_last_k=LASTK)
    if MODE=='dllm_cache':
        if DAR_R>0: _ch.reset_dar()
        if CTAE.startswith(('stitch','reroute')): _ch.reset_stitch()
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
        print('  FAIL %s: %s'%(stem,e),flush=True); detach_attention_recorder(model); continue
    dt=time.time()-t0
    detach_attention_recorder(model)
    txt=tokenizer.batch_decode(cont,skip_special_tokens=True)[0]
    nz=len([x for x in os.listdir(dst) if x.endswith('.npz')]) if os.path.isdir(dst) else 0
    recs.append({'image':p,'mode':MODE,'tag':TAG,'dar_r':DAR_R,'ctae':CTAE,'theta':THETA,
                 'sec':round(dt,1),'n_npz':nz,'text':txt[:400]})
    print('[%d/%d] %s %.1fs  npz=%d'%(n,len(IMAGES),stem,dt,nz),flush=True)
    with open(os.path.join(OUT,'%s_meta.json'%TAG),'w') as f: json.dump(recs,f,indent=1)
print('[f1a] DONE %d'%len(recs))
