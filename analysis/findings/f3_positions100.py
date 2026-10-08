#!/usr/bin/env python
'''F3 on the 100-image sample: one row per suffix position, nothing aggregated. Runs on the server (CPU).

Source: information_flow/multisample/dllm_cache_G128_<stem>.npz and baseline_G128_<stem>.npz
        entropy_bits [steps,33,GEN] (index 0 = embedding, index l = layer l), is_masked [steps,GEN], delta, gen_ids.
Deep window W = layers 26..30 (the window CTEV reads).
Columns
  img pos step delta rep               as in f2f3_positions100.py (rep = repeat on content tokens)
  own_dec                              deep-window entropy of the position itself at its decode moment
  ctx_dec  n_ctx                       mean deep-window entropy of the COMMITTED neighbours within +-5 at the decode moment, and their number
  fin_00..fin_32                       per-layer entropy of the position at the last step (end of generation), cached run
  bfin_00..bfin_32                     the same in the vanilla run
  bctx_dec bn_ctx bstep                the context score in the vanilla run, at the position's own decode moment there
'''
import numpy as np, os, json
from multiprocessing import Pool
IF='/data/zhaoqiyan/autodl-tmp/information_flow'; ROOT=IF+'/multisample'
EOT={126081,126348}; LAYOUT={198,220,197,256,262144}; W=slice(26,31); R5=5
def decision(M):
    us={}
    for i in range(M.shape[1]):
        m=np.where(M[:,i])[0]
        if len(m): us[i]=int(m[-1])
    return us
def repeats(g):
    G=len(g); cut=next((k for k,x in enumerate(g) if int(x) in EOT),G)
    seq=[(k,int(x)) for k,x in enumerate(g[:cut]) if int(x) not in LAYOUT]
    return set(seq[k][0] for k in range(1,len(seq)) if seq[k][1]==seq[k-1][1])
def ctx(E,M,t,i,G):
    js=[j for j in range(max(0,i-R5),min(G,i+R5+1)) if j!=i and not M[t,j]]
    if not js: return np.nan,0
    return float(E[t,W][:,js].mean()),len(js)
def one(a):
    k,stem=a
    fc=os.path.join(ROOT,'dllm_cache_G128_%s.npz'%stem); fb=os.path.join(ROOT,'baseline_G128_%s.npz'%stem)
    c=np.load(fc); b=np.load(fb)
    D=c['delta'].astype(int); M=c['is_masked'].astype(bool); g=np.asarray(c['gen_ids']).reshape(-1); E=c['entropy_bits'].astype(np.float32)
    Mb=b['is_masked'].astype(bool); Eb=b['entropy_bits'].astype(np.float32)
    G=D.shape[1]
    if len(g)<G: g=np.concatenate([g,[126348]*(G-len(g))])
    us=decision(M); ub=decision(Mb); con=repeats(g[:G]); T=E.shape[0]-1
    rows=[]
    for i in range(1,G):
        if i not in us or i not in ub: continue
        t=us[i]; tb=ub[i]
        s,n=ctx(E,M,t,i,G); sb,nb=ctx(Eb,Mb,tb,i,G)
        rows.append([k,i,t,int(D[t,i]),int(i in con),float(E[t,W,i].mean()),s,n]+[float(x) for x in E[T,:,i]]+[float(x) for x in Eb[T,:,i]]+[sb,nb,tb])
    return np.array(rows,dtype=np.float64)
if __name__=='__main__':
    files=json.load(open(IF+'/coco100_seed0.json'))['files']
    stems=[os.path.splitext(os.path.basename(f))[0] for f in files]
    with Pool(20) as p: res=p.map(one,list(enumerate(stems)))
    R=np.concatenate(res)
    cols=['img','pos','step','delta','rep','own_dec','ctx_dec','n_ctx']+['fin_%02d'%l for l in range(33)]+['bfin_%02d'%l for l in range(33)]+['bctx_dec','bn_ctx','bstep']
    assert R.shape[1]==len(cols)
    np.savez_compressed(IF+'/results/F3/f3_100_positions.npz',data=R,cols=np.array(cols),images=np.array(stems))
    print('rows',R.shape,'repeat positions',int(R[:,4].sum()),'images with a repeat',len(set(R[R[:,4]==1][:,0].astype(int))),'rows without a committed neighbour',int(np.isnan(R[:,6]).sum()))
