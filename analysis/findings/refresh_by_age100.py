#!/usr/bin/env python
'''Why does the similarity-based selection of dLLM-Cache favour committed positions?
Chance that a position is recomputed at a step between two full refreshes, as a function of the number of steps
since it was committed (age 1 = committed at the previous step). Still-masked positions are the reference. CPU only.'''
import numpy as np, os, json
from multiprocessing import Pool
IF='/data/zhaoqiyan/autodl-tmp/information_flow'; ROOT=IF+'/attn_multi100/dllm_cache'; GEN=128; AMAX=16
def one(stem):
    d=os.path.join(ROOT,stem); committed=np.zeros(GEN,bool); cstep=np.full(GEN,-1)
    num=np.zeros((4,AMAX+2)); den=np.zeros((4,AMAX+2))      # [quarter, age]; age 0 = still masked, AMAX+1 = older
    for t in range(GEN):
        z=np.load(os.path.join(d,'step_%d.npz'%t))
        rf=np.asarray(z['refresh_q_index']).reshape(-1); rf=np.unique(rf[(rf>=0)&(rf<GEN)])
        pl=int(z['prompt_length']); ti=np.asarray(z['transfer_index']).reshape(-1)
        now=[int(i) for i in (np.where(ti)[0]-pl) if 0<=int(i)<GEN]
        if not ((len(rf)==GEN) or t==0):
            r=np.zeros(GEN,bool); r[rf]=True
            age=np.where(committed,np.minimum(t-cstep,AMAX+1),0)
            q=t//32
            for a in range(AMAX+2):
                m=age==a; den[q,a]+=m.sum(); num[q,a]+=(r&m).sum()
        for i in now:
            if not committed[i]: committed[i]=True; cstep[i]=t
    return num,den
if __name__=='__main__':
    J=json.load(open(IF+'/coco100_seed0.json')); stems=[os.path.splitext(os.path.basename(f))[0] for f in J['files']]
    with Pool(24) as p: res=p.map(one,stems)
    NUM=np.stack([r[0] for r in res]); DEN=np.stack([r[1] for r in res])      # [100,4,AMAX+2]
    np.savez_compressed(IF+'/results/F2/refresh_by_age100.npz',num=NUM,den=DEN)
    n=NUM.sum((0,1)); d=DEN.sum((0,1))
    print('chance of being recomputed at a partial step, by steps since commit (0 = still masked, %d = older):'%(AMAX+1))
    print('  '+'  '.join('%d:%.3f'%(a,n[a]/d[a]) for a in range(AMAX+2)))
    for q in range(4):
        n=NUM[:,q].sum(0); d=DEN[:,q].sum(0)
        print('  quarter %d: '%(q+1)+'  '.join('%d:%.3f'%(a,n[a]/max(d[a],1)) for a in range(AMAX+2)))
    pi=NUM[:,:,1].sum(1)/DEN[:,:,1].sum(1); print('per image, age 1: mean %.3f min %.3f max %.3f'%(pi.mean(),pi.min(),pi.max()))
