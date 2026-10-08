#!/usr/bin/env python
'''Which positions does the similarity-based selection of dLLM-Cache recompute between two full refreshes?
Reads only the small arrays of each recorded step (refresh set, committed position, prompt length). CPU only.'''
import numpy as np, os, json
from multiprocessing import Pool
IF='/data/zhaoqiyan/autodl-tmp/information_flow'; ROOT=IF+'/attn_multi100/dllm_cache'; GEN=128
def one(stem):
    d=os.path.join(ROOT,stem); committed=np.zeros(GEN,bool); commit_step=np.full(GEN,-1)
    rows=[]
    for t in range(GEN):
        z=np.load(os.path.join(d,'step_%d.npz'%t))
        rf=np.asarray(z['refresh_q_index']).reshape(-1); rf=np.unique(rf[(rf>=0)&(rf<GEN)])
        pl=int(z['prompt_length']); ti=np.asarray(z['transfer_index']).reshape(-1)
        now=[int(i) for i in (np.where(ti)[0]-pl) if 0<=int(i)<GEN]
        full = (len(rf)==GEN) or t==0
        if not full:
            r=np.zeros(GEN,bool); r[rf]=True
            masked=~committed
            age=np.where(committed,t-commit_step,-1)
            recent=committed&(age<=7)
            rows.append([t,len(rf),masked.sum(),(r&masked).sum(),committed.sum(),(r&committed).sum(),recent.sum(),(r&recent).sum()])
        else:
            rows.append([t,-len(rf) if len(rf) else -GEN,0,0,0,0,0,0])
        for i in now:
            if not committed[i]: committed[i]=True; commit_step[i]=t
    return np.array(rows)
if __name__=='__main__':
    stems=[os.path.splitext(os.path.basename(f))[0] for f in json.load(open(IF+'/coco100_seed0.json'))['files']]
    with Pool(24) as p: res=p.map(one,stems)
    R=np.stack(res)                      # [100,128,8]
    full=R[:,:,1]<0
    print('images',R.shape[0],'; full-refresh steps per image: min %d max %d ; steps of full refresh in image 0: %s'%(full.sum(1).min(),full.sum(1).max(),np.where(full[0])[0].tolist()))
    P=R[~full]
    print('partial steps: positions recomputed per step mean %.2f (min %d, max %d)'%(P[:,1].mean(),P[:,1].min(),P[:,1].max()))
    print('per-step chance of being recomputed: still-masked %.3f ; committed %.3f ; committed within the last 7 steps %.3f'%(P[:,3].sum()/P[:,2].sum(),P[:,5].sum()/P[:,4].sum(),P[:,7].sum()/P[:,6].sum()))
    print('share of the recomputed slots that go to committed positions %.3f ; to still-masked %.3f'%(P[:,5].sum()/P[:,1].sum(),P[:,3].sum()/P[:,1].sum()))
    # by quarter
    for q in range(4):
        m=(~full)&(R[:,:,0]//32==q); Q=R[m]
        print('  quarter %d: masked %.3f  committed %.3f  | slots to committed %.3f'%(q+1,Q[:,3].sum()/max(Q[:,2].sum(),1),Q[:,5].sum()/max(Q[:,4].sum(),1),Q[:,5].sum()/Q[:,1].sum()))
    # per image, for a test over images
    pm=np.array([r[r[:,1]>0][:,3].sum()/r[r[:,1]>0][:,2].sum() for r in R]); pc=np.array([r[r[:,1]>0][:,5].sum()/r[r[:,1]>0][:,4].sum() for r in R])
    print('per image: masked mean %.3f [min %.3f max %.3f]; committed mean %.3f [min %.3f max %.3f]; committed higher in %d/100 images'%(pm.mean(),pm.min(),pm.max(),pc.mean(),pc.min(),pc.max(),(pc>pm).sum()))
    np.savez_compressed(IF+'/results/F2/refresh_share100.npz',data=R)
