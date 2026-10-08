#!/usr/bin/env python
'''Is the union-over-layers staleness a lower bound of what each layer carries?
The recorder keeps, per layer, the attention rows the model decodes against: a row recomputed at step t is
overwritten, any other row is carried over. A row that differs from the previous step was therefore recomputed
in that layer. From this we rebuild the staleness of every position in every layer and compare it, at the
position's own decoding step, with the union-based Delta of Eq. (staleness). Runs on the server.'''
import numpy as np, os, json, sys
from multiprocessing import Pool
IF='/data/zhaoqiyan/autodl-tmp/information_flow'; ROOT=IF+'/attn_multi100/dllm_cache'; GEN=128
N=int(os.environ.get('LSP_N','20'))
def dq(d): return (d['quantized_attentions'].astype(np.float32)-float(d['zero_point']))*float(d['scale'])
def one(stem):
    d=os.path.join(ROOT,stem); prev=None; L=None
    last_l=None; last_u=np.zeros(GEN,int); us={}; out=[]
    stale_at={}
    for t in range(GEN):
        z=np.load(os.path.join(d,'step_%d.npz'%t)); A=dq(z)[:,0][:,:,-GEN:,-GEN:]      # [L,H,GEN,GEN]
        if L is None: L=A.shape[0]; last_l=np.zeros((L,GEN),int)
        tol=1.5*float(z['scale'])
        if prev is None: ch=np.ones((L,GEN),bool)
        else: ch=(np.abs(A-prev).max(axis=(1,3))>tol)                                     # [L,GEN] row changed in this layer
        last_l[ch]=t
        rf=np.asarray(z['refresh_q_index']).reshape(-1); rf=rf[(rf>=0)&(rf<GEN)]; last_u[rf]=t
        if t==0: last_u[:]=0
        pl=int(z['prompt_length']); ti=np.asarray(z['transfer_index']).reshape(-1)
        for i in (np.where(ti)[0]-pl):
            i=int(i)
            if 0<=i<GEN and i not in us:
                us[i]=t; out.append([i,t,t-last_u[i]]+list(t-last_l[:,i]))
        # agreement of the two ways of telling what was recomputed
        if prev is not None and t%16==5:
            u=np.zeros(GEN,bool); u[rf]=True; any_l=ch.any(0)
            stale_at[t]=(int((u&any_l).sum()),int((u&~any_l).sum()),int((~u&any_l).sum()))
        prev=A
    return stem,np.array(out),stale_at
if __name__=='__main__':
    stems=[os.path.splitext(os.path.basename(f))[0] for f in json.load(open(IF+'/coco100_seed0.json'))['files']][:N]
    with Pool(min(N,20)) as p: res=p.map(one,stems)
    R=np.concatenate([r[1] for r in res]); L=R.shape[1]-3
    u=R[:,2]; per=R[:,3:]
    print('images %d, positions %d, layers %d'%(len(res),len(R),L))
    agree=np.array([v for r in res for v in r[2].values()]).sum(0)
    print('recomputed per the recorder AND seen to change in some layer: %d ; recorder only: %d ; change only: %d'%tuple(agree))
    print('union-based Delta at decode: mean %.2f'%u.mean())
    for nm,sl in (('shallow L1-8',slice(0,8)),('middle L9-24',slice(8,24)),('deep L25-32',slice(24,32))):
        b=per[:,sl]; print('  %-13s per-layer staleness: mean %.2f ; below the union value in %.2f%% of cases ; equal in %.1f%% ; above in %.1f%%'%(nm,b.mean(),100*(b<u[:,None]).mean(),100*(b==u[:,None]).mean(),100*(b>u[:,None]).mean()))
    print('min over layers equals the union value in %.1f%% of positions'%(100*(per.min(1)==u).mean()))
    z0=u==0
    print('positions with union Delta = 0: %d ; of these, share whose deep-band layers are ALL fresh %.1f%% ; mean deep-band staleness %.2f'%(z0.sum(),100*(per[z0][:,24:32].max(1)==0).mean(),per[z0][:,24:32].mean()))
    np.savez_compressed(IF+'/results/F2/layer_staleness_probe.npz',data=R)
