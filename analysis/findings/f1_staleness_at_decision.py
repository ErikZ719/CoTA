'''Why would DAR repair F1's dispersion? F1 measures anchoring AT each position's decision
step; DAR reserves refresh slots for exactly the positions about to be decided. If that is
the mechanism, the staleness Delta (steps since the position was last recomputed) at the
decision step should collapse to ~0 under DAR.
Note: under CTAR the recorder counts every re-anchored row as refreshed, so its Delta is
routing freshness, not state freshness.'''
import numpy as np, os, sys
ROOT='/data/zhaoqiyan/autodl-tmp/information_flow/attn_multi'; GEN=128
CONDS=sys.argv[1:] or ['dllm_cache','dar','ctar','dar_ctar']
REGION=os.environ.get('F1C_REGION','dllm_cache')
def scan(dirp):
    us={}; last={i:None for i in range(GEN)}; dl={}; fin=None
    for t in range(GEN):
        f=os.path.join(dirp,'step_%d.npz'%t)
        if not os.path.exists(f): break
        d=np.load(f)
        pl=int(d['prompt_length']); ti=np.asarray(d['transfer_index']).reshape(-1)
        for j in np.asarray(d['refresh_q_index']).reshape(-1):
            j=int(j)
            if 0<=j<GEN: last[j]=t
        for i in (np.where(ti)[0]-pl):
            i=int(i)
            if 0<=i<GEN and i not in us: us[i]=t; dl[i]=(t-last[i]) if last[i] is not None else t
        if t==GEN-1: fin=np.asarray(d['token_ids']).reshape(-1)[pl:][:GEN]
    return us,dl,fin
stems=sorted(set.intersection(*[set(os.listdir(os.path.join(ROOT,c))) for c in CONDS]))
acc={c:{'all':[],'reg':[]} for c in CONDS}
for stem in stems:
    _,_,fin=scan(os.path.join(ROOT,REGION,stem))
    region=set()
    for i in [i for i in range(1,GEN) if fin[i]==fin[i-1]]: region.update(range(max(0,i-5),min(GEN,i+6)))
    for c in CONDS:
        _,dl,_=scan(os.path.join(ROOT,c,stem))
        for i,v in dl.items():
            acc[c]['all'].append(v)
            if i in region: acc[c]['reg'].append(v)
print('images: %d'%len(stems))
print('%-11s %22s %22s'%('condition','ALL: P(D=0)  mean D','REGION: P(D=0)  mean D'))
for c in CONDS:
    a=np.array(acc[c]['all']); r=np.array(acc[c]['reg'])
    print('%-11s %10.3f %10.2f %12.3f %10.2f   (n=%d / %d)'%(c,(a==0).mean(),a.mean(),(r==0).mean() if len(r) else np.nan,r.mean() if len(r) else np.nan,len(a),len(r)))
