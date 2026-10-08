'''Does a component repair the anchoring dispersion F1 diagnoses?

Same protocol as f1_multisample.py -- each position is measured at ITS OWN decision
step, the vanilla run supplies the undisrupted reference for the identical index
range, and the repeat REGION (repeat +-5) is defined from the plain-cache run so
every condition is scored on the same set of positions.

d_w5(condition) = w5(condition) - w5(vanilla). A condition that repairs anchoring
drives d_w5 toward 0; one that leaves it dispersed keeps it negative.
'''
import numpy as np, os, json, sys
from scipy import stats as sps

ROOT='/data/zhaoqiyan/autodl-tmp/information_flow/attn_multi'
GEN=128
BANDS={'shallow(L1-8)':list(range(0,8)),'mid(L9-24)':list(range(8,24)),'deep(L25-32)':list(range(24,32))}
CONDS=[c for c in sys.argv[1:]] or ['dllm_cache','dar','ctar','dar_ctar']
REGION=os.environ.get('F1C_REGION','dllm_cache')   # plain-cache run that defines the repeat region
OUTJ=os.environ.get('F1C_OUT','f1_components.json')

def dq(d): return (d['quantized_attentions'].astype(np.float32)-float(d['zero_point']))*float(d['scale'])

def scan_meta(dirp):
    us={}; last={i:None for i in range(GEN)}; dl={}; fin=None
    for t in range(GEN):
        f=os.path.join(dirp,'step_%d.npz'%t)
        if not os.path.exists(f): break
        d=np.load(f)
        pl=int(d['prompt_length']); ti=np.asarray(d['transfer_index']).reshape(-1)
        rf=np.asarray(d['refresh_q_index']).reshape(-1)
        for j in rf:
            j=int(j)
            if 0<=j<GEN: last[j]=t
        for i in (np.where(ti)[0]-pl):
            i=int(i)
            if 0<=i<GEN and i not in us: us[i]=t; dl[i]=(t-last[i]) if last[i] is not None else t
        if t==GEN-1: fin=np.asarray(d['token_ids']).reshape(-1)[pl:][:GEN]
    return us,dl,fin

def metrics_for(dirp,us,bands):
    out={b:{} for b in bands}; by_step={}
    for i,t in us.items(): by_step.setdefault(t,[]).append(i)
    for t,ids in sorted(by_step.items()):
        f=os.path.join(dirp,'step_%d.npz'%t)
        if not os.path.exists(f): continue
        A=dq(np.load(f))
        for bname,lids in bands.items():
            B=np.mean([A[L,0].mean(0) for L in lids],axis=0)[-GEN:,-GEN:]
            for i in ids:
                r=B[i]; s=r.sum()
                if s<=0: continue
                r=r/s; lo,hi=max(0,i-5),min(GEN,i+6)
                out[bname][i]=float(r[lo:hi].sum())
        del A
    return out

avail=[c for c in CONDS if os.path.isdir(os.path.join(ROOT,c))]
stems=set(os.listdir(os.path.join(ROOT,'baseline'))) & set(os.listdir(os.path.join(ROOT,REGION)))
for c in avail: stems &= set(os.listdir(os.path.join(ROOT,c)))
stems=sorted(stems)
print('conditions: %s | paired images: %d'%(avail,len(stems)))

acc={c:{b:{'reg':[],'far':[]} for b in BANDS} for c in avail}
for k,stem in enumerate(stems,1):
    bdir=os.path.join(ROOT,'baseline',stem)
    try:
        usb,_,_=scan_meta(bdir)
        usc,_,finc=scan_meta(os.path.join(ROOT,REGION,stem))
    except Exception as e:
        print('  skip %s (%s)'%(stem,e)); continue
    if finc is None: continue
    reps=[i for i in range(1,GEN) if finc[i]==finc[i-1]]
    region=set()
    for i in reps: region.update(range(max(0,i-5),min(GEN,i+6)))
    mb=metrics_for(bdir,usb,BANDS)
    for c in avail:
        cdir=os.path.join(ROOT,c,stem)
        try: usx,_,_=scan_meta(cdir)
        except Exception: continue
        mx=metrics_for(cdir,usx,BANDS)
        for b in BANDS:
            for i in range(1,GEN):
                if i not in mx[b] or i not in mb[b]: continue
                d=mx[b][i]-mb[b][i]
                (acc[c][b]['reg'] if i in region else acc[c][b]['far']).append(d)
    print('[%d/%d] %s repeats=%d'%(k,len(stems),stem,len(reps)),flush=True)

res={}
for b in BANDS:
    print('\n=== %s ==='%b)
    print('  %-12s %10s %10s %10s %12s'%('condition','d_w5 REGION','d_w5 FAR','gap','n_region'))
    for c in avail:
        a=np.array(acc[c][b]['reg']); f=np.array(acc[c][b]['far'])
        if len(a)<5 or len(f)<5: continue
        p=float(sps.mannwhitneyu(a,f,alternative='less').pvalue)
        t=float(sps.ttest_1samp(a,0.0).pvalue)
        print('  %-12s %+10.4f %+10.4f %+10.4f %12d   region-vs-0 p=%.2g  region-vs-far p=%.2g'
              %(c,a.mean(),f.mean(),a.mean()-f.mean(),len(a),t,p))
        res.setdefault(b,{})[c]={'d_w5_region':float(a.mean()),'d_w5_far':float(f.mean()),
                                 'n_region':int(len(a)),'n_far':int(len(f)),
                                 'p_region_vs_zero':t,'p_region_vs_far':p}
out='/data/zhaoqiyan/autodl-tmp/information_flow/results/F1/'+OUTJ
os.makedirs(os.path.dirname(out),exist_ok=True)
json.dump(res,open(out,'w'),indent=1)
print('\nsaved %s'%out)
