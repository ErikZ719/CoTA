#!/usr/bin/env python
'''Every number of Sec. IV-B (paragraphs 2 and 3) from the per-position table of f1_positions100.py.
All intervals resample IMAGES (cluster bootstrap), and every test is also taken over images, because
positions of one image are not independent.  usage: f1_stats100.py <positions.npz> [con|raw] [out.json]'''
import numpy as np, json, sys
from scipy import stats as sps
src=sys.argv[1]; CRIT=sys.argv[2] if len(sys.argv)>2 else 'con'; out=sys.argv[3] if len(sys.argv)>3 else None
z=np.load(src,allow_pickle=True); R=z['data']; c={k:i for i,k in enumerate(z['cols'])}
BANDS=['shallow','mid','deep']; NB=5000; rng=np.random.default_rng(0)
img=R[:,c['img']].astype(int); reg=R[:,c['reg_'+CRIT]]==1
ids_all=np.unique(img); K=[i for i in ids_all if reg[img==i].any() and (~reg[img==i]).any()]
print(f"criterion={CRIT}   images analysed={len(ids_all)}   with a repeat (enter the comparison)={len(K)}")
res=dict(criterion=CRIT,n_images=int(len(ids_all)),n_images_repeat=int(len(K)))
sel=np.isin(img,K); P=R[sel]; pim=img[sel]; preg=reg[sel]
res.update(n_region=int(preg.sum()),n_distant=int((~preg).sum()))
print(f"positions: repeat regions {preg.sum()}, distant {(~preg).sum()}")
idx={i:np.where(pim==i)[0] for i in K}
DRAWS=[np.concatenate([idx[i] for i in rng.choice(K,len(K))]) for _ in range(NB)]   # the same image resamples for every statistic
def boot(fn):
    v=np.array([fn(r) for r in DRAWS],dtype=float); return [float(np.nanpercentile(v,2.5)),float(np.nanpercentile(v,97.5))]
allrows=np.arange(len(P))
for b in BANDS:
    d=P[:,c['w5c_'+b]]-P[:,c['w5b_'+b]]; wb=P[:,c['w5b_'+b]]
    f_reg=lambda r: d[r][preg[r]].mean(); f_far=lambda r: d[r][~preg[r]].mean()
    f_dif=lambda r: f_reg(r)-f_far(r); f_rel=lambda r: -d[r][preg[r]].mean()/wb[r][preg[r]].mean()
    pr=np.array([d[idx[i]][preg[idx[i]]].mean() for i in K]); pf=np.array([d[idx[i]][~preg[idx[i]]].mean() for i in K])
    w=sps.wilcoxon(pr,pf,alternative='less'); mw=sps.mannwhitneyu(d[preg],d[~preg],alternative='less')
    q={}
    for lo,hi in ((0,32),(32,64),(64,96),(96,128)):
        m=(P[:,c['step']]>=lo)&(P[:,c['step']]<hi)
        q[f"{lo}-{hi-1}"]=dict(n=int(m.sum()),d=float(d[m].mean()),ci=boot(lambda r,m=m: d[r][m[r]].mean()))
    q1=np.array([d[idx[i]][P[idx[i],c['step']]<32].mean() for i in K]); q4=np.array([d[idx[i]][P[idx[i],c['step']]>=96].mean() for i in K])
    wq=sps.wilcoxon(q4,q1,alternative='less')
    res[b]=dict(region=float(f_reg(allrows)),region_ci=boot(f_reg),distant=float(f_far(allrows)),distant_ci=boot(f_far),
                diff=float(f_dif(allrows)),diff_ci=boot(f_dif),
                images_region_loses_more=int((pr<pf).sum()),wilcoxon_p=float(w.pvalue),mwu_position_level_p=float(mw.pvalue),
                vanilla_w5_region=float(wb[preg].mean()),rel_loss_region=float(f_rel(allrows)),rel_loss_ci=boot(f_rel),
                quarters=q,images_q4_below_q1=int((q4<q1).sum()),wilcoxon_q4_vs_q1_p=float(wq.pvalue))
    r=res[b]
    print(f"\n[{b}] region {r['region']:+.4f} {r['region_ci']}  distant {r['distant']:+.4f} {r['distant_ci']}")
    print(f"   difference {r['diff']:+.4f}  95% CI {r['diff_ci']}   images where region loses more {r['images_region_loses_more']}/{len(K)}  Wilcoxon p={r['wilcoxon_p']:.2e}   (position-level MWU p={r['mwu_position_level_p']:.1e})")
    print(f"   vanilla w5 in regions {r['vanilla_w5_region']:.3f}  relative loss {100*r['rel_loss_region']:.1f}%  CI [{100*r['rel_loss_ci'][0]:.1f}, {100*r['rel_loss_ci'][1]:.1f}]%")
    print("   quarters: "+"  ".join(f"{k}: {v['d']:+.4f}" for k,v in q.items())+f"   images with Q4<Q1: {r['images_q4_below_q1']}/{len(K)}  p={r['wilcoxon_q4_vs_q1_p']:.2e}")
# ---- missed anchors (panel d), deep band
d=P[:,c['w5c_deep']]-P[:,c['w5b_deep']]; late=P[:,c['late']]
sp=sps.spearmanr(late,d); f_rho=lambda r: sps.spearmanr(late[r],d[r])[0]
dose={}
for nm,mk in (('all',np.ones(len(P),bool)),('region',preg),('distant',~preg)):
    dose[nm]={}
    for k in (0,1,2,3,4):
        m=mk&((late==k) if k<4 else (late>=4))
        if m.sum()>5: dose[nm][str(k) if k<4 else '>=4']=dict(n=int(m.sum()),d=float(d[m].mean()),ci=boot(lambda r,m=m: d[r][m[r]].mean() if m[r].sum()>0 else np.nan))
lr=np.array([late[idx[i]][preg[idx[i]]].mean() for i in K]); lf=np.array([late[idx[i]][~preg[idx[i]]].mean() for i in K])
wl=sps.wilcoxon(lr,lf,alternative='greater')
rhos=np.array([sps.spearmanr(late[idx[i]],d[idx[i]])[0] for i in K]); rhos=rhos[np.isfinite(rhos)]
res['missed_anchors']=dict(spearman_rho=float(sp[0]),spearman_p_position_level=float(sp[1]),rho_ci=boot(f_rho),
     images_negative_rho=int((rhos<0).sum()),n_images_rho=int(len(rhos)),wilcoxon_rho_p=float(sps.wilcoxon(rhos,alternative='less').pvalue),
     mean_missed_region=float(late[preg].mean()),mean_missed_distant=float(late[~preg].mean()),
     images_region_misses_more=int((lr>lf).sum()),wilcoxon_missed_p=float(wl.pvalue),by_count=dose)
m=res['missed_anchors']
print(f"\n[missed anchors, deep band] Spearman rho={m['spearman_rho']:+.3f} CI {m['rho_ci']}  (position-level p={m['spearman_p_position_level']:.1e}); per-image rho<0 in {m['images_negative_rho']}/{m['n_images_rho']}, Wilcoxon p={m['wilcoxon_rho_p']:.2e}")
print(f"   mean missed: region {m['mean_missed_region']:.2f} vs distant {m['mean_missed_distant']:.2f}; region misses more in {m['images_region_misses_more']}/{len(K)} images, Wilcoxon p={m['wilcoxon_missed_p']:.2e}")
for nm in ('all','region','distant'): print(f"   {nm:8s} "+"  ".join(f"{k}: {v['d']:+.4f} (n={v['n']})" for k,v in dose[nm].items()))
if out: json.dump(res,open(out,'w'),indent=1); print("\nsaved",out)
