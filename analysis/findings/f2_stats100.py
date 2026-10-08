#!/usr/bin/env python
'''Every number of Sec. IV-C from the per-position table of f2f3_positions100.py.
Intervals resample IMAGES and every test is taken over images.   usage: f2_stats100.py <positions.npz> <out.json>'''
import numpy as np, json, sys
from scipy import stats as sps
z=np.load(sys.argv[1],allow_pickle=True); R=z['data']; c={k:i for i,k in enumerate(z['cols'])}
TAU=3; NB=5000; rng=np.random.default_rng(0)
img=R[:,c['img']].astype(int); rep=R[:,c['rep_con']]==1; ok=np.ones(len(R),bool)   # all suffix positions 1..127, as in the F1 table
K=[i for i in np.unique(img) if rep[(img==i)&ok].any()]
sel=np.isin(img,K)&ok; P=R[sel]; pim=img[sel]; prep=rep[sel]; D=P[:,c['delta']]; st=P[:,c['step']]
idx={i:np.where(pim==i)[0] for i in K}; DR=[np.concatenate([idx[i] for i in rng.choice(K,len(K))]) for _ in range(NB)]
boot=lambda fn: [float(np.nanpercentile([fn(r) for r in DR],q)) for q in (2.5,97.5)]
print(f"images analysed {len(np.unique(img))}, with a repeat {len(K)};  repeat positions {prep.sum()}, normal {(~prep).sum()}")
res=dict(n_images=int(len(np.unique(img))),n_images_repeat=int(len(K)),n_repeat=int(prep.sum()),n_normal=int((~prep).sum()),tau=TAU)
for nm,m in (('repeat',prep),('normal',~prep)):
    a=D[m]; res[nm]=dict(median=float(np.median(a)),p25=float(np.percentile(a,25)),p75=float(np.percentile(a,75)),mean=float(a.mean()),max=int(a.max()),
                         frac_gt_tau=float((a>TAU).mean()),frac_zero=float((a==0).mean()),hist=[float((a==k).mean()) for k in range(int(D.max())+1)])
    print(f"  {nm:7s} median {res[nm]['median']:.0f}  IQR [{res[nm]['p25']:.0f},{res[nm]['p75']:.0f}]  mean {res[nm]['mean']:.2f}  share beyond tau {100*res[nm]['frac_gt_tau']:.1f}%  share at 0 {100*res[nm]['frac_zero']:.1f}%")
f_gap=lambda r: (D[r][prep[r]]>TAU).mean()-(D[r][~prep[r]]>TAU).mean(); f_mean=lambda r: D[r][prep[r]].mean()-D[r][~prep[r]].mean()
allr=np.arange(len(P))
pr=np.array([(D[idx[i]][prep[idx[i]]]>TAU).mean() for i in K]); pn=np.array([(D[idx[i]][~prep[idx[i]]]>TAU).mean() for i in K])
mr=np.array([D[idx[i]][prep[idx[i]]].mean() for i in K]); mn=np.array([D[idx[i]][~prep[idx[i]]].mean() for i in K])
res.update(gap_frac=float(f_gap(allr)),gap_frac_ci=boot(f_gap),gap_mean=float(f_mean(allr)),gap_mean_ci=boot(f_mean),
    images_repeat_staler_frac=int((pr>pn).sum()),wilcoxon_frac_p=float(sps.wilcoxon(pr,pn,alternative='greater').pvalue),
    images_repeat_staler_mean=int((mr>mn).sum()),wilcoxon_mean_p=float(sps.wilcoxon(mr,mn,alternative='greater').pvalue),
    KS_D=float(sps.ks_2samp(D[prep],D[~prep]).statistic),MWU_position_level_p=float(sps.mannwhitneyu(D[prep],D[~prep],alternative='greater').pvalue))
print(f"  share beyond tau: gap {100*res['gap_frac']:+.1f} points, 95% CI [{100*res['gap_frac_ci'][0]:+.1f}, {100*res['gap_frac_ci'][1]:+.1f}]; repeat staler in {res['images_repeat_staler_frac']}/{len(K)} images, Wilcoxon p={res['wilcoxon_frac_p']:.2e}")
print(f"  mean staleness: gap {res['gap_mean']:+.2f} steps, 95% CI [{res['gap_mean_ci'][0]:+.2f}, {res['gap_mean_ci'][1]:+.2f}]; repeat staler in {res['images_repeat_staler_mean']}/{len(K)} images, Wilcoxon p={res['wilcoxon_mean_p']:.2e}")
print(f"  KS D={res['KS_D']:.3f}  (position-level MWU p={res['MWU_position_level_p']:.1e})")
res['quarters']={}
for lo,hi in ((0,32),(32,64),(64,96),(96,128)):
    m=(st>=lo)&(st<hi); a=m&prep; b=m&(~prep)
    q=dict(n_rep=int(a.sum()),n_norm=int(b.sum()),frac_rep=float((D[a]>TAU).mean()) if a.sum() else None,frac_norm=float((D[b]>TAU).mean()),
           share_of_repeats=float(a.sum()/prep.sum()))
    ii=[i for i in K if (a[idx[i]]).any() and (b[idx[i]]).any()]
    if len(ii)>5:
        x=np.array([(D[idx[i]][a[idx[i]]]>TAU).mean() for i in ii]); y=np.array([(D[idx[i]][b[idx[i]]]>TAU).mean() for i in ii])
        nz=(x-y)!=0
        q.update(n_images=len(ii),images_repeat_staler=int((x>y).sum()),wilcoxon_p=float(sps.wilcoxon(x[nz],y[nz],alternative='greater').pvalue) if nz.sum()>5 else None)
    res['quarters'][f"{lo}-{hi-1}"]=q
    print(f"  steps {lo:3d}-{hi-1:3d}: repeats {q['n_rep']:4d} ({100*q['share_of_repeats']:.1f}% of all), normal {q['n_norm']:4d} | beyond tau: repeat {q['frac_rep'] if q['frac_rep'] is None else round(q['frac_rep'],2)} normal {q['frac_norm']:.2f} | images {q.get('images_repeat_staler')}/{q.get('n_images')} p={q.get('wilcoxon_p')}")
json.dump(res,open(sys.argv[2],'w'),indent=1); print("saved",sys.argv[2])
