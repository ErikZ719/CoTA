#!/usr/bin/env python
'''Section IV-E and its appendix table: how the three findings relate, on the 94 of 100 images that repeat under dLLM-Cache.
Joins the per-position tables of F1, F2 and F3 (same positions, same order). Everything is tested over images.
Signatures of a position:  F2 stale at the decode moment (Delta > 3) ; F3 context entropy in the top fifth ;
                           F1 deep-band anchoring change in the lowest fifth (largest loss).
Run with /opt/anaconda3/bin/python.'''
import numpy as np, json
from scipy import stats
R='/Users/zhaoqiyan/Desktop/TPAMI-CoTA++/information_flow/results/'
a=np.load(R+'F1/f1_100_positions.npz',allow_pickle=True); A=a['data']; ac=list(a['cols'])
b=np.load(R+'F2/f2f3_100_positions.npz',allow_pickle=True); B=b['data']; bc=list(b['cols'])
f=np.load(R+'F3/f3_100_positions.npz',allow_pickle=True); F=f['data']; fc=list(f['cols'])
ca=lambda k:A[:,ac.index(k)]; cb=lambda k:B[:,bc.index(k)]; cf=lambda k:F[:,fc.index(k)]
assert (ca('img')==cb('img')).all() and (ca('pos')==cb('pos')).all() and (cf('pos')==cb('pos')).all() and (ca('age')==cb('delta')).all()
img=cb('img').astype(int); rep=cb('rep_con').astype(int); dl=cb('delta').astype(int); reg=ca('reg_con').astype(int)
nrep=b['n_rep_con']; keep=nrep[img]>0; ids=np.where(nrep>0)[0]
dw={k:ca('w5c_'+k)-ca('w5b_'+k) for k in ('shallow','mid','deep')}
ctx=cf('ctx_dec'); FIN=F[:,fc.index('fin_00'):fc.index('fin_32')+1]
pk=FIN[:,1:26].max(1); below=FIN[:,1:33]<(0.5*pk)[:,None]; fl=np.full(len(FIN),33); m_=below.any(1); fl[m_]=below[m_].argmax(1)+1; unc=(fl>=31).astype(float)
rng=np.random.default_rng(0); out={}
print('== (a) anchoring change by staleness')
out['a']={}
for k in ('shallow','mid','deep'):
    d=dw[k]; g0=keep&(dl==0); g1=keep&(dl>=1)&(dl<=3); g2=keep&(dl>3)
    rho=np.array([stats.spearmanr(dl[img==i],d[img==i])[0] for i in ids]); rho=rho[np.isfinite(rho)]
    pr=[];pf=[]
    for i in ids:
        g=(img==i)&(dl==0)
        if (g&(reg==1)).sum() and (g&(reg==0)).sum(): pr.append(d[g&(reg==1)].mean()); pf.append(d[g&(reg==0)].mean())
    pr=np.array(pr);pf=np.array(pf)
    out['a'][k]=dict(d0=d[g0].mean(),d13=d[g1].mean(),d46=d[g2].mean(),images=int((rho<0).sum()),n=len(rho),p=float(stats.wilcoxon(rho)[1]),
        fresh_region=d[g0&(reg==1)].mean(),fresh_distant=d[g0&(reg==0)].mean(),fresh_images=int((pr<pf).sum()),fresh_n=len(pr),fresh_p=float(stats.wilcoxon(pr-pf)[1]))
    o=out['a'][k]; print('  %-8s Delta=0 %+.3f  1-3 %+.3f  >3 %+.3f | loss grows with staleness in %d/%d images p=%.1e | at Delta=0: repeat regions %+.3f vs distant %+.3f, %d/%d images p=%.1e'%(k,o['d0'],o['d13'],o['d46'],o['images'],o['n'],o['p'],o['fresh_region'],o['fresh_distant'],o['fresh_images'],o['fresh_n'],o['fresh_p']))
print('== (b) consolidation at equal staleness (end of generation, entropy halved not before layer 31)')
r=keep&(rep==1); n=keep&(rep==0)
wr=np.bincount(dl[r],minlength=7).astype(float); yn=np.array([unc[n&(dl==v)].mean() for v in range(7)])
matched=(yn*wr).sum()/wr.sum()
per=[]
for i in ids:
    g=img==i; num=0;den=0
    for v in range(7):
        a_=g&(rep==1)&(dl==v); b_=g&(rep==0)&(dl==v)
        if a_.sum() and b_.sum(): num+=a_.sum()*(unc[a_].mean()-unc[b_].mean()); den+=a_.sum()
    if den: per.append(num/den)
per=np.array(per)
out['b']=dict(repeat=unc[r].mean(),normal=unc[n].mean(),normal_matched=matched,images=int((per>0).sum()),n=len(per),p=float(stats.wilcoxon(per)[1]))
print('  repeat %.1f%% ; normal %.1f%% ; normal at the staleness of the repeats %.1f%% ; repeat higher in %d/%d images, p=%.1e'%(100*unc[r].mean(),100*unc[n].mean(),100*matched,out['b']['images'],len(per),out['b']['p']))
print('== (c) signatures')
ok=keep&np.isfinite(ctx)
thrC=np.quantile(ctx[ok],0.8); thrD=np.quantile(dw['deep'][ok],0.2)
S=(dl>3); C=(ctx>=thrC); Rr=(dw['deep']<=thrD); k=S.astype(int)+C.astype(int)+Rr.astype(int)
rp=ok&(rep==1); nm=ok&(rep==0)
out['c']=dict(n=int(ok.sum()),n_rep=int(rp.sum()),n_norm=int(nm.sum()),
   share_rep=dict(stale=S[rp].mean(),context=C[rp].mean(),anchoring=Rr[rp].mean()),share_norm=dict(stale=S[nm].mean(),context=C[nm].mean(),anchoring=Rr[nm].mean()),
   carried_rep=[float((k[rp]==j).mean()) for j in range(4)],carried_norm=[float((k[nm]==j).mean()) for j in range(4)],
   rate=[float(rep[ok&(k==j)].mean()) for j in range(4)],npos=[int((ok&(k==j)).sum()) for j in range(4)])
idx={i:np.where(ok&(img==i))[0] for i in ids}; bs=np.zeros((2000,4))
for t in range(2000):
    m=np.concatenate([idx[i] for i in rng.choice(ids,len(ids))])
    for j in range(4):
        mm=m[k[m]==j]; bs[t,j]=rep[mm].mean() if len(mm) else np.nan
out['c']['rate_ci']=[[float(np.nanpercentile(bs[:,j],2.5)),float(np.nanpercentile(bs[:,j],97.5))] for j in range(4)]
# image level: does the rate grow with the number of signatures?
rho=[];d20=[]
for i in ids:
    g=ok&(img==i)
    r_=stats.spearmanr(k[g],rep[g])[0]
    if np.isfinite(r_): rho.append(r_)
    if (g&(k==0)).sum() and (g&(k>=2)).sum(): d20.append(rep[g&(k>=2)].mean()-rep[g&(k==0)].mean())
rho=np.array(rho); d20=np.array(d20)
out['c'].update(images_trend=int((rho>0).sum()),n_trend=len(rho),p_trend=float(stats.wilcoxon(rho)[1]),images_two_vs_none=int((d20>0).sum()),n_two_vs_none=len(d20),p_two_vs_none=float(stats.wilcoxon(d20)[1]))
c=out['c']
print('  share of repeat positions: stale %.0f%% context %.0f%% anchoring %.0f%% | normal: %.0f%% %.0f%% %.0f%%'%(100*c['share_rep']['stale'],100*c['share_rep']['context'],100*c['share_rep']['anchoring'],100*c['share_norm']['stale'],100*c['share_norm']['context'],100*c['share_norm']['anchoring']))
print('  signatures carried 0/1/2/3: repeat '+' '.join('%.0f%%'%(100*x) for x in c['carried_rep'])+' | normal '+' '.join('%.0f%%'%(100*x) for x in c['carried_norm']))
print('  repetition rate by number carried: '+' ; '.join('%d: %.1f%% [%.1f, %.1f] (n=%d)'%(j,100*c['rate'][j],100*c['rate_ci'][j][0],100*c['rate_ci'][j][1],c['npos'][j]) for j in range(4)))
print('  rate grows with the number carried in %d/%d images, p=%.1e ; two or more vs none higher in %d/%d images, p=%.1e'%(c['images_trend'],c['n_trend'],c['p_trend'],c['images_two_vs_none'],c['n_two_vs_none'],c['p_two_vs_none']))
g=ok; out['corr']=dict(stale_context=float(stats.spearmanr(dl[g],ctx[g])[0]),stale_anchoring=float(stats.spearmanr(dl[g],dw['deep'][g])[0]),context_anchoring=float(stats.spearmanr(ctx[g],dw['deep'][g])[0]))
print('  rank correlations over positions:',out['corr'])
json.dump(out,open(R+'F3/joint_stats_94.json','w'),indent=1,default=float); print('saved F3/joint_stats_94.json')
