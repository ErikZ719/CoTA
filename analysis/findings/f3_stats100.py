#!/usr/bin/env python
'''F3 statistics on the 94 of 100 images that repeat under dLLM-Cache. Everything is taken over images.
Input: results/F3/f3_100_positions.npz (f3_positions100.py). Run with /opt/anaconda3/bin/python.'''
import numpy as np, json
from scipy import stats
B='/Users/zhaoqiyan/Desktop/TPAMI-CoTA++/information_flow/results/F3/'
z=np.load(B+'f3_100_positions.npz',allow_pickle=True); D=z['data']; cols=list(z['cols']); c=lambda k:D[:,cols.index(k)]
img=c('img').astype(int); pos=c('pos').astype(int); step=c('step').astype(int); dl=c('delta').astype(int); rep=c('rep').astype(int)
own=c('own_dec'); ctx=c('ctx_dec'); nctx=c('n_ctx').astype(int)
FIN=D[:,cols.index('fin_00'):cols.index('fin_32')+1]; BFIN=D[:,cols.index('bfin_00'):cols.index('bfin_32')+1]
has=np.array([rep[img==k].sum()>0 for k in range(100)]); keep=has[img]; ids=np.where(has)[0]
rng=np.random.default_rng(0); out={}
print('images %d ; positions %d ; repeat %d ; normal %d'%(has.sum(),keep.sum(),(keep&(rep==1)).sum(),(keep&(rep==0)).sum()))
def auc(s,y):
    r=stats.rankdata(s); n1=y.sum(); n0=len(y)-n1
    return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
def boot(fn,nb=2000):
    v=[]
    for _ in range(nb):
        s=rng.choice(ids,len(ids)); m=np.concatenate([np.where(img==k)[0] for k in s]); v.append(fn(m))
    return np.percentile(v,[2.5,97.5])
# ---------- (1) end of generation: per-layer entropy, repeat vs normal, cached run; vanilla for reference
W=slice(26,31)
print('\n== end of generation, deep window (layers 26-30)')
fw=FIN[:,W].mean(1); bw=BFIN[:,W].mean(1)
r=keep&(rep==1); n=keep&(rep==0)
pr=np.array([fw[(img==k)&(rep==1)].mean() for k in ids]); pn=np.array([fw[(img==k)&(rep==0)].mean() for k in ids])
ci=boot(lambda m: fw[m][(rep[m]==1)].mean()-fw[m][(rep[m]==0)].mean())
print('  cached: repeat %.2f bits (median %.2f) vs normal %.2f (median %.2f) ; gap %.2f [%.2f, %.2f] ; repeat higher in %d/%d images ; Wilcoxon p=%.2e'%(
    fw[r].mean(),np.median(fw[r]),fw[n].mean(),np.median(fw[n]),fw[r].mean()-fw[n].mean(),ci[0],ci[1],(pr>pn).sum(),len(ids),stats.wilcoxon(pr-pn)[1]))
print('  vanilla run, all positions of the same images: %.2f bits (median %.2f)'%(bw[keep].mean(),np.median(bw[keep])))
out['final']={'repeat_mean':fw[r].mean(),'normal_mean':fw[n].mean(),'repeat_median':float(np.median(fw[r])),'normal_median':float(np.median(fw[n])),'gap':fw[r].mean()-fw[n].mean(),'gap_ci':list(ci),
    'images_higher':int((pr>pn).sum()),'p':float(stats.wilcoxon(pr-pn)[1]),'vanilla_mean':bw[keep].mean(),'vanilla_median':float(np.median(bw[keep]))}
print('  per layer (mean bits): layer  normal  repeat  vanilla')
for l in (1,8,16,20,24,25,26,27,28,29,30,31,32):
    print('     %2d   %6.2f  %6.2f  %6.2f'%(l,FIN[n][:,l].mean(),FIN[r][:,l].mean(),BFIN[keep][:,l].mean()))
out['final_per_layer']={'normal':[float(FIN[n][:,l].mean()) for l in range(33)],'repeat':[float(FIN[r][:,l].mean()) for l in range(33)],'vanilla':[float(BFIN[keep][:,l].mean()) for l in range(33)]}
# share of positions whose deep-window entropy at the end stays above a level
for thr in (8,10,12):
    print('  share above %d bits at the end: repeat %.3f normal %.3f'%(thr,(fw[r]>thr).mean(),(fw[n]>thr).mean()))
# ---------- (2) decode moment: context score and the risk of repeating
ok=keep&~np.isnan(ctx)
s=ctx[ok]; y=rep[ok]; im=img[ok]; st=step[ok]; d_=dl[ok]
print('\n== decode moment: mean deep-window entropy of the committed neighbours within +-5')
print('  positions %d (repeat %d) ; overall rate %.3f'%(ok.sum(),y.sum(),y.mean()))
edges=np.quantile(s,np.linspace(0,1,11)); b=np.clip(np.searchsorted(edges,s,side='right')-1,0,9)
rate=[y[b==k].mean() for k in range(10)]
cis=[]
idx_by_img={k:np.where(im==k)[0] for k in ids}
bs=np.zeros((2000,10))
for t in range(2000):
    sm=rng.choice(ids,len(ids)); m=np.concatenate([idx_by_img[k] for k in sm])
    for k in range(10):
        mk=m[b[m]==k]; bs[t,k]=y[mk].mean() if len(mk) else np.nan
lo,hi=np.nanpercentile(bs,2.5,axis=0),np.nanpercentile(bs,97.5,axis=0)
for k in range(10): print('   decile %2d: score %.2f-%.2f  n=%4d  rate %.3f  [%.3f, %.3f]'%(k+1,edges[k],edges[k+1],(b==k).sum(),rate[k],lo[k],hi[k]))
A=auc(s,y); aci=boot(lambda m: auc(ctx[m][ok[m]],rep[m][ok[m]]))
print('  AUC %.3f  95%% CI over images [%.3f, %.3f]'%(A,aci[0],aci[1]))
top=b>=8; ratio=y[top].mean()/y[~top].mean()
print('  top two deciles %.3f vs the other eight %.3f : ratio %.2f ; top decile %.3f vs rest %.3f : ratio %.2f'%(y[top].mean(),y[~top].mean(),ratio,y[b==9].mean(),y[b<9].mean(),y[b==9].mean()/y[b<9].mean()))
# image level: is the context score higher at repeat positions?
pr=[];pn=[];pa=[]
for k in ids:
    m=im==k
    if y[m].sum()==0 or (y[m]==0).sum()==0: continue
    pr.append(s[m][y[m]==1].mean()); pn.append(s[m][y[m]==0].mean()); pa.append(auc(s[m],y[m]))
pr=np.array(pr);pn=np.array(pn);pa=np.array(pa)
print('  per image: context score repeat %.2f vs normal %.2f ; higher in %d/%d images ; Wilcoxon p=%.2e ; per-image AUC mean %.3f, above 0.5 in %d images, p=%.2e'%(
    pr.mean(),pn.mean(),(pr>pn).sum(),len(pr),stats.wilcoxon(pr-pn)[1],pa.mean(),(pa>0.5).sum(),stats.wilcoxon(pa-0.5)[1]))
# image level for the top-two-deciles contrast
a_=[];b_=[]
for k in ids:
    m=im==k
    if top[m].sum()==0 or (~top[m]).sum()==0: continue
    a_.append(y[m][top[m]].mean()); b_.append(y[m][~top[m]].mean())
a_=np.array(a_);b_=np.array(b_)
print('  per image: rate in the top two deciles vs the rest: higher in %d, lower in %d, tie %d of %d images ; Wilcoxon p=%.2e'%((a_>b_).sum(),(a_<b_).sum(),(a_==b_).sum(),len(a_),stats.wilcoxon(a_-b_)[1]))
out['risk']={'n':int(ok.sum()),'n_rep':int(y.sum()),'base':float(y.mean()),'edges':list(edges),'rate':rate,'lo':list(lo),'hi':list(hi),'auc':A,'auc_ci':list(aci),
   'top2':float(y[top].mean()),'rest8':float(y[~top].mean()),'images_top2_higher':int((a_>b_).sum()),'images_n':len(a_),'p_top2':float(stats.wilcoxon(a_-b_)[1]),
   'images_score_higher':int((pr>pn).sum()),'p_score':float(stats.wilcoxon(pr-pn)[1])}
# ---------- (3) is it lateness? is it staleness?
print('\n== controls')
q=st//32
for k in range(4):
    m=q==k
    if y[m].sum()>5: print('   quarter %d: n=%4d repeats %3d  AUC %.3f ; score repeat %.2f vs normal %.2f'%(k+1,m.sum(),y[m].sum(),auc(s[m],y[m]),s[m][y[m]==1].mean(),s[m][y[m]==0].mean()))
for lab,m in (('Delta=0',d_==0),('Delta 1-3',(d_>=1)&(d_<=3)),('Delta>3',d_>3)):
    print('   %-9s: n=%4d repeats %3d  AUC %.3f'%(lab,m.sum(),y[m].sum(),auc(s[m],y[m])))
print('   score vs step: Spearman %.3f ; score vs Delta: Spearman %.3f ; n_ctx vs step %.3f'%(stats.spearmanr(s,st)[0],stats.spearmanr(s,d_)[0],stats.spearmanr(nctx[ok],st)[0]))
# quarter-matched per-image contrast of the score
per=[]
for k in ids:
    m=im==k; num=0;den=0
    for qq in range(4):
        rq=m&(q==qq)&(y==1); nq=m&(q==qq)&(y==0)
        if rq.sum() and nq.sum(): num+=rq.sum()*(s[rq].mean()-s[nq].mean()); den+=rq.sum()
    if den: per.append(num/den)
per=np.array(per); print('   quarter-matched score gap per image: mean %.2f bits, positive in %d/%d, Wilcoxon p=%.2e'%(per.mean(),(per>0).sum(),len(per),stats.wilcoxon(per)[1]))
out['controls']={'quarter_matched_gap':float(per.mean()),'images_pos':int((per>0).sum()),'images_n':len(per),'p':float(stats.wilcoxon(per)[1])}
# own entropy at the decode moment
print('\n== the position itself at its decode moment (deep window): repeat %.2f vs normal %.2f ; AUC %.3f'%(own[r].mean(),own[n].mean(),auc(own[keep],rep[keep])))
json.dump(out,open(B+'f3_100_stats.json','w'),indent=1,default=float); print('saved f3_100_stats.json')
