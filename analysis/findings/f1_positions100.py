#!/usr/bin/env python
'''F1 on the 100-image sample: one row per suffix position, nothing aggregated.

Same validated protocol as f1_multisample.py and probes/verify_frozen2.py (cross-run same-position
pairing, each position read at ITS OWN decision step in each run, row-normalised w5 over the suffix),
but it keeps the image index of every position so that statistics can be taken over images, and it
runs the images in parallel. Runs on the server.

Env: F1P_ROOT (default attn_multi100)  F1P_CACHE (dllm_cache)  F1P_BASE (baseline)  F1P_REGION (= F1P_CACHE)
     F1P_OUT (results/F1/f1_100_positions.npz)  F1P_WORKERS (24)
'''
import numpy as np, os, json, sys
from multiprocessing import Pool
IF='/data/zhaoqiyan/autodl-tmp/information_flow'
ROOT=os.environ.get('F1P_ROOT',IF+'/attn_multi100')
CACHE=os.environ.get('F1P_CACHE','dllm_cache'); BASE=os.environ.get('F1P_BASE','baseline')
REGION=os.environ.get('F1P_REGION',CACHE)
OUT=os.environ.get('F1P_OUT',IF+'/results/F1/f1_100_positions.npz')
GEN=128; W=5
BANDS=[('shallow',range(0,8)),('mid',range(8,24)),('deep',range(24,32))]
EOT={126081,126348}; LAYOUT={198,220,197,256,262144}

def dq(d): return (d['quantized_attentions'].astype(np.float32)-float(d['zero_point']))*float(d['scale'])

def scan(dirp):
    '''decision step, staleness at that step (union over layers), final ids'''
    us={}; last={i:None for i in range(GEN)}; age={}; fin=None
    for t in range(GEN):
        f=os.path.join(dirp,'step_%d.npz'%t)
        if not os.path.exists(f): return None
        d=np.load(f)
        pl=int(d['prompt_length']); ti=np.asarray(d['transfer_index']).reshape(-1)
        for j in np.asarray(d['refresh_q_index']).reshape(-1):
            j=int(j)
            if 0<=j<GEN: last[j]=t
        for i in (np.where(ti)[0]-pl):
            i=int(i)
            if 0<=i<GEN and i not in us:
                us[i]=t; age[i]=(t-last[i]) if last[i] is not None else t
        if t==GEN-1: fin=np.asarray(d['token_ids']).reshape(-1)[pl:][:GEN]
    return us,age,fin

def repeats(fin):
    '''(a) raw criterion of the earlier analysis; (b) the criterion of Sec. IV-A: content tokens,
    i.e. cut at the first end-of-text token and layout ids removed, second token of each equal pair'''
    raw=[i for i in range(1,GEN) if fin[i]==fin[i-1]]
    cut=next((k for k,x in enumerate(fin) if int(x) in EOT),len(fin))
    seq=[(k,int(x)) for k,x in enumerate(fin[:cut]) if int(x) not in LAYOUT]
    con=[seq[k][0] for k in range(1,len(seq)) if seq[k][1]==seq[k-1][1]]
    return raw,con,cut

def w5(dirp,us):
    out={}; by={}
    for i,t in us.items(): by.setdefault(t,[]).append(i)
    for t,ids in sorted(by.items()):
        A=dq(np.load(os.path.join(dirp,'step_%d.npz'%t)))      # [L,1,H,GEN,GEN]
        for b,(nm,lids) in enumerate(BANDS):
            B=np.mean([A[L,0].mean(0) for L in lids],axis=0)[-GEN:,-GEN:]
            for i in ids:
                r=B[i]; s=r.sum()
                if s<=0: continue
                r=r/s; lo,hi=max(0,i-W),min(GEN,i+W+1)
                out.setdefault(i,[np.nan]*3)[b]=float(r[lo:hi].sum())
        del A
    return out

def one(arg):
    k,stem=arg
    dc,db,dr=(os.path.join(ROOT,t,stem) for t in (CACHE,BASE,REGION))
    for d in (dc,db,dr):
        if not os.path.exists(os.path.join(d,'step_%d.npz'%(GEN-1))): return None
    sc,sb,sr=scan(dc),scan(db),scan(dr)
    if sc is None or sb is None or sr is None: return None
    usc,agec,_=sc; usb,_,_=sb
    raw,con,cut=repeats(sr[2])
    reg_raw=set(j for i in raw for j in range(max(0,i-5),min(GEN,i+6)))
    reg_con=set(j for i in con for j in range(max(0,i-5),min(GEN,i+6)))
    wc,wb=w5(dc,usc),w5(db,usb)
    rows=[]
    for i in range(1,GEN):
        if i not in wc or i not in wb: continue
        t=usc[i]; a=agec.get(i,0); lo,hi=max(0,i-W),min(GEN,i+W+1)
        late=sum(1 for j in range(lo,hi) if j!=i and t-a < usc.get(j,10**9) <= t)
        rows.append([k,i,t,a,late,int(i in reg_raw),int(i in reg_con),int(i in raw),int(i in con),int(i<cut)]+wc[i]+wb[i])
    return stem,len(raw),len(con),cut,rows

if __name__=='__main__':
    files=json.load(open(IF+'/coco100_seed0.json'))['files']
    stems=[os.path.splitext(os.path.basename(f))[0] for f in files]
    with Pool(int(os.environ.get('F1P_WORKERS','24'))) as p: res=p.map(one,list(enumerate(stems)),chunksize=1)
    ok=[r for r in res if r is not None]
    R=np.array([x for r in ok for x in r[4]],dtype=np.float64)
    cols=['img','pos','step','age','late','reg_raw','reg_con','rep_raw','rep_con','in_resp',
          'w5c_shallow','w5c_mid','w5c_deep','w5b_shallow','w5b_mid','w5b_deep']
    os.makedirs(os.path.dirname(OUT),exist_ok=True)
    np.savez_compressed(OUT,data=R,cols=np.array(cols),
        images=np.array([r[0] for r in ok]),n_rep_raw=np.array([r[1] for r in ok]),
        n_rep_con=np.array([r[2] for r in ok]),resp_len=np.array([r[3] for r in ok]),
        tags=np.array([CACHE,BASE,REGION]))
    print('images analysed: %d of %d listed   positions: %d'%(len(ok),len(stems),len(R)))
    print('images with a repeat: raw criterion %d, content-token criterion %d'%(sum(r[1]>0 for r in ok),sum(r[2]>0 for r in ok)))
    print('saved',OUT)
