#!/usr/bin/env python
'''F2 / F3 on the 100-image sample: one row per suffix position, nothing aggregated. Runs on the server.

Source: information_flow/multisample/dllm_cache_G128_<stem>.npz and baseline_G128_<stem>.npz
(delta [steps,GEN], is_masked [steps,GEN], gen_ids [GEN], entropy_bits [steps,33,GEN]).
Each position is read at ITS OWN decision step, the last step at which it is still masked, as in
f2_staleness_multisample.py. Keeps the image index, so that statistics can be taken over images.
Columns: img pos step delta rep_raw rep_con in_resp  ent_00..ent_32 (cached run, decision step)
         bstep  bent_00..bent_32 (vanilla run, its own decision step)
'''
import numpy as np, os, json
IF='/data/zhaoqiyan/autodl-tmp/information_flow'; ROOT=IF+'/multisample'
OUT=os.environ.get('F2P_OUT',IF+'/results/F2/f2f3_100_positions.npz')
EOT={126081,126348}; LAYOUT={198,220,197,256,262144}
def decision(M):
    us={}
    for i in range(M.shape[1]):
        m=np.where(M[:,i])[0]
        if len(m): us[i]=int(m[-1])
    return us
def repeats(g):
    G=len(g); raw=[i for i in range(1,G) if g[i]==g[i-1]]
    cut=next((k for k,x in enumerate(g) if int(x) in EOT),G)
    seq=[(k,int(x)) for k,x in enumerate(g[:cut]) if int(x) not in LAYOUT]
    con=[seq[k][0] for k in range(1,len(seq)) if seq[k][1]==seq[k-1][1]]
    return set(raw),set(con),cut
files=json.load(open(IF+'/coco100_seed0.json'))['files']
rows=[]; info=[]
for k,f in enumerate(files):
    stem=os.path.splitext(os.path.basename(f))[0]
    fc=os.path.join(ROOT,'dllm_cache_G128_%s.npz'%stem); fb=os.path.join(ROOT,'baseline_G128_%s.npz'%stem)
    if not (os.path.exists(fc) and os.path.exists(fb)): print('missing',stem); continue
    c=np.load(fc); b=np.load(fb)
    D=c['delta'].astype(int); M=c['is_masked']; g=c['gen_ids']; E=c['entropy_bits']
    Mb=b['is_masked']; Eb=b['entropy_bits']; gb=b['gen_ids']
    # gen_ids drops the closing end-of-text token, the recordings keep its position: put it back so that
    # the positions are the same 1..127 as in the F1 table (the end-of-text position is never a repeat)
    if len(g)<D.shape[1]:  g=np.concatenate([np.asarray(g).reshape(-1),[126348]*(D.shape[1]-len(g))])
    if len(gb)<D.shape[1]: gb=np.concatenate([np.asarray(gb).reshape(-1),[126348]*(D.shape[1]-len(gb))])
    G=min(D.shape[1],len(g),M.shape[1]); us=decision(M[:,:G]); ub=decision(Mb[:,:G])
    raw,con,cut=repeats(g[:G]); braw,bcon,_=repeats(gb[:G])
    info.append((stem,len(raw),len(con),cut,len(bcon)))
    for i in range(1,G):
        if i not in us or i not in ub: continue
        t=us[i]; tb=ub[i]
        rows.append([k,i,t,int(D[t,i]),int(i in raw),int(i in con),int(i<cut)]+[float(x) for x in E[t,:,i]]+[tb]+[float(x) for x in Eb[tb,:,i]])
R=np.array(rows,dtype=np.float64); L=E.shape[1]
cols=['img','pos','step','delta','rep_raw','rep_con','in_resp']+['ent_%02d'%l for l in range(L)]+['bstep']+['bent_%02d'%l for l in range(L)]
os.makedirs(os.path.dirname(OUT),exist_ok=True)
np.savez_compressed(OUT,data=R,cols=np.array(cols),images=np.array([x[0] for x in info]),
    n_rep_raw=np.array([x[1] for x in info]),n_rep_con=np.array([x[2] for x in info]),resp_len=np.array([x[3] for x in info]),
    n_rep_con_vanilla=np.array([x[4] for x in info]))
print('images %d   positions %d   layers %d'%(len(info),len(R),L))
print('images with a repeat under the cache: raw %d, content %d ; vanilla images with a repeat (content): %d'%(
    sum(x[1]>0 for x in info),sum(x[2]>0 for x in info),sum(x[4]>0 for x in info)))
# do the two recordings describe the same responses? compare with the attention recordings of F1
same=0; n=0
for stem,_,_,_,_ in info:
    p=IF+'/attn_multi100/dllm_cache/%s/step_127.npz'%stem
    if not os.path.exists(p): continue
    d=np.load(p); pl=int(d['prompt_length']); a=np.asarray(d['token_ids']).reshape(-1)[pl:][:128]
    g=np.load(os.path.join(ROOT,'dllm_cache_G128_%s.npz'%stem))['gen_ids'][:128]; n+=1; same+=int(np.array_equal(a,g))
print('responses identical to the F1 recordings: %d of %d'%(same,n))
print('saved',OUT)
