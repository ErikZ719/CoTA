#!/usr/bin/env python
"""Dynamic idle-GPU scheduler (2026-09-17, v2). Job file lines:
    <name> <done-marker path relative to E/results> <command ...>
The file is re-read before every launch, so jobs can be inserted or appended while it runs;
the first line whose name has not been started is taken next. A line 'END' marks the end of
the queue: the scheduler exits only after reaching END with nothing running (lines after END
are ignored). One job per GPU among COTA_GPUS (default 0,1,2,3), GPU idle = <4000 MiB used."""
import subprocess, sys, os, time, shlex, collections
Z="/data/zhaoqiyan"; E=Z+"/autodl-tmp/experiments/repeat_eval"
ENV=dict(os.environ, HF_HOME=Z+"/autodl-tmp/hf_cache", HUGGINGFACE_HUB_CACHE=Z+"/autodl-tmp/hf_cache/hub", HF_HUB_OFFLINE="1")
PY={"PY": Z+"/miniconda3/envs/llada-v/bin/python", "PYMM": Z+"/miniconda3/envs/mmada/bin/python"}
GPUS=[int(x) for x in os.environ.get("COTA_GPUS","0,1,2,3").split(",") if x.strip()]
# A job is admitted to a GPU only while that GPU holds less than FREE_MIB, and at most MAXPER of
# our own jobs share one GPU. A 17 GiB job therefore doubles up, a 45 GiB job keeps the card.
FREE_MIB=int(os.environ.get("COTA_FREE_MIB","25000")); MAXPER=int(os.environ.get("COTA_MAXPER","2"))
JOBF=sys.argv[1]; logdir=f"{E}/logs/{sys.argv[2] if len(sys.argv)>2 else 'sched_cmd'}"; os.makedirs(logdir,exist_ok=True)
def gpus():
    """GPU list: jobs/COTA_GPUS (re-read every call) overrides the environment."""
    f=E+"/jobs/COTA_GPUS"
    if os.path.exists(f):
        try: return [int(x) for x in open(f).read().strip().split(",") if x.strip()]
        except ValueError: pass
    return GPUS
def idle():
    G=gpus()
    out=subprocess.run(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout
    return [int(i) for i,m in (l.split(",") for l in out.strip().split("\n")) if int(i) in G and int(m)<FREE_MIB]
def _being_written(argv):
    """True if some live process already has this job's --out directory. Two writers on one
    outputs.jsonl corrupt it, and a restarted scheduler cannot know what its predecessor launched."""
    ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout.split("\n")
    sh = [i for i, a in enumerate(argv) if a.endswith("bench_eval.sh")]
    if sh and len(argv) > sh[0] + 2:                       # lmms-eval jobs: bench_eval.sh <config> <task>
        key = "bench_eval.sh " + argv[sh[0] + 1] + " " + argv[sh[0] + 2]
        root = next((a.split("=", 1)[1] for a in argv if a.startswith("BENCH_ROOT=")), None)
        return any(key in l + " " and "sched_cmd.py" not in l and ((root is None) == ("BENCH_ROOT=" not in l)) for l in ps)
    if "--out" not in argv: return False
    out = argv[argv.index("--out") + 1]
    return any(("--out " + out) in l + " " and "sched_cmd.py" not in l for l in ps)
started=set()
def pending():
    """(next job or None, END reached?)"""
    for line in open(JOBF):
        line=line.strip()
        if not line or line.startswith("#"): continue
        if line=="END": return None, True
        name,marker,cmd=line.split(None,2)
        if name in started: continue
        exe,rest=cmd.split(None,1); cwd=Z+"/autodl-tmp/LLaDA-V/train"
        if exe.startswith("CWD="):            # optional working-directory override
            cwd=exe[4:]; exe,rest=rest.split(None,1)
        return (name,marker,[PY.get(exe,exe)]+shlex.split(rest.replace("$E",E)),cwd), False
    return None, False
LOCK=JOBF+".lock"
def _alive(pid):
    try: os.kill(pid,0); return True
    except OSError: return False
if os.path.exists(LOCK):
    try: _owner=int(open(LOCK).read().strip())
    except ValueError: _owner=-1
    if _owner>0 and _owner!=os.getpid() and _alive(_owner):
        print(f"queue owned by scheduler pid {_owner}; waiting for it to finish",flush=True)
        while _alive(_owner): time.sleep(60)
        print("ALL-JOBS-COMPLETE",flush=True); sys.exit(0)
open(LOCK,"w").write(str(os.getpid()))
running=[]; print(f"dynamic scheduler on {JOBF}, GPUs {gpus()}",flush=True)
while True:
    busy=collections.Counter(g for _,_,g in running)
    for g in idle():
        if busy[g] >= MAXPER: continue
        while True:
            job,end=pending()
            if job is None: break
            name,marker,argv,cwd=job; started.add(name)
            if os.path.exists(f"{E}/results/{marker}"): print("skip(done)",name,flush=True); continue
            if _being_written(argv): print("skip(already running)",name,flush=True); continue
            break
        if job is None: break
        p=subprocess.Popen(argv,cwd=cwd,env=dict(ENV,CUDA_VISIBLE_DEVICES=str(g)),stdout=open(f"{logdir}/{name}.log","w"),stderr=subprocess.STDOUT)
        running.append((p,name,g)); busy[g]+=1; print(f"start {name} gpu{g} {time.strftime('%m-%d %H:%M')}",flush=True); time.sleep(90)
    time.sleep(30)
    for it in list(running):
        if it[0].poll() is not None: running.remove(it); print(f"done  {it[1]} rc={it[0].returncode} {time.strftime('%m-%d %H:%M')}",flush=True)
    job,end=pending()
    if job is None and end and not running: break
print("ALL-JOBS-COMPLETE",flush=True)
