"""Restart the scheduler without disturbing the jobs it already launched.

Its children survive it, but a fresh scheduler would relaunch their lines and two processes would
write the same outputs.jsonl. So the lines of the jobs that are still running are parked in
jobs/ch6_resume.parked and put back once their summary.json exists.
"""
import os, re, subprocess, signal, time
E = "/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval"
JF, PARK = f"{E}/jobs/ch6_resume.txt", f"{E}/jobs/ch6_resume.parked"

ps = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
running_out = set(re.findall(r"--out \S*?results/(\S+)", ps))
# lmms-eval jobs carry no --out: bench_eval.sh <config> <task> writes results/bench/<config>/<task>
running_out |= {f"bench/{c}/{t}" for c, t in re.findall(r"bench_eval\.sh (\S+) (\S+)", ps)}
lines = open(JF).read().rstrip("\n").split("\n")
def marker(l):
    parts = l.split(None, 2)
    return re.sub(r"/(summary\.json|DONE|answers\.jsonl)$", "", parts[1]) if len(parts) > 2 else None
parked = [l for l in lines if marker(l) in running_out]
keep = [l for l in lines if l not in parked]
open(JF, "w").write("\n".join(keep) + "\n")
with open(PARK, "a") as f:
    for l in parked:
        f.write(l + "\n")
print("parked while they finish:", [l.split()[0] for l in parked])

sch = subprocess.run(["pgrep", "-f", "sched_cmd.py"], capture_output=True, text=True).stdout.split()
for pid in sch:
    try:
        os.kill(int(pid), signal.SIGTERM); print("stopped scheduler", pid)
    except OSError:
        pass
time.sleep(3)
# defaults: one job per GPU, admit a GPU only while it holds < 4 GB. Override on the command line, e.g.
#   COTA_MAXPER=1 COTA_FREE_MIB=40000 python scripts/restart_sched.py   (admits GPU 5 next to a colleague's 36 GB job)
os.environ.setdefault("COTA_MAXPER", "1"); os.environ.setdefault("COTA_FREE_MIB", "4000")
log = open(f"{E}/logs/ch6_resume_driver.log", "a")
p = subprocess.Popen(["/data/zhaoqiyan/miniconda3/envs/llada-v/bin/python", "-u",
                      f"{E}/scripts/sched_cmd.py", f"{E}/jobs/ch6_resume.txt", "ch6_resume"],
                     cwd=E, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                     start_new_session=True, env=dict(os.environ))
print("scheduler restarted, pid", p.pid)
