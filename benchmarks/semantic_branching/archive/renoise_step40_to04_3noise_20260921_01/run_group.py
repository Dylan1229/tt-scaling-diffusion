from pathlib import Path
import argparse,os,subprocess,json
p=argparse.ArgumentParser();p.add_argument("--gpu",required=True);p.add_argument("--seeds",nargs="+",required=True,type=int);a=p.parse_args()
root=Path(__file__).resolve().parent
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS="8",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
for seed in a.seeds:
 out=root/f"seed_{seed}"
 with (out/"run.log").open("x") as f:
  child=subprocess.Popen(["/data/datasets/peihao/tt-scaling-diffusion/.venv/bin/python","-u",str(out/"run_case.py")],
   cwd="/data/datasets/peihao/tt-scaling-diffusion",env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
  (out/"process.json").write_text(json.dumps({"pid":child.pid,"gpu":a.gpu,"noise_seed":seed})+"\n")
  rc=child.wait()
  if rc:raise RuntimeError(f"seed{seed} failed, code{rc}; inspect {out}/run.log")
 print("DONE",seed,flush=True)
