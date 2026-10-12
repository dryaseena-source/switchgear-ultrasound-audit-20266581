import sys,time,torch,pandas as pd,numpy as np
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE)); import train_models as R
rows=[]
for domain in ['time','frequency']:
 raw,y,ids,g,X=R.load(domain); r,s,f,tr,va,te=next(R.outer(y,g)); Xt,Xv,Xe=R.scale(X,tr,va,te)
 for mn in ['CNN-GRU','CNN-LSTM']:
  seed=33; R.seed_all(seed); m=R.build(mn,Xt.shape[2],domain); ep,train_s,w=R.train(m,Xt,y[tr],Xv,y[va],seed); m.eval(); x=torch.from_numpy(Xe[:1])
  with torch.inference_mode():
   for _ in range(100):m(x)
   vv=[]
   for _ in range(300):
    t=time.perf_counter_ns();m(x);vv.append((time.perf_counter_ns()-t)/1e6)
  rows.append({'domain':domain,'model':mn,'outer_seed':s,'fold':f,'init_seed':seed,'best_epoch':ep,'train_seconds':train_s,'latency_median_ms':float(np.median(vv)),'latency_p95_ms':float(np.percentile(vv,95)),'params':sum(p.numel() for p in m.parameters()),'protocol':'CPU batch=1; 5 torch threads; inference_mode; 100 warm-up + 300 timed forward passes; excludes preprocessing/I-O'})
  print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv(R.ART/'timing_measurements.csv',index=False)
