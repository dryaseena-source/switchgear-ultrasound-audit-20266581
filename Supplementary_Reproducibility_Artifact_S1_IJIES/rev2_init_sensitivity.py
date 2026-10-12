import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import train_models as R
import pandas as pd, numpy as np, torch
ART=R.ART
seeds=[11,33,55]
for domain in ['time','frequency']:
 raw,y,ids,g,X=R.load(domain)
 r,s,f,tr,va,te=next(R.outer(y,g))
 Xt,Xv,Xe=R.scale(X,tr,va,te)
 rows=[]
 for model_name in ['1D-CNN','CNN-LSTM','CNN-GRU','Light-DSCNN']:
  for seed in seeds:
   R.seed_all(seed); m=R.build(model_name,Xt.shape[2],domain); ep,sec,w=R.train(m,Xt,y[tr],Xv,y[va],seed); m.eval()
   with torch.inference_mode(): pp=torch.softmax(m(torch.from_numpy(Xe)),1).numpy(); pred=pp.argmax(1)
   mm=R.mets(y[te],pred); rows.append({'domain':domain,'outer_seed':s,'fold':f,'model':model_name,'init_seed':seed,'test_n':len(te),'test_arc_n':int(y[te].sum()),'best_epoch':ep,**mm,'params':sum(p.numel() for p in m.parameters())}); print(domain,model_name,seed,mm,flush=True)
 pd.DataFrame(rows).to_csv(ART/f'initialization_sensitivity_{domain}.csv',index=False)
