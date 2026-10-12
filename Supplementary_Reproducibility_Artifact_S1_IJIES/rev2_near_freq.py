import sys, json
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import train_models as R
import pandas as pd, numpy as np, torch
from sklearn.metrics import confusion_matrix
ART=R.ART
raw,y,ids,g_exact,X=R.load('frequency')
au=pd.read_csv(ART/'record_identity_audit.csv')
q=au[au.domain=='frequency'].set_index('record_id').loc[ids]
g=q['near_group_0.9999'].to_numpy()
rows=[]
for r,s,f,tr,va,te in R.outer(y,g):
 seed=5000+r*100+f; Xt,Xv,Xe=R.scale(X,tr,va,te); m=R.build('CNN-GRU',Xt.shape[2],'frequency'); ep,sec,w=R.train(m,Xt,y[tr],Xv,y[va],seed); m.eval()
 with torch.inference_mode(): prob=torch.softmax(m(torch.from_numpy(Xe)),1).numpy(); pr=prob.argmax(1)
 mm=R.mets(y[te],pr); rows.append(dict(domain='frequency',grouping='abs_corr_0.9999_components',repeat=r,outer_seed=s,fold=f,test_n=len(te),test_arc_n=int(y[te].sum()),confusion_matrix=json.dumps(confusion_matrix(y[te],pr,labels=[0,1]).tolist()),**mm)); print(r,f,mm,flush=True)
pd.DataFrame(rows).to_csv(ART/'near_duplicate_grouping_frequency_cnn_gru.csv',index=False)
print(pd.DataFrame(rows)[['accuracy','macro_f1','arc_recall']].agg(['mean','std']))
