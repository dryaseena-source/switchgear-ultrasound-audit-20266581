import sys, json
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import train_models as R
import pandas as pd, numpy as np, torch
from sklearn.metrics import confusion_matrix
ART=R.ART
raw,y,ids,g,X=R.load('frequency')
au=pd.read_csv(ART/'record_identity_audit.csv')
q=au[au.domain=='frequency'].set_index('record_id').loc[ids]
keep=~q['cross_label_conflict'].astype(bool).to_numpy()
orig=np.where(keep)[0]
y2=y[keep]; g2=g[keep]; X2=X[keep]; ids2=ids[keep]
rows=[]
for r,s,f,tr,va,te in R.outer(y2,g2):
 seed=5000+r*100+f; Xt,Xv,Xe=R.scale(X2,tr,va,te); m=R.build('CNN-GRU',Xt.shape[2],'frequency'); ep,sec,w=R.train(m,Xt,y2[tr],Xv,y2[va],seed); m.eval()
 with torch.inference_mode(): prob=torch.softmax(m(torch.from_numpy(Xe)),1).numpy(); pr=prob.argmax(1)
 mm=R.mets(y2[te],pr); rows.append(dict(domain='frequency',policy='exclude_exact_cross_label_conflict_groups',repeat=r,outer_seed=s,fold=f,test_n=len(te),test_arc_n=int(y2[te].sum()),confusion_matrix=json.dumps(confusion_matrix(y2[te],pr,labels=[0,1]).tolist()),**mm)); print(r,f,mm,flush=True)
pd.DataFrame(rows).to_csv(ART/'frequency_conflict_exclusion_cnn_gru.csv',index=False)
print(pd.DataFrame(rows)[['accuracy','macro_f1','arc_recall']].agg(['mean','std']))
