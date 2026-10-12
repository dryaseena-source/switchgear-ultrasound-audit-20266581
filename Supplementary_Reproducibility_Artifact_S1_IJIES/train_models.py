import argparse, os, time, random, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd, scipy.io as sio, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,confusion_matrix
ART=Path(__file__).resolve().parent; ROOT=Path(os.environ.get('ARC_AUDIT_DATA_DIR', ART)); OUT=ART/'prepared'; FS=11025
OUTER_SEEDS=[2026,2027]; MAX_EPOCHS=25; PATIENCE=4; BATCH=32; LR=1e-3; torch.set_num_threads(5)

def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)
def sha(v): return hashlib.sha256(np.ascontiguousarray(v,dtype='<f8').tobytes()).hexdigest()
def load(domain):
 if domain=='time':
  arc=sio.loadmat(ROOT/'ArchingNorm.mat')['we'].astype(float); parts=[('Mechanical','MechanicalNorm2_20k.mat'),('Corona','CoronaNorm2_20k.mat'),('Tracking','TrackingNorm2_20k.mat'),('Normal','NormalNorm2_20k.mat')]
  non=np.concatenate([sio.loadmat(ROOT/f)['we'].astype(float) for _,f in parts],axis=1); raw=np.concatenate([arc,non],axis=1).T; y=np.r_[np.ones(arc.shape[1],int),np.zeros(non.shape[1],int)]; ids=np.array([f'T{i:03d}' for i in range(len(y))]); rep=np.load(OUT/'X_time_logmel.npy')
 else:
  m=sio.loadmat(ROOT/'FrequncyDomain.mat'); arc=m['x1'].astype(float); non=np.concatenate([m[k].astype(float) for k in ['x2','x3','x4','x5']],axis=1); raw=np.concatenate([arc,non],axis=1).T; y=np.r_[np.ones(arc.shape[1],int),np.zeros(non.shape[1],int)]; ids=np.array([f'F{i:03d}' for i in range(len(y))]); rep=np.load(OUT/'X_frequency_logmel.npy')
 hs=np.array([sha(v) for v in raw]); un={h:i for i,h in enumerate(sorted(set(hs)))}; groups=np.array([f'{domain[0].upper()}G{un[h]:03d}' for h in hs]); return raw,y,ids,groups,rep

def outer(y,g):
 for r,s in enumerate(OUTER_SEEDS):
  sg=StratifiedGroupKFold(5,shuffle=True,random_state=s)
  for f,(trv,te) in enumerate(sg.split(np.zeros(len(y)),y,g)):
   inn=StratifiedGroupKFold(5,shuffle=True,random_state=8000+r*100+f); trrel,va_rel=next(inn.split(np.zeros(len(trv)),y[trv],g[trv])); yield r,s,f,trv[trrel],trv[va_rel],te

def scale(X,tr,va,te):
 n,seq,feat=X.shape; sc=RobustScaler()
 if feat>1:
  sc.fit(X[tr].reshape(-1,feat)); ap=lambda ix: sc.transform(X[ix].reshape(-1,feat)).reshape(len(ix),seq,feat).astype('float32')
 else:
  sc.fit(X[tr].reshape(len(tr),-1)); ap=lambda ix: sc.transform(X[ix].reshape(len(ix),-1)).reshape(len(ix),seq,feat).astype('float32')
 return ap(tr),ap(va),ap(te)

class CNN(nn.Module):
 def __init__(self,inf,d):
  super().__init__(); c1,c2,k1,k2,dr=(64,128,3,5,.2) if d=='time' else (32,64,3,3,.25); self.n=nn.Sequential(nn.Conv1d(inf,c1,k1,padding=k1//2),nn.ReLU(),nn.MaxPool1d(2),nn.Dropout(dr),nn.Conv1d(c1,c2,k2,padding=k2//2),nn.ReLU(),nn.MaxPool1d(2),nn.Dropout(dr),nn.AdaptiveAvgPool1d(1)); self.f=nn.Linear(c2,2)
 def forward(self,x): return self.f(self.n(x.transpose(1,2)).squeeze(-1))
class GRU(nn.Module):
 def __init__(self,inf,d):
  super().__init__(); h1=128 if d=='time' else 64; self.a=nn.GRU(inf,h1,batch_first=True); self.b=nn.GRU(h1,32,batch_first=True); self.f=nn.Linear(32,2)
 def forward(self,x): z,_=self.a(x); z,_=self.b(z); return self.f(z[:,-1])
class CNNRNN(nn.Module):
 def __init__(self,inf,d,cell):
  super().__init__(); c1,c2,k1,k2,h1,dr=(64,128,3,5,128,.2) if d=='time' else (32,64,3,3,64,.25); self.c=nn.Sequential(nn.Conv1d(inf,c1,k1,padding=k1//2),nn.ReLU(),nn.MaxPool1d(2),nn.Dropout(dr),nn.Conv1d(c1,c2,k2,padding=k2//2),nn.ReLU(),nn.MaxPool1d(2),nn.Dropout(dr)); R=nn.GRU if cell=='GRU' else nn.LSTM; self.a=R(c2,h1,batch_first=True); self.b=R(h1,32,batch_first=True); self.d=nn.Linear(32,32) if d=='frequency' else None; self.f=nn.Linear(32,2)
 def forward(self,x): z=self.c(x.transpose(1,2)).transpose(1,2); z,_=self.a(z); z,_=self.b(z); z=z[:,-1]; z=torch.relu(self.d(z)) if self.d else z; return self.f(z)
class Light(nn.Module):
 def __init__(self,inf,d):
  super().__init__(); c=32 if d=='time' else 16; self.p0=nn.Conv1d(inf,c,1); self.dw1=nn.Conv1d(c,c,5,padding=2,groups=c); self.p1=nn.Conv1d(c,2*c,1); self.dw2=nn.Conv1d(2*c,2*c,3,padding=1,groups=2*c); self.p2=nn.Conv1d(2*c,2*c,1); self.pool=nn.AdaptiveAvgPool1d(1); self.f=nn.Linear(2*c,2)
 def forward(self,x): z=torch.relu(self.p0(x.transpose(1,2))); z=torch.relu(self.p1(self.dw1(z))); z=torch.relu(self.p2(self.dw2(z))); return self.f(self.pool(z).squeeze(-1))
def build(n,inf,d): return {'1D-CNN':CNN,'GRU':GRU,'CNN-GRU':lambda a,b:CNNRNN(a,b,'GRU'),'CNN-LSTM':lambda a,b:CNNRNN(a,b,'LSTM'),'Light-DSCNN':Light}[n](inf,d)
def mets(y,p): return dict(accuracy=accuracy_score(y,p),macro_f1=f1_score(y,p,average='macro',zero_division=0),arc_recall=recall_score(y,p,pos_label=1,zero_division=0),macro_precision=precision_score(y,p,average='macro',zero_division=0),macro_recall=recall_score(y,p,average='macro',zero_division=0))
def train(m,X,y,Xv,yv,seed):
 seed_all(seed); cnt=np.bincount(y,minlength=2).astype(float); w=len(y)/(2*np.maximum(cnt,1)); crit=nn.CrossEntropyLoss(weight=torch.tensor(w,dtype=torch.float32)); opt=torch.optim.Adam(m.parameters(),lr=LR); gen=torch.Generator().manual_seed(seed); dl=DataLoader(TensorDataset(torch.from_numpy(X),torch.from_numpy(y.astype('int64'))),batch_size=BATCH,shuffle=True,generator=gen); Xt=torch.from_numpy(Xv); yt=torch.from_numpy(yv.astype('int64')); best=None; bl=1e99; bad=0; bep=0; t=time.perf_counter()
 for ep in range(MAX_EPOCHS):
  m.train()
  for xb,yb in dl: opt.zero_grad(); l=crit(m(xb),yb); l.backward(); opt.step()
  m.eval();
  with torch.inference_mode(): vl=float(crit(m(Xt),yt))
  if vl<bl-1e-5: bl=vl; best={k:v.detach().clone() for k,v in m.state_dict().items()}; bad=0; bep=ep+1
  else:
   bad+=1
   if bad>=PATIENCE: break
 m.load_state_dict(best); return bep,time.perf_counter()-t,w

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--domain',required=True); ap.add_argument('--model',required=True); ap.add_argument('--init-seed',type=int); args=ap.parse_args(); raw,y,ids,g,X=load(args.domain); rows=[]; preds=[]
 for r,s,f,tr,va,te in outer(y,g):
  seed=args.init_seed if args.init_seed is not None else 5000+r*100+f; Xt,Xv,Xe=scale(X,tr,va,te); m=build(args.model,Xt.shape[2],args.domain); ep,sec,w=train(m,Xt,y[tr],Xv,y[va],seed); m.eval();
  with torch.inference_mode(): prob=torch.softmax(m(torch.from_numpy(Xe)),1).numpy(); pr=prob.argmax(1)
  mm=mets(y[te],pr); rows.append(dict(domain=args.domain,model=args.model,repeat=r,outer_seed=s,fold=f,init_seed=seed,best_epoch=ep,train_n=len(tr),val_n=len(va),test_n=len(te),test_arc_n=int(y[te].sum()),params=sum(p.numel() for p in m.parameters()),train_seconds=sec,class_weight_nonarc=w[0],class_weight_arc=w[1],confusion_matrix=json.dumps(confusion_matrix(y[te],pr,labels=[0,1]).tolist()),**mm))
  for j,i in enumerate(te): preds.append(dict(domain=args.domain,model=args.model,repeat=r,fold=f,record_id=ids[i],exact_group=g[i],label=int(y[i]),p_arc=float(prob[j,1]),prediction=int(pr[j])))
  print(args.domain,args.model,r,f,mm['accuracy'],mm['macro_f1'],flush=True)
 tag=args.model.replace('/','-').replace(' ','_'); pd.DataFrame(rows).to_csv(ART/f'fold_metrics_{args.domain}_{tag}.csv',index=False); pd.DataFrame(preds).to_csv(ART/f'fold_predictions_{args.domain}_{tag}.csv',index=False)
if __name__=='__main__': main()
