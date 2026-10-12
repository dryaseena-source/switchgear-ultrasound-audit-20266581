from pathlib import Path
import numpy as np, pandas as pd, scipy.io as sio, hashlib, json
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,confusion_matrix
ROOT=Path('/mnt/data'); ART=ROOT/'Reproducibility_Artifact'; FS=11025; N_TIME=20001
OUTER_SEEDS=[2026,2027]; BOOT_SEED=909; N_BOOT=5000

def sha(v): return hashlib.sha256(np.ascontiguousarray(v,dtype='<f8').tobytes()).hexdigest()
def load(domain):
 if domain=='time':
  arc=sio.loadmat(ROOT/'ArchingNorm.mat')['we'].astype(float); non=np.concatenate([sio.loadmat(ROOT/f)['we'].astype(float) for f in ['MechanicalNorm2_20k.mat','CoronaNorm2_20k.mat','TrackingNorm2_20k.mat','NormalNorm2_20k.mat']],axis=1); raw=np.concatenate([arc,non],axis=1).T; y=np.r_[np.ones(arc.shape[1],int),np.zeros(non.shape[1],int)]; ids=np.array([f'T{i:03d}' for i in range(len(y))])
 else:
  m=sio.loadmat(ROOT/'FrequncyDomain.mat'); arc=m['x1'].astype(float); non=np.concatenate([m[k].astype(float) for k in ['x2','x3','x4','x5']],axis=1); raw=np.concatenate([arc,non],axis=1).T; y=np.r_[np.ones(arc.shape[1],int),np.zeros(non.shape[1],int)]; ids=np.array([f'F{i:03d}' for i in range(len(y))])
 hs=np.array([sha(v) for v in raw]); un={h:i for i,h in enumerate(sorted(set(hs)))}; g=np.array([f'{domain[0].upper()}G{un[h]:03d}' for h in hs]); return raw,y,ids,g

def outer(y,g):
 for r,s in enumerate(OUTER_SEEDS):
  sg=StratifiedGroupKFold(5,shuffle=True,random_state=s)
  for f,(trv,te) in enumerate(sg.split(np.zeros(len(y)),y,g)):
   inn=StratifiedGroupKFold(5,shuffle=True,random_state=8000+r*100+f); trrel,va_rel=next(inn.split(np.zeros(len(trv)),y[trv],g[trv])); yield r,s,f,trv[trrel],trv[va_rel],te

def mets(y,p): return dict(accuracy=accuracy_score(y,p),macro_f1=f1_score(y,p,average='macro',zero_division=0),arc_recall=recall_score(y,p,pos_label=1,zero_division=0),macro_precision=precision_score(y,p,average='macro',zero_division=0),macro_recall=recall_score(y,p,average='macro',zero_division=0))
def features(raw,d):
 if d=='time':
  rms=np.sqrt(np.mean(raw*raw,1)); meanabs=np.mean(np.abs(raw),1); std=np.std(raw,1); p2p=np.ptp(raw,1); peak=np.max(np.abs(raw),1); crest=peak/np.maximum(rms,1e-12); zcr=np.mean(np.diff(np.signbit(raw),axis=1)!=0,axis=1); return np.c_[rms,meanabs,std,p2p,crest,zcr]
 f=np.fft.rfftfreq(N_TIME,1/FS); x=np.maximum(raw,0); p=x*x; total=p.sum(1); centroid=(p@f)/np.maximum(total,1e-12); bw=np.sqrt(np.sum(p*(f-centroid[:,None])**2,1)/np.maximum(total,1e-12)); peakf=f[np.argmax(p,1)]; prob=p/np.maximum(total[:,None],1e-12); ent=-np.sum(prob*np.log(prob+1e-12),1)/np.log(p.shape[1]); c=np.cumsum(p,1); ridx=(c>=.85*total[:,None]).argmax(1); roll=f[ridx]; return np.c_[np.log1p(total),centroid,bw,peakf,ent,roll]

# simple baselines
for domain in ['time','frequency']:
 raw,y,ids,g=load(domain); H=features(raw,domain); rows=[]; preds=[]
 for r,s,f,tr,va,te in outer(y,g):
  maj=int(np.bincount(y[tr]).argmax()); pr=np.full(len(te),maj); p_arc=np.full(len(te),float(maj)); mm=mets(y[te],pr); rows.append(dict(domain=domain,model='Majority',repeat=r,outer_seed=s,fold=f,init_seed=-1,best_epoch=-1,train_n=len(tr),val_n=len(va),test_n=len(te),test_arc_n=int(y[te].sum()),params=0,train_seconds=0,confusion_matrix=json.dumps(confusion_matrix(y[te],pr,labels=[0,1]).tolist()),**mm));
  for j,i in enumerate(te): preds.append(dict(domain=domain,model='Majority',repeat=r,fold=f,record_id=ids[i],exact_group=g[i],label=int(y[i]),p_arc=float(p_arc[j]),prediction=int(pr[j])))
  sc=StandardScaler().fit(H[tr]); lr=LogisticRegression(class_weight='balanced',solver='liblinear',random_state=5000+r*100+f,max_iter=1000).fit(sc.transform(H[tr]),y[tr]); pp=lr.predict_proba(sc.transform(H[te])); pr=pp.argmax(1); mm=mets(y[te],pr); rows.append(dict(domain=domain,model='Signal-LogReg',repeat=r,outer_seed=s,fold=f,init_seed=5000+r*100+f,best_epoch=-1,train_n=len(tr),val_n=len(va),test_n=len(te),test_arc_n=int(y[te].sum()),params=len(lr.coef_.ravel())+1,train_seconds=0,confusion_matrix=json.dumps(confusion_matrix(y[te],pr,labels=[0,1]).tolist()),**mm))
  for j,i in enumerate(te): preds.append(dict(domain=domain,model='Signal-LogReg',repeat=r,fold=f,record_id=ids[i],exact_group=g[i],label=int(y[i]),p_arc=float(pp[j,1]),prediction=int(pr[j])))
 pd.DataFrame(rows).to_csv(ART/f'fold_metrics_{domain}_simple.csv',index=False); pd.DataFrame(preds).to_csv(ART/f'fold_predictions_{domain}_simple.csv',index=False)

# combine completed main comparator + simple files
metrics=[]; preds=[]
for d in ['time','frequency']:
 for tag in ['1D-CNN','CNN-LSTM','CNN-GRU','Light-DSCNN','simple']:
  mf=ART/f'fold_metrics_{d}_{tag}.csv'; pf=ART/f'fold_predictions_{d}_{tag}.csv'
  if mf.exists(): metrics.append(pd.read_csv(mf)); preds.append(pd.read_csv(pf))
fm=pd.concat(metrics,ignore_index=True); pr=pd.concat(preds,ignore_index=True)
fm.to_csv(ART/'fold_metrics_main_comparators.csv',index=False); pr.to_csv(ART/'fold_predictions_main_comparators.csv',index=False)

summary=[]
for (d,m),g in fm.groupby(['domain','model']):
 row={'domain':d,'model':m,'n_outer_folds':len(g),'mean_test_n':g.test_n.mean(),'mean_test_arc_n':g.test_arc_n.mean(),'params':g.params.median()}
 for met in ['accuracy','macro_f1','arc_recall']:
  row[met+'_mean']=g[met].mean(); row[met+'_sd']=g[met].std(ddof=1)
 summary.append(row)
pd.DataFrame(summary).to_csv(ART/'repeated_comparator_summary.csv',index=False)

# pooled OOF (average probabilities across two repeats per record)
pool=[]
for (d,m,rid),g in pr.groupby(['domain','model','record_id']):
 p=float(g.p_arc.mean()); y=int(g.label.iloc[0]); pool.append({'domain':d,'model':m,'record_id':rid,'exact_group':g.exact_group.iloc[0],'label':y,'p_arc_mean_over_repeats':p,'prediction':int(p>=.5)})
pool=pd.DataFrame(pool); pool.to_csv(ART/'pooled_oof_predictions.csv',index=False)
psum=[]
for (d,m),g in pool.groupby(['domain','model']):
 mm=mets(g.label,g.prediction); psum.append({'domain':d,'model':m,'n_records':len(g),'arc_n':int(g.label.sum()),**mm,'confusion_matrix':json.dumps(confusion_matrix(g.label,g.prediction,labels=[0,1]).tolist())})
pd.DataFrame(psum).to_csv(ART/'pooled_oof_summary.csv',index=False)

# paired differences against CNN-GRU
rows=[]
for d in ['time','frequency']:
 b=fm[(fm.domain==d)&(fm.model=='CNN-GRU')].set_index(['repeat','fold'])
 for m in ['CNN-LSTM','1D-CNN','Light-DSCNN','Signal-LogReg','Majority']:
  o=fm[(fm.domain==d)&(fm.model==m)].set_index(['repeat','fold']); ix=b.index.intersection(o.index)
  for met in ['accuracy','macro_f1','arc_recall']:
   dd=o.loc[ix,met]-b.loc[ix,met]; rows.append({'domain':d,'comparator':m,'metric':met,'mean_difference_comparator_minus_CNN_GRU':dd.mean(),'sd_difference':dd.std(ddof=1),'n_paired_folds':len(dd)})
pd.DataFrame(rows).to_csv(ART/'paired_outer_fold_differences.csv',index=False)

# exact identity-cluster bootstrap for CNN-GRU pooled predictions
rng=np.random.default_rng(BOOT_SEED); br=[]
for d in ['time','frequency']:
 g0=pool[(pool.domain==d)&(pool.model=='CNN-GRU')].copy(); groups=np.array(sorted(g0.exact_group.unique())); by={x:g0[g0.exact_group==x] for x in groups}; point=mets(g0.label,g0.prediction); vals={k:[] for k in ['accuracy','macro_f1','arc_recall']}
 for b in range(N_BOOT):
  samp=rng.choice(groups,size=len(groups),replace=True); bb=pd.concat([by[x] for x in samp],ignore_index=True)
  if bb.label.nunique()<2: continue
  mm=mets(bb.label,bb.prediction)
  for k in vals: vals[k].append(mm[k])
 for k,v in vals.items():
  lo,hi=np.percentile(v,[2.5,97.5]); br.append({'domain':d,'metric':k,'point':point[k],'ci95_low':lo,'ci95_high':hi,'bootstrap_replicates':len(v),'bootstrap_seed':BOOT_SEED})
pd.DataFrame(br).to_csv(ART/'identity_cluster_bootstrap_uncertainty.csv',index=False)

# conflict support
au=pd.read_csv(ART/'record_identity_audit.csv'); cs=[]
for d,g in au.groupby('domain'):
 for pol,keep in [('retain_conflicts_grouped',np.ones(len(g),bool)),('exclude_exact_cross_label_conflict_groups',~g.cross_label_conflict.astype(bool).values)]:
  q=g.iloc[np.where(keep)[0]]; cs.append({'domain':d,'policy':pol,'records':len(q),'arc_records':int(q.label_binary.sum()),'nonarc_records':int((q.label_binary==0).sum()),'exact_groups':q.exact_group.nunique(),'arc_groups':q[q.label_binary==1].exact_group.nunique()})
pd.DataFrame(cs).to_csv(ART/'conflict_policy_supports.csv',index=False)
print('postprocess done')
