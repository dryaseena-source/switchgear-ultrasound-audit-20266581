from pathlib import Path
import os, json, hashlib
import numpy as np
import pandas as pd
import scipy.io as sio
import librosa
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit, train_test_split
from sklearn.utils import resample

ART = Path(__file__).resolve().parent
DATA = Path(os.environ.get('ARC_AUDIT_DATA_DIR', ART))
PREP = ART / 'prepared'
PREP.mkdir(exist_ok=True)
FS=11025; N_TIME=20001; N_MELS=128
OUTER_SEEDS=[2026,2027]

REQ=['ArchingNorm.mat','MechanicalNorm2_20k.mat','CoronaNorm2_20k.mat','TrackingNorm2_20k.mat','NormalNorm2_20k.mat','FrequncyDomain.mat']
missing=[x for x in REQ if not (DATA/x).exists()]
if missing:
    raise FileNotFoundError('Place source MAT files in ARC_AUDIT_DATA_DIR (or artifact folder). Missing: '+', '.join(missing))

def sha_bytes(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def sha_record(v):
    return hashlib.sha256(np.ascontiguousarray(v,dtype='<f8').tobytes()).hexdigest()

def load_time():
    arc=sio.loadmat(DATA/'ArchingNorm.mat')['we'].astype(np.float64)
    parts=[]
    for src,fn in [('Mechanical','MechanicalNorm2_20k.mat'),('Corona','CoronaNorm2_20k.mat'),('Tracking','TrackingNorm2_20k.mat'),('Normal','NormalNorm2_20k.mat')]:
        parts.append((src,sio.loadmat(DATA/fn)['we'].astype(np.float64)))
    X=np.concatenate([arc]+[x for _,x in parts],axis=1).T
    y=np.r_[np.ones(arc.shape[1],int),np.zeros(sum(x.shape[1] for _,x in parts),int)]
    src=['Arching']*arc.shape[1]; srcidx=list(range(arc.shape[1]))
    for s,x in parts: src += [s]*x.shape[1]; srcidx += list(range(x.shape[1]))
    ids=np.array([f'T{i:03d}' for i in range(len(y))])
    return X,y,np.array(src),np.array(srcidx),ids

def load_freq():
    m=sio.loadmat(DATA/'FrequncyDomain.mat')
    mapping=[('Arching','x1',1),('Corona','x2',0),('Mechanical','x3',0),('Tracking','x4',0),('Normal','x5',0)]
    xs=[]; y=[]; src=[]; idx=[]
    for s,k,lab in mapping:
        a=m[k].astype(np.float64).T; xs.append(a); y += [lab]*len(a); src += [s]*len(a); idx += list(range(len(a)))
    X=np.concatenate(xs); y=np.array(y,int); ids=np.array([f'F{i:03d}' for i in range(len(y))])
    return X,y,np.array(src),np.array(idx),ids

def groups_for(X,domain):
    hs=np.array([sha_record(v) for v in X]); un={h:i for i,h in enumerate(sorted(set(hs)))}
    g=np.array([f'{domain[0].upper()}G{un[h]:03d}' for h in hs]); return hs,g

def time_rep(X):
    out=[]
    for v in X:
        m=librosa.feature.melspectrogram(y=v.astype(np.float32),sr=FS,n_fft=2048,hop_length=512,n_mels=N_MELS,power=2.0,center=True,pad_mode='constant')
        out.append(np.log1p(m).T.astype(np.float32))
    return np.stack(out)

def mel_filter():
    freqs=np.fft.rfftfreq(N_TIME,d=1/FS)
    edges=librosa.mel_frequencies(n_mels=N_MELS+2,fmin=0.0,fmax=FS/2,htk=False)
    fb=np.zeros((N_MELS,len(freqs)),np.float32)
    for m in range(N_MELS):
        l,c,r=edges[m:m+3]; left=(freqs-l)/max(c-l,1e-12); right=(r-freqs)/max(r-c,1e-12)
        fb[m]=np.maximum(0,np.minimum(left,right))
    fb *= (2.0/np.maximum(edges[2:N_MELS+2]-edges[:N_MELS],1e-12))[:,None]
    return fb

def freq_rep(X):
    fb=mel_filter(); p=np.maximum(X,0).astype(np.float64)**2; mel=p@fb.T
    return np.log1p(mel).astype(np.float32)[:,:,None]

def fixed_group_split(X,y,g,seed0=2028):
    best=None; target=np.array([.70,.15,.15]); overall=y.mean()
    for s in range(seed0,seed0+500):
        tr,rem=next(GroupShuffleSplit(1,test_size=.30,random_state=s).split(X,y,g))
        va_rel,te_rel=next(GroupShuffleSplit(1,test_size=.50,random_state=s+10000).split(X[rem],y[rem],g[rem]))
        va,te=rem[va_rel],rem[te_rel]; sets=[tr,va,te]
        props=np.array([len(z)/len(y) for z in sets])
        classdiff=sum(abs(y[z].mean()-overall) if len(np.unique(y[z]))>1 else 1.0 for z in sets)
        obj=float(np.abs(props-target).sum()+.35*classdiff)
        if best is None or obj<best[0]: best=(obj,s,sets)
    return best[1],best[2]

def near_components(X,y,domain,thresholds=(.999,.9999,.99999)):
    Z=X.astype(np.float32) if domain=='time' else np.log1p(np.maximum(X,0)).astype(np.float32)
    Z-=Z.mean(1,keepdims=True); n=np.linalg.norm(Z,axis=1,keepdims=True); n[n==0]=1; Z/=n
    S=np.abs(Z@Z.T); np.fill_diagonal(S,0)
    rows=[]; primary=None
    for th in thresholds:
        parent=list(range(len(X)))
        def find(a):
            while parent[a]!=a: parent[a]=parent[parent[a]]; a=parent[a]
            return a
        def union(a,b):
            a,b=find(a),find(b)
            if a!=b: parent[b]=a
        ii,jj=np.where(np.triu(S>=th,1))
        for a,b in zip(ii,jj): union(int(a),int(b))
        roots=[find(i) for i in range(len(X))]; rem={r:j for j,r in enumerate(sorted(set(roots)))}; g=np.array([f'{domain[0].upper()}N{rem[r]:03d}' for r in roots])
        cross=sum(len(set(y[g==gg]))>1 for gg in np.unique(g)); largest=max(np.bincount(pd.factorize(g)[0]))
        rows.append(dict(domain=domain,threshold_abs_corr=th,candidate_pairs=len(ii),groups=len(np.unique(g)),cross_label_components=cross,largest_group=int(largest)))
        if abs(th-.9999)<1e-12: primary=g
    return rows,primary

TD=load_time(); FD=load_freq(); datasets={'time':TD,'frequency':FD}
reps={'time':time_rep(TD[0]),'frequency':freq_rep(FD[0])}
np.save(PREP/'X_time_logmel.npy',reps['time']); np.save(PREP/'X_frequency_logmel.npy',reps['frequency'])

audit=[]; groupmap={}; nearrows=[]; nearmap={}
for domain,(X,y,src,sidx,ids) in datasets.items():
    hs,g=groups_for(X,domain); groupmap[domain]=g
    conflicts={gg:(len(set(y[g==gg]))>1) for gg in np.unique(g)}
    nr,ng=near_components(X,y,domain); nearrows += nr; nearmap[domain]=ng
    for i in range(len(y)):
        audit.append(dict(domain=domain,record_id=ids[i],source_class=src[i],source_index=int(sidx[i]),label='Arcing' if y[i] else 'Non-arcing',label_binary=int(y[i]),sha256=hs[i],exact_group=g[i],cross_label_conflict=conflicts[g[i]],near_group_0_9999=ng[i]))
pd.DataFrame(audit).to_csv(ART/'record_identity_audit.csv',index=False)
pd.DataFrame(nearrows).to_csv(ART/'near_duplicate_threshold_sensitivity.csv',index=False)

# deterministic fixed descriptive splits
rows=[]
for domain,(X,y,src,sidx,ids) in datasets.items():
    s,sets=fixed_group_split(reps[domain],y,groupmap[domain],2028)
    for part,ix in zip(['train','validation','test'],sets):
        for i in ix: rows.append(dict(domain=domain,split_seed=s,partition=part,record_id=ids[i],label=int(y[i]),exact_group=groupmap[domain][i]))
pd.DataFrame(rows).to_csv(ART/'corrected_fixed_split_indices.csv',index=False)

# original split membership after pre-split arcing bootstrap
rows=[]
for domain,(X,y,src,sidx,ids),s1,s2 in [('time',TD,21,20),('frequency',FD,0,0)]:
    arc=np.where(y==1)[0]; non=np.where(y==0)[0]; sampled=resample(arc,replace=True,n_samples=len(arc),random_state=42)
    combined=np.r_[sampled,non]; pos=np.arange(len(combined)); tr,rem=train_test_split(pos,test_size=.30,random_state=s1); va,te=train_test_split(rem,test_size=.50,random_state=s2)
    for part,pp in [('train',tr),('validation',va),('test',te)]:
        for j in pp:
            i=combined[j]; rows.append(dict(domain=domain,partition=part,pipeline_sample_position=int(j),record_id=ids[i],source_class=src[i],source_index=int(sidx[i]),label=int(y[i]),pre_split_arc_bootstrap=bool(j<len(sampled))))
pd.DataFrame(rows).to_csv(ART/'initial_pipeline_split_indices.csv',index=False)

# all outer/inner partitions
rows=[]
for domain,(X,y,src,sidx,ids) in datasets.items():
    g=groupmap[domain]
    for rep,seed in enumerate(OUTER_SEEDS):
        outer=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
        for fold,(trv,te) in enumerate(outer.split(np.zeros(len(y)),y,g)):
            inner_seed=8000+100*rep+fold; inner=StratifiedGroupKFold(5,shuffle=True,random_state=inner_seed)
            trrel,varel=next(inner.split(np.zeros(len(trv)),y[trv],g[trv])); tr,va=trv[trrel],trv[varel]
            for part,ix in [('train',tr),('validation',va),('test',te)]:
                for i in ix: rows.append(dict(domain=domain,repeat=rep,outer_seed=seed,fold=fold,inner_seed=inner_seed,partition=part,record_id=ids[i],label=int(y[i]),exact_group=g[i]))
pd.DataFrame(rows).to_csv(ART/'grouped_cv_partitions.csv',index=False)

# source hashes
rows=[]
for fn in REQ + ['Time_Domain_1D_CNN_GRU_Colab.ipynb','Frequency_Domain_1D_CNN_GRU.ipynb']:
    p=DATA/fn
    if p.exists(): rows.append(dict(file=fn,sha256=sha_bytes(p),bytes=p.stat().st_size))
pd.DataFrame(rows).to_csv(ART/'source_file_hashes.csv',index=False)
print('Prepared representations and audit metadata in',ART)
