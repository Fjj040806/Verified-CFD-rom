"""Recompute the core ROM result from stored, settled CFD snapshots."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy.sparse import load_npz
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from flowrom.rom import WeightedPOD,ReducedDMD
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=root/'demo-output');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
f=root/'data/demo/re100-settled.npz'
if not f.exists():raise FileNotFoundError('Missing bundled demonstration data: '+str(f))
d=np.load(f);t=d['t'];U=d['u'].T;M=load_npz(root/'data/demo/velocity-mass.npz');n=len(t);nt=int(.6*n);nv=int(.2*n)
training=U[:,:nt];test=U[:,nt+nv:];pod=WeightedPOD(training,M);model=ReducedDMD(pod,training,20,float(np.median(np.diff(t))));forecast=model.forecast(training[:,-1],n-nt)
summary=dict(training=[float(t[0]),float(t[nt-1])],test=[float(t[nt+nv]),float(t[-1])],pod_rank=10,dmd_rank=20,pod_test=pod.errors(pod.reconstruct(test,10),test),dmd_test=pod.errors(forecast[:,nv:],test),dmd_frequency=model.dominant_frequency,note='Ranks frozen from the earlier validation study. Same-Re short-horizon demonstration, not a fresh independent test.')
(a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
c=pod.coefficients(U,1)[0];pred=pod.coefficients(forecast,1)[0]
fig,ax=plt.subplots(figsize=(9,3.5),constrained_layout=True);fig.set_facecolor('#FBFCFB');ax.plot(t,c,color='#318B88',label='Computed CFD');ax.plot(t[nt:],pred,'--',color='#CC8868',label='Affine DMD forecast');ax.axvline(t[nt-1],color='#788393',ls=':',label='Last training frame');ax.axvspan(t[nt+nv],t[-1],color='#318B88',alpha=.08,label='Test interval');ax.set(xlabel='Nondimensional time',ylabel='First mass-weighted POD coefficient',title='Re100 cylinder wake: forecasting from saved CFD data');ax.legend(fontsize=8);ax.grid(alpha=.15);fig.savefig(a.output/'forecast.png',dpi=180)
