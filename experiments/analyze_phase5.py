from pathlib import Path
import json,numpy as np
from scipy.sparse import load_npz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flowrom.rom import WeightedPOD,ReducedDMD
root=Path(__file__).resolve().parents[1];out=root/'results/phase5';out.mkdir(exist_ok=True)
d=np.load(root/'results/phase4b/coarse-half-snapshots.npz' if (root/'results/phase4b/coarse-half-snapshots.npz').exists() else root/'data/demo/re100-settled.npz');sel=d['t']>=27-1e-9;t=d['t'][sel];U=d['u'][sel].T
M=load_npz(root/'results/phase4b/coarse-half-velocity-mass.npz');B=load_npz(root/'results/phase4b/coarse-half-divergence.npz');n=len(t);ntr=int(.6*n);nva=int(.2*n)
train=U[:,:ntr];validation=U[:,ntr:ntr+nva];test=U[:,ntr+nva:];pod=WeightedPOD(train,M)
reference=json.loads((root/'results/phase4b/analysis.json').read_text());frequency=next(r['frequency'] for r in reference['runs'] if r['name']=='coarse-half');dt=float(np.median(np.diff(t)));ranks=[r for r in [1,2,3,4,6,8,10,12,16,20] if r<=len(pod.values)]
rows=[];energy=np.cumsum(pod.values)/pod.values.sum()
for r in ranks:
 rec=pod.reconstruct(validation,r);model=ReducedDMD(pod,train,r,dt);forecast=model.forecast(train[:,-1],n-ntr)
 linear=ReducedDMD(pod,train,r,dt,affine=False).forecast(train[:,-1],n-ntr)
 rows.append(dict(rank=r,training_energy_fraction=float(energy[r-1]),pod_validation=pod.errors(rec,validation),pod_test=pod.errors(pod.reconstruct(test,r),test),dmd_validation=pod.errors(forecast[:,:nva],validation),dmd_test=pod.errors(forecast[:,nva:],test),dmd_frequency=model.dominant_frequency,linear_dmd_validation=pod.errors(linear[:,:nva],validation),linear_dmd_test=pod.errors(linear[:,nva:],test)))
# Select using validation only; reveal final test metrics after selection.
pod_candidates=[x for x in rows if x['training_energy_fraction']>=.99 and x['pod_validation']['fluctuation_relative']<=.01]
pod_rank=pod_candidates[0]['rank'] if pod_candidates else min(rows,key=lambda x:x['pod_validation']['fluctuation_relative'])['rank']
dmd_rank=min(rows,key=lambda x:x['dmd_validation']['fluctuation_relative'])['rank'];model=ReducedDMD(pod,train,dmd_rank,dt);forecast=model.forecast(train[:,-1],n-ntr)
mean=np.repeat(pod.mean[:,None],test.shape[1],axis=1);persistence=np.repeat(train[:,-1,None],test.shape[1],axis=1)
reconstruction=pod.reconstruct(U,pod_rank);sensitivity=[]
for stride in [1,2,4,8,16]:
 indices=np.arange(0,n,stride);data=U[:,indices];times=t[indices];a=np.count_nonzero(indices<ntr);b=np.count_nonzero(indices<ntr+nva)-a
 p=WeightedPOD(data[:,:a],M);rank=min(dmd_rank,len(p.values),a-2);dm=ReducedDMD(p,data[:,:a],rank,dt*stride);pred=dm.forecast(data[:,a-1],len(indices)-a)
 sensitivity.append(dict(stride=stride,dt=dt*stride,nyquist=1/(2*dt*stride),rank=rank,training_start=float(times[0]),training_end=float(times[a-1]),test_points=len(indices)-a-b,dmd_frequency=dm.dominant_frequency,test= p.errors(pred[:,b:],data[:,a+b:])))
# Hold cadence fixed; vary training window ending at the same training boundary.
windows=[]
for duration in [.5,1.,1.5]:
 mask=(t[:ntr]>=t[ntr-1]-duration);tr=train[:,mask];p=WeightedPOD(tr,M);rank=min(dmd_rank,len(p.values),tr.shape[1]-2);dm=ReducedDMD(p,tr,rank,dt);pred=dm.forecast(tr[:,-1],n-ntr)
 windows.append(dict(requested_duration=duration,actual_duration=float(t[ntr-1]-t[:ntr][mask][0]),rank=rank,frequency=dm.dominant_frequency,test=p.errors(pred[:,nva:],test)))
selected_pod=next(r for r in rows if r['rank']==pod_rank);selected_dmd=next(r for r in rows if r['rank']==dmd_rank)
summary=dict(samples=n,dt=dt,training=[float(t[0]),float(t[ntr-1])],validation=[float(t[ntr]),float(t[ntr+nva-1])],test=[float(t[ntr+nva]),float(t[-1])],pod_rank=pod_rank,dmd_rank=dmd_rank,selected_pod=selected_pod,selected_dmd=selected_dmd,force_frequency=frequency,dmd_frequency_relative_error=abs(model.dominant_frequency/frequency-1) if model.dominant_frequency else None,mean_baseline=pod.errors(mean,test),persistence_baseline=pod.errors(persistence,test),mass_orthogonality_max_error=float(np.max(np.abs(pod.modes.T@(M@pod.modes)-np.eye(len(pod.values))))),reconstructed_weak_divergence_max=float(np.max(np.linalg.norm(B@reconstruction,axis=0))),rank_study=rows,cadence_study=sensitivity,window_study=windows)
(out/'summary.json').write_text(json.dumps(summary,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)));np.savez_compressed(out/'models-and-reconstructions.npz',t=t,mean=pod.mean,modes=pod.modes[:,:20],eigenvalues=pod.values,reconstruction=reconstruction,forecast=forecast,forecast_times=t[ntr:],pod_rank=pod_rank,dmd_rank=dmd_rank,dmd_eigenvalues=model.eigenvalues,dmd_operator=model.A,dmd_bias=model.bias)
fig,axs=plt.subplots(2,2,figsize=(11,8),constrained_layout=True);fig.set_facecolor('#FBFCFB');teal='#318B88';clay='#CC8868'
axs[0,0].semilogy(np.arange(1,len(pod.values)+1),pod.values/pod.values.sum(),'o-',color=teal);axs[0,0].set(xlabel='POD mode',ylabel='Training fluctuation energy fraction',title='Mass-weighted spectrum',xlim=(0,22))
for key,label,color in [('pod_test','POD reconstruction',teal),('dmd_test','DMD forecast',clay)]:axs[0,1].semilogy(ranks,[r[key]['fluctuation_relative'] for r in rows],'o-',label=label,color=color)
axs[0,1].set(xlabel='Rank',ylabel='Held-out fluctuation error',title='Reconstruction vs genuine future forecast');axs[0,1].legend()
axs[0,1].semilogy(ranks,[r['linear_dmd_test']['fluctuation_relative'] for r in rows],':',color='#788393',label='DMD without bias');axs[0,1].legend()
c=pod.coefficients(U,2);fc=pod.coefficients(forecast,2)
axs[1,0].plot(t,c[0],color=teal,label='Computed flow');axs[1,0].plot(t[ntr:],fc[0],'--',color=clay,label='DMD forecast');axs[1,0].axvline(t[ntr],color='#243746',ls=':',label='Training end');axs[1,0].set(xlabel='Time',ylabel='First POD coefficient',title='Forecast starts at the training boundary');axs[1,0].legend(fontsize=8)
axs[1,1].plot([s['dt'] for s in sensitivity],[s['dmd_frequency'] for s in sensitivity],'o-',color=teal);axs[1,1].axhline(frequency,ls='--',color=clay,label='Lift frequency');axs[1,1].set(xlabel='Snapshot interval',ylabel='DMD frequency',title='Cadence sensitivity · near Nyquist at 0.16');axs[1,1].legend()
axs[1,1].ticklabel_format(axis='y',useOffset=False)
err_ax=axs[1,1].twinx();err_ax.semilogy([s['dt'] for s in sensitivity],[s['test']['fluctuation_relative'] for s in sensitivity],'s:',color='#788393');err_ax.set_ylabel('Forecast fluctuation error',color='#788393')
for ax in axs.flat:ax.grid(alpha=.15)
for ext in ('png','svg','pdf'):fig.savefig(root/f'figures/phase5-rom.{ext}',dpi=180)
print(json.dumps({k:v for k,v in summary.items() if k not in ['rank_study']},indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
