"""Predeclared rank/cadence/sample-count/phase controls on saved Re100 data.
All models use training data only. No selection from these held-out diagnostics.
"""
from pathlib import Path
import json,time,platform
import scipy
import numpy as np
from scipy.sparse import load_npz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flowrom.rom import WeightedPOD,ReducedDMD
root=Path(__file__).resolve().parents[1];out=root/'results/phase6';out.mkdir(exist_ok=True)
start=time.perf_counter();data=np.load(root/'results/phase4b/coarse-half-snapshots.npz' if (root/'results/phase4b/coarse-half-snapshots.npz').exists() else root/'data/demo/re100-settled.npz');mask=data['t']>=27-1e-9;t=data['t'][mask];U=data['u'][mask].T
M=load_npz(root/'results/phase4b/coarse-half-velocity-mass.npz');dt=float(np.median(np.diff(t)))
ref=json.loads((root/'results/phase5/summary.json').read_text());freq=ref['force_frequency'];period=1/freq

def evaluate(indices,rank,horizon,common_offsets=None):
 train=U[:,indices];pod=WeightedPOD(train,M)
 if len(pod.values)<rank:raise ValueError('Insufficient resolved POD rank')
 stride=int(indices[1]-indices[0]);steps=horizon//stride;future_indices=indices[-1]+stride*np.arange(1,steps+1)
 truth=U[:,future_indices];rows={}
 for affine,name in [(False,'linear'),(True,'affine')]:
  model=ReducedDMD(pod,train,rank,dt*stride,affine=affine);pred=model.forecast(train[:,-1],steps)
  c=pod.coefficients(train,rank);design=np.vstack([c[:,:-1],np.ones(c.shape[1]-1)]) if affine else c[:,:-1]
  sv=np.linalg.svd(design,compute_uv=False);threshold=sv[0]*1e-10
  row=dict(native_error=pod.errors(pred,truth),frequency=model.dominant_frequency,frequency_relative_error=None if model.dominant_frequency is None else abs(model.dominant_frequency/freq-1),spectral_radius=float(max(abs(model.eigenvalues))),design_condition=float(sv[0]/sv[-1]),effective_design_rank=int(np.sum(sv>threshold)),design_rows=int(design.shape[0]),design_columns=int(design.shape[1]),bias_norm=float(np.linalg.norm(model.bias)))
  if common_offsets is not None:
   common_columns=np.asarray(common_offsets)//stride-1
   row['common_error']=pod.errors(pred[:,common_columns],U[:,indices[-1]+np.asarray(common_offsets)])
  rows[name]=row
 return dict(rank=rank,snapshot_interval=dt*stride,training_samples=len(indices),training_start=float(t[indices[0]]),training_end=float(t[indices[-1]]),training_duration=float(t[indices[-1]]-t[indices[0]]),observed_cycles=float((t[indices[-1]]-t[indices[0]])*freq),forecast_end=float(t[future_indices[-1]]),forecast_points=len(future_indices),pod_native_error=pod.errors(pod.reconstruct(truth,rank),truth),models=rows)

# Fixed rank and training endpoints: fewer snapshots at coarser cadence.
# Evaluate all on identical 4 future times (0.16,0.32,0.48,0.64 after seed).
common=[16,32,48,64];cadence=[];sample_count=[]
for stride in [1,2,4,8,16]:
 cadence.append(evaluate(np.arange(20,181,stride),6,64,common))
 sample_count.append(evaluate(np.arange(180-10*stride,181,stride),6,64,common))
# Fixed 121-snapshot training windows, rank and 1.2 time-unit horizon.
# Quarter/eighth-cycle shifts rounded to the available 0.01 snapshot grid.
phase=[]
for fraction in np.arange(8)/8:
 shift=int(round(fraction*period/dt))
 for rank in [10,20]:
  r=evaluate(np.arange(shift,shift+121),rank,120)
  r.update(requested_phase_fraction=float(fraction),actual_phase_fraction=float(shift*dt/period));phase.append(r)
# Fixed seed, cadence and rank; window-length diagnostics, not model selection.
window=[]
for count in [34,51,68,84,101,118,135,151,168,180]:
 window.append(evaluate(np.arange(181-count,181),10,64,common))
summary=dict(protocol=dict(rank_cadence=6,rank_phase=[10,20],rank_window=10,common_future_offsets=[.16,.32,.48,.64],fixed_duration=1.6,fixed_samples=11,phase_training_duration=1.2,phase_prediction_duration=1.2,pseudoinverse_rcond=1e-10,test_role='Controlled diagnostics; no model/hyperparameter selection',limitations=['Cadence, duration and sample count obey N-1=T/dt; they cannot all be held fixed while dt varies.','Four common evaluation times do not resolve all within-cycle forecast errors.','Phase windows overlap on one deterministic trajectory; no independent replicates or confidence intervals.','Entire study concerns the same Re100 mesh and numerical solver.']),force_frequency=freq,cadence_fixed_duration=cadence,cadence_fixed_samples=sample_count,phase_study=phase,window_study=window,runtime_seconds=time.perf_counter()-start)
summary['software']=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__)
all_rows=cadence+sample_count+phase+window
assert len(all_rows)==36
for r in all_rows:
 assert r['forecast_end']>r['training_end']
 for m in r['models'].values():
  assert np.isfinite(m['native_error']['fluctuation_relative'])
  assert m['effective_design_rank']==m['design_rows']
assert len({r['rank'] for r in cadence})==1
assert len({r['training_samples'] for r in sample_count})==1
assert max(abs(r['training_duration']-1.6) for r in cadence)<1e-10
(out/'audit.json').write_text(json.dumps(dict(data_configurations=36,model_fits=72,all_finite_errors=True,all_designs_full_row_rank_at_rcond=True,fixed_rank_cadence_verified=True,fixed_count_verified=True,fixed_training_endpoints_verified=True,notes='Protocol and algebra checks; not independent experimental replication.'),indent=2))
(out/'summary.json').write_text(json.dumps(summary,indent=2))
fig,axs=plt.subplots(2,2,figsize=(11,8),constrained_layout=True);fig.set_facecolor('#FBFCFB');teal='#318B88';clay='#CC8868';gray='#788393'
for ax,rows,title in [(axs[0,0],cadence,'A · Fixed duration 1.6, fixed rank 6'),(axs[0,1],sample_count,'B · Fixed sample count 11, fixed rank 6')]:
 for name,label,color in [('linear','Without constant term',gray),('affine','With constant term',teal)]:
  ax.semilogy([r['snapshot_interval'] for r in rows],[r['models'][name]['common_error']['fluctuation_relative']*100 for r in rows],'o-',color=color,label=label)
 ax.set(xlabel='Snapshot interval',ylabel='Common-time fluctuation error (%)',title=title);ax.legend(fontsize=8)
for rank,color in [(10,teal),(20,clay)]:
 rows=[r for r in phase if r['rank']==rank]
 for name,style in [('linear',':'),('affine','-')]:
  axs[1,0].semilogy([r['actual_phase_fraction'] for r in rows],[r['models'][name]['native_error']['fluctuation_relative']*100 for r in rows],style+'o',color=color,label=f'Rank {rank}, {name}')
axs[1,0].set(xlabel='Training-window shift / shedding period',ylabel='Future fluctuation error (%)',title='C · Same window length and forecast horizon');axs[1,0].legend(fontsize=8,ncol=2)
for name,label,color in [('linear','Without constant term',gray),('affine','With constant term',teal)]:
 axs[1,1].semilogy([r['observed_cycles'] for r in window],[r['models'][name]['common_error']['fluctuation_relative']*100 for r in window],'o-',color=color,label=label)
axs[1,1].set(xlabel='Observed shedding cycles',ylabel='Common-time fluctuation error (%)',title='D · Fixed endpoint, cadence 0.01, rank 10');axs[1,1].legend(fontsize=8)
for ax in axs.flat:ax.grid(alpha=.15)
for ext in ('png','svg','pdf'):fig.savefig(root/f'figures/phase6-controls.{ext}',dpi=180)
print(json.dumps(dict(cadence=[(r['snapshot_interval'],r['models']['affine']['common_error']['fluctuation_relative']) for r in cadence],samples=[(r['snapshot_interval'],r['models']['affine']['common_error']['fluctuation_relative'],r['models']['affine']['design_condition']) for r in sample_count],phase=[(r['rank'],r['actual_phase_fraction'],r['models']['linear']['native_error']['fluctuation_relative'],r['models']['affine']['native_error']['fluctuation_relative']) for r in phase],window=[(r['observed_cycles'],r['models']['linear']['common_error']['fluctuation_relative'],r['models']['affine']['common_error']['fluctuation_relative']) for r in window]),indent=2))
