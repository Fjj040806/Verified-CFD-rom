from pathlib import Path
import json
import numpy as np
from scipy.signal import find_peaks
from scipy.sparse import load_npz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];out=root/'results/phase4b'
reference=json.loads((out/'official-reference.json').read_text());rows=[]

def cycle_metrics(a,start):
 a=a[a['t']>=start];dt=np.median(np.diff(a['t']));peaks,_=find_peaks(-a['cl'],prominence=.5,distance=round(.2/dt))
 refined=[]
 for i in peaks:
  y=a['cl'];den=y[i-1]-2*y[i]+y[i+1];offset=.5*(y[i-1]-y[i+1])/den
  refined.append(float(a['t'][i]+offset*dt))
 periods=np.diff(refined)
 if len(periods)<5:raise RuntimeError('Fewer than five settled cycles')
 cycle_extrema=[]
 for i,j in zip(peaks,peaks[1:]):
  segment=a[i:j+1]
  def extremum(key,minimum):
   values=a[key];index=i+int(np.argmin(segment[key]) if minimum else np.argmax(segment[key]))
   if index<=0 or index>=len(values)-1:return float(values[index])
   den=values[index-1]-2*values[index]+values[index+1]
   return float(values[index]-(values[index+1]-values[index-1])**2/(8*den)) if abs(den)>1e-14 else float(values[index])
  cycle_extrema.append([extremum('cd',True),extremum('cd',False),extremum('cl',True),extremum('cl',False)])
 extrema=np.array(cycle_extrema)
 drift=float(np.max(np.abs(extrema[-3:].mean(axis=0)-extrema[:3].mean(axis=0)))/np.ptp(a['cl']))
 period=float(periods.mean());s=dict(start=float(a['t'][0]),end=float(a['t'][-1]),cycles=len(periods),period=period,period_std=float(periods.std()),frequency=1/period,st=.1/period,cycle_extrema_drift_relative_to_lift_range=drift,cd_min=float(a['cd'].min()),cd_max=float(a['cd'].max()),cl_min=float(a['cl'].min()),cl_max=float(a['cl'].max()),cd_time_mean=float(a['cd'].mean()),dp_min=float(a['dp'].min()),dp_max=float(a['dp'].max()),max_nonlinear_residual=float(a['nonlinear_residual'].max()),max_weak_divergence=float(a['weak_divergence'].max()))
 # Official 'mean' is midpoint of extrema, not arithmetic time mean.
 s['cd_extrema_midpoint']=(s['cd_min']+s['cd_max'])/2;s['cl_extrema_midpoint']=(s['cl_min']+s['cl_max'])/2
 s['st_relative_error']=abs(s['st']/reference['st']-1)
 s['late_cycles_settled']=bool(drift<.001 and periods.std()/period<.001)
 return s,a
for name in ['coarse','refined','coarse-half']:
 meta=out/f'{name}-metadata.json'
 if not meta.exists():continue
 metadata=json.loads(meta.read_text())
 # Exclude short pilots or incomplete metadata; only explicit names count.
 a=np.genfromtxt(out/f'{name}-history.csv',delimiter=',',names=True)
 start=metadata['end']-5 if name in ['coarse','refined'] else metadata['end']-3
 metrics,a=cycle_metrics(a,start);metrics.update(name=name,**metadata);metrics['analysis_start']=start;metrics['analysis_end']=metadata['end']
 rows.append(metrics)
 baseline='coarse-half' if (out/'coarse-half-snapshots.npz').exists() else 'coarse'
 if name==baseline:
  snap=np.load(out/f'{name}-snapshots.npz');U=snap['u'].astype(float);M=load_npz(out/f'{name}-velocity-mass.npz');B=load_npz(out/f'{name}-divergence.npz')
  metrics['snapshot_count']=len(U);metrics['snapshot_dt']=float(np.median(np.diff(snap['t'])));metrics['snapshot_dtype']=str(snap['u'].dtype)
  metrics['max_saved_weak_divergence']=float(np.max(np.linalg.norm(B@U.T,axis=0)))
  mass_energy=.5*np.sum(U.T*(M@U.T),axis=0)
  np.savez_compressed(out/'rom-preparation.npz',t=snap['t'],mass_energy=mass_energy,baseline=name)
(out/'analysis.json').write_text(json.dumps(dict(reference=reference,runs=rows,baseline=baseline),indent=2))
fig,axs=plt.subplots(2,2,figsize=(11,8),constrained_layout=True);fig.set_facecolor('#FBFCFB')
for metrics,color in zip(rows,['#243746','#318B88','#CC8868','#8B789D']):
 name=metrics['name'];a=np.genfromtxt(out/f'{name}-history.csv',delimiter=',',names=True);late=a[a['t']>=metrics['analysis_start']]
 axs[0,0].plot(a['t'],a['cl'],lw=.65,color=color,label=name)
 # Align each cycle on a lift minimum, do not compare raw phase at equal times.
 minima,_=find_peaks(-late['cl'],prominence=.5);i,j=minima[-2:]
 cycle=late[i:j+1];phase=(cycle['t']-cycle['t'][0])/(cycle['t'][-1]-cycle['t'][0])
 axs[0,1].plot(phase,cycle['cl'],color=color,label=name);axs[1,0].plot(phase,cycle['cd'],color=color,label=name)
 x=late['cl']-late['cl'].mean();freq=np.fft.rfftfreq(len(x),np.median(np.diff(late['t'])));spectrum=np.abs(np.fft.rfft(x*np.hanning(len(x))))**2
 axs[1,1].plot(freq,spectrum/spectrum.max(),color=color,label=name)
axs[0,0].set(xlabel='Time',ylabel='Lift coefficient',title='Startup and long-time periodic lift')
axs[0,1].set(xlabel='Cycle phase',ylabel='Lift coefficient',title='Last cycle · aligned at lift minimum')
axs[1,0].set(xlabel='Cycle phase',ylabel='Drag coefficient',title='Last cycle · aligned at lift minimum')
axs[1,1].axvline(1/reference['period'],color='#71808A',ls='--',label='updated official reference');axs[1,1].set(xlabel='Frequency',ylabel='Normalized windowed power',title='Lift spectrum · finite-window resolution',xlim=(0,8))
for ax in axs.flat:ax.legend(fontsize=8);ax.grid(alpha=.15)
for ext in ('png','svg','pdf'):fig.savefig(root/f'figures/phase4b-diagnostics.{ext}',dpi=180)
print(json.dumps(dict(runs=rows,baseline=baseline),indent=2))
