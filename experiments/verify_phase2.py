from pathlib import Path
import csv,json,time,platform,sys
import numpy as np
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from flowrom import PeriodicMAC
from flowrom.navier_stokes import PeriodicFlow,exact_taylor_green,centered_fields
from flowrom.export import write_vti,write_pvd

root=Path(__file__).resolve().parents[1];out=root/'results/phase2';out.mkdir(exist_ok=True)
start=time.perf_counter();nu=.1

def run(n,dt,T,background=(0.,0.),snapshots=False):
    g=PeriodicMAC(n,n);s=PeriodicFlow(g,nu)
    steps=round(T/dt);dt=T/steps
    u,v,p=exact_taylor_green(g,0,nu,background)
    history=[];frames=[];maxdiv=0.;maxcfl=0.
    if snapshots:frames.append((0.,u.copy(),v.copy(),p.copy()))
    for k in range(steps):
        u,v,p,d=s.step(u,v,p,dt)
        maxdiv=max(maxdiv,d['divergence_rms']);maxcfl=max(maxcfl,d['cfl'])
        if snapshots and (k+1)%max(1,steps//20)==0:
            t=(k+1)*dt
            history.append((t,d['kinetic_energy'],.25*np.exp(-4*nu*t)))
            frames.append((t,u.copy(),v.copy(),p.copy()))
    ue,ve,pe=exact_taylor_green(g,T,nu,background)
    error=np.sqrt(.5*np.mean((u-ue)**2+(v-ve)**2))
    perr=np.sqrt(np.mean((p-pe)**2))
    rate=8*np.sin(g.hx/2)**2/g.hx**2
    # Only stationary TG has this simple exact semidiscrete velocity trajectory.
    u0,v0,_=exact_taylor_green(g,0,nu)
    semi=np.sqrt(.5*np.mean((u-u0*np.exp(-nu*rate*T))**2+(v-v0*np.exp(-nu*rate*T))**2)) if background==(0.,0.) else None
    return dict(n=n,dt=dt,T=T,velocity_rms=float(error),pressure_rms=float(perr),semidiscrete_velocity_rms=None if semi is None else float(semi),max_divergence=maxdiv,max_cfl=maxcfl),g,history,frames

spatial=[]
for n in (16,32,64,128):
    # dt proportional to h² keeps first-order temporal contamination small.
    dt=.0005*(16/n)**2
    r,*_=run(n,dt,.02);check,*_=run(n,dt/2,.02)
    r['dt_halving_error_change']=abs(r['velocity_rms']-check['velocity_rms'])
    r['temporal_change_fraction']=r['dt_halving_error_change']/r['velocity_rms']
    spatial.append(r)
    print('spatial',n,flush=True)
temporal=[]
for dt in (.02,.01,.005,.0025):
    r,*_=run(64,dt,.5);temporal.append(r)
translated=[]
for n,dt in [(32,.002),(64,.001),(128,.0005)]:
    r,*_=run(n,dt,.2,background=(.3,.2));translated.append(r)
    print('translated',n,flush=True)
for rows,key in [(spatial,'velocity_rms'),(temporal,'semidiscrete_velocity_rms')]:
    for k,r in enumerate(rows):r['order']=None if k==0 else float(np.log2(rows[k-1][key]/r[key]))
for name,rows in [('spatial',spatial),('temporal',temporal),('translated',translated)]:
    with (out/f'{name}.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
r,g,history,frames=run(64,.005,2.,snapshots=True)
# Include initial energy in history.
history=[(0.,.25,.25)]+history
np.savetxt(out/'energy.csv',history,delimiter=',',header='t,numerical_energy,exact_energy',comments='')
vdir=out/'paraview';vdir.mkdir(exist_ok=True);refs=[]
for k,(t,u,v,p) in enumerate(frames):
    name=f'flow_{k:03d}.vti';write_vti(vdir/name,g,u,v,p);refs.append((t,name))
write_pvd(vdir/'flow.pvd',refs)
np.savez_compressed(out/'snapshots.npz',t=np.array([f[0] for f in frames]),u=np.stack([f[1] for f in frames]),v=np.stack([f[2] for f in frames]),p=np.stack([f[3] for f in frames]),nu=nu,hx=g.hx,hy=g.hy)
fig,axes=plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
axes[0,0].loglog([2*np.pi/a['n'] for a in spatial],[a['velocity_rms'] for a in spatial],'o-',label='velocity')
axes[0,0].loglog([2*np.pi/a['n'] for a in spatial],[a['pressure_rms'] for a in spatial],'s-',label='pressure')
axes[0,0].set(xlabel='h',ylabel='RMS error',title='Full-flow spatial verification, T=0.02');axes[0,0].legend()
axes[0,1].loglog([a['dt'] for a in temporal],[a['semidiscrete_velocity_rms'] for a in temporal],'o-',label='vs semidiscrete exact')
axes[0,1].loglog([a['dt'] for a in temporal],[a['velocity_rms'] for a in temporal],'s-',label='vs continuum exact')
axes[0,1].set(xlabel='Time step',ylabel='Velocity RMS error',title='Temporal error and spatial floor');axes[0,1].legend(fontsize=8)
a=np.array(history);axes[1,0].plot(a[:,0],a[:,1],'o',ms=3,label='numerical');axes[1,0].plot(a[:,0],a[:,2],label='exact')
axes[1,0].set(xlabel='Time',ylabel='Mean kinetic energy',title='Taylor-Green energy decay');axes[1,0].legend()
u,v,p=frames[-1][1:];_,_,w=centered_fields(g,u,v);x,y=g.coordinates()
im=axes[1,1].pcolormesh(x,y,w,cmap='RdBu_r',vmin=-2,vmax=2,shading='auto');fig.colorbar(im,ax=axes[1,1],label='Vorticity')
axes[1,1].set(xlabel='x',ylabel='y',title='Computed vorticity, T=2');axes[1,1].set_aspect('equal')
for ax in axes.flat:ax.grid(alpha=.15)
for ext in ('png','pdf','svg'):fig.savefig(root/f'figures/phase2-verification.{ext}',dpi=170)
plt.close(fig)
fig,ax=plt.subplots(figsize=(5.5,4.6),constrained_layout=True)
im=ax.imshow(centered_fields(g,*frames[0][1:3])[2],origin='lower',extent=[0,g.lx,0,g.ly],cmap='RdBu_r',vmin=-2,vmax=2)
fig.colorbar(im,ax=ax,label='Vorticity');ax.set(xlabel='x',ylabel='y')
def update(k):
    im.set_data(centered_fields(g,*frames[k][1:3])[2]);ax.set_title(f'Computed Taylor-Green flow, t={frames[k][0]:.2f}');return [im]
FuncAnimation(fig,update,frames=len(frames)).save(root/'figures/phase2-taylor-green.gif',writer=PillowWriter(fps=8));plt.close(fig)
summary={'nu':nu,'spatial':spatial,'temporal':temporal,'translated':translated,'trajectory':r,'elapsed_seconds':time.perf_counter()-start,'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform()}
(out/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
