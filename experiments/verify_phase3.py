from pathlib import Path
import json,csv,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flowrom.cavity import Cavity
root=Path(__file__).resolve().parents[1];out=root/'results/phase3';out.mkdir(exist_ok=True)
# Ghia et al. 1982 Table I/II, Re=100; provenance in report.
yref=np.array([1,.9766,.9688,.9609,.9531,.8516,.7344,.6172,.5,.4531,.2813,.1719,.1016,.0703,.0625,.0547,0])
uref=np.array([1,.84123,.78871,.73722,.68717,.23151,.00332,-.13641,-.20581,-.21090,-.15662,-.10150,-.06434,-.04775,-.04192,-.03717,0])
xref=np.array([1,.9688,.9609,.9531,.9453,.9063,.8594,.8047,.5,.2344,.2266,.1563,.0938,.0781,.0703,.0625,0])
vref=np.array([0,-.05906,-.07391,-.08864,-.10313,-.16914,-.22445,-.24533,.05454,.17527,.17507,.16077,.12317,.10890,.10091,.09233,0])

def run(n,dt,label):
 c=Cavity(n);u,v,p=c.initial();history=[];t=0;start=time.perf_counter();consecutive=0;frames=[]
 maxsteps=int(60/dt);window=max(1,round(.5/dt));uold=u.copy();vold=v.copy()
 for step in range(1,maxsteps+1):
  u,v,p=c.step(u,v,p,dt);t=step*dt
  if step%window==0:
   rate=np.sqrt((np.mean((u-uold)**2)+np.mean((v-vold)**2))/2)/(window*dt)
   div=float(np.sqrt(np.mean(c.div(u,v)**2)));uc,vc=c.centered(u,v)
   history.append([t,rate,div,.5*np.mean(uc**2+vc**2)]);uold=u.copy();vold=v.copy()
   consecutive=consecutive+1 if rate<1e-7 else 0
   if n==64 and label=='base' and len(frames)<80:frames.append((t,uc.copy(),vc.copy()))
   if step%(window*10)==0:print(label,n,'t',round(t,2),'rate',rate,flush=True)
   if consecutive>=4:break
 if consecutive<4:raise RuntimeError(f'No steady state at n={n}')
 uc,vc=c.centered(u,v)
 # u at x=.5 is native face; v at y=.5 is native face.
 coordinates=np.r_[0,(np.arange(n)+.5)/n,1]
 up=np.r_[0,u[:,n//2],1];vp=np.r_[0,v[n//2,:],0]
 errors={'u_rms':float(np.sqrt(np.mean((np.interp(yref,coordinates,up)-uref)**2))),'v_rms':float(np.sqrt(np.mean((np.interp(xref,coordinates,vp)-vref)**2)))}
 np.savez_compressed(out/f'cavity-{n}-{label}.npz',u=u,v=v,p=p,uc=uc,vc=vc,coords=coordinates,u_profile=up,v_profile=vp,dt=dt,re=100)
 np.savetxt(out/f'history-{n}-{label}.csv',history,delimiter=',',header='t,steady_rate,divergence_rms,energy',comments='')
 result=dict(n=n,dt=dt,t=t,steps=step,rate=history[-1][1],divergence=history[-1][2],seconds=time.perf_counter()-start,**errors)
 print('FINISHED',result,flush=True);return result,frames
results=[]
for n,dt in [(32,.006),(64,.003),(128,.0015)]:
 r,frames=run(n,dt,'base');results.append(r)
 if n==64:np.savez_compressed(out/'transient-64.npz',t=np.array([f[0] for f in frames]),uc=np.stack([f[1] for f in frames]),vc=np.stack([f[2] for f in frames]))
r,_=run(64,.0015,'half-dt');results.append(r)
(root/'results/phase3/summary.json').write_text(json.dumps(results,indent=2))
navy='#243746';teal='#318B88';clay='#CC8868'
plt.rcParams.update({'font.family':'DejaVu Sans','text.color':navy,'axes.labelcolor':navy,'axes.titlecolor':navy,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(11,9),constrained_layout=True);fig.set_facecolor('#FBFCFB')
for r,color in zip(results[:3],['#A5C8C2',teal,navy]):
 d=np.load(out/f"cavity-{r['n']}-base.npz");axs[0,0].plot(d['u_profile'],d['coords'],color=color,label=f"{r['n']}²")
 axs[0,1].plot(d['coords'],d['v_profile'],color=color,label=f"{r['n']}²")
 h=np.loadtxt(out/f"history-{r['n']}-base.csv",delimiter=',',skiprows=1);axs[1,1].semilogy(h[:,0],h[:,1],color=color,label=f"{r['n']}²")
axs[0,0].scatter(uref,yref,color=clay,s=20,label='Ghia et al.');axs[0,1].scatter(xref,vref,color=clay,s=20,label='Ghia et al.')
axs[0,0].set(xlabel='u / U lid',ylabel='y / L',title='Vertical centreline · x = 0.5');axs[0,1].set(xlabel='x / L',ylabel='v / U lid',title='Horizontal centreline · y = 0.5')
d=np.load(out/'cavity-128-base.npz');x=(np.arange(128)+.5)/128;X,Y=np.meshgrid(x,x);speed=np.hypot(d['uc'],d['vc'])
im=axs[1,0].pcolormesh(X,Y,speed,cmap='viridis',shading='auto');axs[1,0].streamplot(x,x,d['uc'],d['vc'],density=1.25,color='white',linewidth=.5)
axs[1,0].set(xlabel='x / L',ylabel='y / L',title='Computed steady flow · Re = 100',aspect='equal');fig.colorbar(im,ax=axs[1,0],label='Speed / U lid')
axs[1,1].axhline(1e-7,color=clay,ls='--',lw=1,label='stopping threshold');axs[1,1].set(xlabel='Time / (L/U lid)',ylabel='Windowed velocity change rate',title='Approach to steady state')
for ax in [axs[0,0],axs[0,1],axs[1,1]]:ax.legend(fontsize=9);ax.grid(alpha=.15)
for ext in ('png','svg','pdf'):fig.savefig(root/f'figures/phase3-cavity.{ext}',dpi=180)
print('All cavity experiments complete.',flush=True)
