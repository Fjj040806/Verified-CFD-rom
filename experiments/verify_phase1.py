"""Reproducible actual numerical verification; no Navier-Stokes integration."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flowrom import PeriodicMAC

root=Path(__file__).resolve().parents[1]
rows=[]
for n in (16,32,64,128):
    g=PeriodicMAC(n,n); x,y=g.coordinates()
    exact=np.sin(x)*np.cos(2*y)
    phi,d=g.poisson(-5*exact)
    gx,gy=g.gradient(exact)
    xu,yu=g.coordinates('u');xv,yv=g.coordinates('v')
    u=np.sin(xu)*np.cos(2*yu);v=np.cos(xv)*np.sin(2*yv)
    div_exact=np.cos(x)*np.cos(2*y)+2*np.cos(x)*np.cos(2*y)
    rms=lambda a:float(np.sqrt(np.mean(a*a)))
    rows.append(dict(n=n,h=g.hx,poisson_l2=rms(phi-exact),
       gradient_l2=np.sqrt(rms(gx-np.cos(xu)*np.cos(2*yu))**2+rms(gy+2*np.sin(xv)*np.sin(2*yv))**2),
       divergence_l2=rms(g.divergence(u,v)-div_exact),
       laplacian_l2=rms(g.laplacian(exact)+5*exact),residual=d['absolute_residual']))
for idx,row in enumerate(rows):
    row['poisson_order']=None if idx==0 else float(np.log2(rows[idx-1]['poisson_l2']/row['poisson_l2']))
with (root/'results/phase1-convergence.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
g=PeriodicMAC(64,64);rng=np.random.default_rng(42)
u,v=[rng.normal(size=(64,64)) for _ in range(2)]
records=[]
for tol in (1e-3,1e-5,1e-7,1e-9,1e-11):
    _,_,_,d=g.project(u,v,.02,method='cg',rtol=tol)
    records.append(dict(rtol=tol,**d))
_,_,_,fft=g.project(u,v,.02)
(root/'results/phase1-projection.json').write_text(json.dumps({'cg':records,'fft':fft},indent=2))
fig,axes=plt.subplots(1,2,figsize=(11,4.4),constrained_layout=True)
for key,label in [('poisson_l2','Poisson solution'),('gradient_l2','Gradient'),('divergence_l2','Divergence'),('laplacian_l2','Laplacian')]:
    axes[0].loglog([r['h'] for r in rows],[r[key] for r in rows],'o-',label=label)
h=np.array([r['h'] for r in rows]);axes[0].loglog(h,.2*h*h,'k--',label='O(h²) reference')
axes[0].set(xlabel='Grid spacing h',ylabel='RMS error',title='Periodic analytic verification');axes[0].legend(fontsize=8)
axes[1].loglog([r['rtol'] for r in records],[r['divergence_rms'] for r in records],'o-')
axes[1].set(xlabel='Requested CG relative tolerance',ylabel='Corrected divergence RMS',title='Projection tolerance experiment (64²)')
for ax in axes:ax.grid(True,which='both',alpha=.2)
for ext in ('png','svg','pdf'):fig.savefig(root/f'figures/phase1-verification.{ext}',dpi=180)
print(json.dumps({'convergence':rows,'projection':records,'fft':fft},indent=2))
