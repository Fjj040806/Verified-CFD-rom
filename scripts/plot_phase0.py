"""Design schematic and exact Taylor-Green field; no solver results."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path(__file__).resolve().parents[1] / 'figures'
out.mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
ax = axes[0]
for k in range(4):
    ax.axhline(k, color='0.8', lw=1)
    ax.axvline(k, color='0.8', lw=1)
x,y=np.meshgrid(np.arange(3)+.5,np.arange(3)+.5)
ax.scatter(x,y,c='#333333',marker='o',label='p: cell centres')
x,y=np.meshgrid(np.arange(4),np.arange(3)+.5)
ax.scatter(x,y,c='#0072B2',marker='>',label='u: vertical faces')
x,y=np.meshgrid(np.arange(3)+.5,np.arange(4))
ax.scatter(x,y,c='#D55E00',marker='^',label='v: horizontal faces')
ax.set(xlim=(-.2,3.2),ylim=(-.2,3.2),xlabel='x / h',ylabel='y / h',title='MAC grid — design schematic')
ax.set_aspect('equal'); ax.legend(loc='upper left',fontsize=8)
ax=axes[1]
x=np.linspace(0,2*np.pi,100); X,Y=np.meshgrid(x,x)
u=np.sin(X)*np.cos(Y); v=-np.cos(X)*np.sin(Y)
omega=2*np.sin(X)*np.sin(Y)
im=ax.contourf(X,Y,omega,levels=np.linspace(-2,2,25),cmap='RdBu_r')
ax.streamplot(x,x,u,v,color='0.2',density=.7,linewidth=.6)
ax.set(xlabel='x',ylabel='y',title='Taylor-Green vortex — exact field, t=0')
ax.set_aspect('equal');fig.colorbar(im,ax=ax,label='Nondimensional vorticity')
for ext in ('png','svg','pdf'): fig.savefig(out/f'phase0-design.{ext}',dpi=180)
print('Saved design figure: analytical reference only.')
