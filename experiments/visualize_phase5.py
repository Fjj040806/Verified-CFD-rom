"""Compare held-out nodal velocity speeds; linear display interpolation only."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.patches import Circle
root=Path(__file__).resolve().parents[1]
d=np.load(root/'results/phase5/models-and-reconstructions.npz');raw=np.load(root/'results/phase4b/coarse-half-snapshots.npz' if (root/'results/phase4b/coarse-half-snapshots.npz').exists() else root/'data/demo/re100-settled.npz');mesh=np.load(root/'results/phase4b/coarse-half-mesh.npz')
i=int(np.argmin(abs(d['t']-29.7)));time=d['t'][i];j=int(np.argmin(abs(raw['t']-time)));k=int(np.argmin(abs(d['forecast_times']-time)))
computed=raw['u'][j];reconstructed=d['reconstruction'][:,i];predicted=d['forecast'][:,k];nodes=mesh['nodal_dofs'];xy=mesh['points'];tr=tri.Triangulation(*xy,mesh['triangles'].T)
speed=lambda u:np.linalg.norm(u[nodes],axis=0)
fields=[speed(computed),speed(reconstructed),speed(predicted),speed(predicted-computed)]
fig,axes=plt.subplots(4,1,figsize=(12,8),constrained_layout=True);fig.set_facecolor('#FBFCFB')
labels=['Computed CFD','10-mode POD reconstruction','20-mode affine DMD forecast','DMD vector-velocity error magnitude']
for index,(ax,field,label) in enumerate(zip(axes,fields,labels)):
 vmax=2.2 if index<3 else max(fields[3].max(),1e-8)
 image=ax.tripcolor(tr,field,shading='gouraud',cmap='viridis' if index<3 else 'magma',vmin=0,vmax=vmax)
 ax.add_patch(Circle((.2,.2),.05,facecolor='#FBFCFB',edgecolor='#243746',lw=.7));ax.set_aspect('equal');ax.set(xlim=(0,2.2),ylim=(0,.41),ylabel='y',title=label);fig.colorbar(image,ax=ax,shrink=.75,label='Speed' if index<3 else '|u forecast − u CFD|')
axes[-1].set_xlabel('x');fig.suptitle(f'Held-out flow at t = {time:.2f} · identical speed scales in first three panels',fontsize=14)
for ext in ('png','svg','pdf'):fig.savefig(root/f'figures/phase5-flow-comparison.{ext}',dpi=180)
