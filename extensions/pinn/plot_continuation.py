"""Scientific field rendering on the saved P2 finite-element mesh."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from skfem import MeshTri,Basis,ElementTriP2,ElementVector
p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);args=p.parse_args();here=Path(__file__).resolve().parent;out=here/'phase3-results'
op=np.load(here/'phase1-results/reduced-operators.npz');new=np.load(out/'predictions.npz');old=np.load(here/'phase2-results/predictions.npz');truth=np.load(args.baseline/'data/demo/re100-settled.npz')['u'][180:]
saved=np.load(args.baseline/'results/phase4b/coarse-half-mesh.npz');mesh=MeshTri(saved['points'],saved['triangles']);vb=Basis(mesh,ElementVector(ElementTriP2()));scalar=Basis(mesh,ElementTriP2());split=vb.split_indices();assert np.array_equal(vb.doflocs,saved['velocity_doflocs'])
x=np.linspace(.001,2.199,330);y=np.linspace(.001,.409,70);xx,yy=np.meshgrid(x,y);valid=(xx-.2)**2+(yy-.2)**2>.052**2;points=np.array([xx[valid],yy[valid]]);P=scalar.probes(points)
keys=['truth','continuation_b0_s0','continuation_b1_s0','affine_dmd_assimilation'];labels=['CFD reference','Data-only continuation (seed 0)','Physics continuation (beta=1, seed 0)','Sensor-assimilated DMD']
fields={};vfields={}
for key in keys:
 if key=='truth':u=truth
 else:
  a=new[key] if key in new else old[key];u=op['mean']+a@op['modes'][:,:10].T
 fields[key]=np.stack([(P@u[:,idx].T).T for idx in split],axis=-1)
 vfields[key]=fields[key][:,:,1]
def grid(v):
 a=np.full(xx.shape,np.nan);a[valid]=v;return a
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def decorate(ax):
 ax.add_patch(plt.Circle((.2,.2),.05,color='#263b50'));ax.set(xlim=(0,2.2),ylim=(0,.41),ylabel='y');ax.set_aspect('equal')
# Snapshot fixed in advance at last assimilated time, not selected on error.
k=59;fig,axs=plt.subplots(2,3,figsize=(13,3.6),constrained_layout=True)
for j,key in enumerate(keys[:3]):
 im=axs[0,j].imshow(grid(vfields[key][k]),origin='lower',extent=[x[0],x[-1],y[0],y[-1]],cmap='RdBu_r',vmin=-.65,vmax=.65);decorate(axs[0,j]);axs[0,j].set_title(['CFD reference','Data-only | seed 0','Physics continuation | beta=1, seed 0'][j],fontsize=10)
for j,key in enumerate([keys[3],keys[1],keys[2]]):
 err=np.linalg.norm(fields[key][k]-fields['truth'][k],axis=-1);ie=axs[1,j].imshow(grid(err),origin='lower',extent=[x[0],x[-1],y[0],y[-1]],cmap='magma',vmin=0,vmax=.12);decorate(axs[1,j]);axs[1,j].set_title(['Error: sensor-assimilated DMD','Error: data-only continuation','Error: physics continuation'][j],fontsize=10);axs[1,j].set_xlabel('x')
fig.colorbar(im,ax=axs[0,:],label='Transverse velocity v',shrink=.85);fig.colorbar(ie,ax=axs[1,:],label='Velocity error magnitude',shrink=.85,extend='max');fig.suptitle('Re=100 cylinder wake | t=29.39 | fixed seed-zero examples')
for ext in ['png','svg','pdf']:fig.savefig(out/f'wake-reconstruction.{ext}',dpi=180)
plt.close(fig)
fig,axs=plt.subplots(4,1,figsize=(10,7.8),constrained_layout=True);ims=[]
for ax,key,label in zip(axs,keys,labels):
 ims.append(ax.imshow(grid(vfields[key][0]),origin='lower',extent=[x[0],x[-1],y[0],y[-1]],cmap='RdBu_r',vmin=-.65,vmax=.65,animated=True));decorate(ax);ax.set_title(label,fontsize=10)
fig.colorbar(ims[0],ax=axs,label='Transverse velocity v',shrink=.6);title=fig.suptitle('')
def update(i):
 for im,key in zip(ims,keys):im.set_data(grid(vfields[key][i]))
 title.set_text(f'Re=100 | t={28.8+.01*i:.2f} | '+('Sparse-observation window' if i<60 else 'Future extrapolation (no observations)'));return ims+[title]
FuncAnimation(fig,update,frames=list(range(0,121,3)),interval=130,blit=False).save(out/'wake-evolution.gif',writer=PillowWriter(fps=8),dpi=95);plt.close(fig)
from PIL import Image,ImageSequence
gifpath=out/'wake-evolution.gif'
with Image.open(gifpath) as im:
 frames=[f.convert('RGB').resize((760,592)).quantize(colors=64) for f in ImageSequence.Iterator(im)]
frames[0].save(gifpath,save_all=True,append_images=frames[1:],duration=125,loop=0,optimize=True)
# Paired errors, all seeds and arms.
m=json.loads((out/'metrics.json').read_text());fig,axs=plt.subplots(1,2,figsize=(9,3.5),constrained_layout=True)
for ax,window in zip(axs,['online','future']):
 for seed in [0,1,2]:
  vals=[100*next(q['relative_fluctuation_error'] for q in m if q['model']==f'continuation_b{b:g}_s{seed}' and q['window']==window) for b in [0,.1,1]]
  ax.plot([0,1,2],vals,'o-',label=f'Seed {seed}',lw=1.2)
 ax.set(xticks=[0,1,2],xticklabels=['0','0.1','1'],xlabel='Normalized physics weight beta',ylabel='Fluctuation error (%)',title=window.capitalize());ax.legend(frameon=False)
for ext in ['png','svg','pdf']:fig.savefig(out/f'continuation-comparison.{ext}',dpi=180)
print('FIGURES COMPLETE')
