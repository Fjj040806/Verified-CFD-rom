"""Exact operator schematic and data-backed POD/DMD diagnostic figures."""
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation,PillowWriter
from flowrom.rom import WeightedPOD,ReducedDMD
root=Path(__file__).resolve().parents[1];teal='#318B88';clay='#CC8868';ink='#243746';gray='#788393'
fig,axs=plt.subplots(1,2,figsize=(11,4.4),constrained_layout=True)
ax=axs[0]
for x in range(4):ax.plot([x,x],[0,3],color=gray,alpha=.25)
for y in range(4):ax.plot([0,3],[y,y],color=gray,alpha=.25)
for i in range(3):
 for j in range(3):ax.scatter(i+.5,j+.5,color=ink,s=25)
for x,y,label,color in [(1.5,1.5,'$p_{i,j}$',ink),(1,1.5,'$u_{i-1/2,j}$',teal),(2,1.5,'$u_{i+1/2,j}$',teal),(1.5,1,'$v_{i,j-1/2}$',clay),(1.5,2,'$v_{i,j+1/2}$',clay)]:
 ax.scatter(x,y,color=color,s=75,zorder=3);ax.annotate(label,(x,y),xytext=(5,9),textcoords='offset points',fontsize=10,color=color)
ax.annotate('',xy=(2,1.5),xytext=(1,1.5),arrowprops=dict(arrowstyle='->',color=teal,lw=1.6));ax.annotate('',xy=(1.5,2),xytext=(1.5,1),arrowprops=dict(arrowstyle='->',color=clay,lw=1.6));ax.set_aspect('equal');ax.set(xlim=(-.15,3.15),ylim=(-.15,3.15),title='A · Compatible MAC locations');ax.set_xticks([]);ax.set_yticks([])
ax=axs[1];ax.axis('off');ax.set_title('B · Projection geometry and residual')
for y,text in [(0.86,r'$G=-D_h^*\quad\Longrightarrow\quad -D_hG\succeq0$'),(.65,r'$P=I-G(D_hG)^\dagger D_h$'),(.43,r'$\|P\mathbf{u}\|_h^2+\|(I-P)\mathbf{u}\|_h^2=\|\mathbf{u}\|_h^2$'),(.2,r'$D_h\mathbf{u}^{n+1}=\Delta t\,r$')]:ax.text(.04,y,text,fontsize=15,color=ink,transform=ax.transAxes)
ax.text(.04,.03,'Periodic grid · compatible quadrature · exact pressure solve\nWith an inexact solve, the final identity quantifies continuity error.',fontsize=10,color=gray,transform=ax.transAxes)
for ext in ['png','pdf','svg']:fig.savefig(root/f'figures/math-projection.{ext}',dpi=180)
plt.close(fig)
d=np.load(root/'data/demo/re100-settled.npz');t=d['t'];U=d['u'].T;M=load_npz(root/'data/demo/velocity-mass.npz');mesh=np.load(root/'results/phase4b/coarse-half-mesh.npz');nt=int(.6*len(t));pod=WeightedPOD(U[:,:nt],M);model=ReducedDMD(pod,U[:,:nt],20,float(np.median(np.diff(t))));tri=Triangulation(*mesh['points'],mesh['triangles'].T);nodes=mesh['nodal_dofs']
fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
for j,ax in enumerate(axs[0]):
 raw=pod.modes[nodes[0],j];normalized=raw/np.max(np.abs(raw));im=ax.tripcolor(tri,normalized,shading='gouraud',cmap='BrBG',vmin=-1,vmax=1);ax.add_patch(Circle((.2,.2),.05,facecolor='white',edgecolor=ink,lw=.8));ax.set_aspect('equal');ax.set(xlabel='x',ylabel='y',title=f'{chr(65+j)} · POD mode {j+1}: normalized horizontal component');fig.colorbar(im,ax=ax,shrink=.65,label='Display-normalized mode')
c=pod.coefficients(U,2);ax=axs[1,0]
for s,label,color in [(slice(0,nt),'Training',teal),(slice(nt,nt+60),'Validation',gray),(slice(nt+60,None),'Test',clay)]:ax.plot(c[0,s],c[1,s],color=color,lw=1.5,label=label)
ax.set(xlabel='First POD coefficient',ylabel='Second POD coefficient',title='C · Actual coefficients on the settled trajectory');ax.legend(fontsize=8);ax.grid(alpha=.15)
ax=axs[1,1];angle=np.linspace(0,2*np.pi,300);ax.plot(np.cos(angle),np.sin(angle),':',color=gray,label='Unit circle');ax.scatter(model.eigenvalues.real,model.eigenvalues.imag,s=30,color=teal,label='Affine DMD eigenvalues');ax.axhline(0,lw=.4,color=gray);ax.axvline(0,lw=.4,color=gray);ax.set_aspect('equal');ax.set(xlim=(-1.12,1.12),ylim=(-1.12,1.12),xlabel='Real part',ylabel='Imaginary part',title='D · Rank-20 discrete-time spectrum');ax.legend(fontsize=8);ax.grid(alpha=.15)
for ext in ['png','pdf','svg']:fig.savefig(root/f'figures/math-modes-spectrum.{ext}',dpi=180)
plt.close(fig)
fig,ax=plt.subplots(figsize=(8.2,2.9),constrained_layout=True);fig.set_facecolor('#FBFCFB');speed=np.linalg.norm(U[nodes,0],axis=0);im=ax.tripcolor(tri,speed,shading='gouraud',cmap='viridis',vmin=0,vmax=2.2);ax.add_patch(Circle((.2,.2),.05,facecolor='white',edgecolor=ink,lw=.8));ax.set_aspect('equal');ax.set(xlabel='x',ylabel='y',xlim=(0,2.2),ylim=(0,.41));fig.colorbar(im,ax=ax,shrink=.65,label='Velocity magnitude');title=ax.set_title('Computed Re100 cylinder wake')
frames=np.arange(0,len(t),5)
def update(k):
 im.set_array(np.linalg.norm(U[nodes,k],axis=0));title.set_text(f'Computed Re100 cylinder wake · t = {t[k]:.2f}');return im,title
movie=FuncAnimation(fig,update,frames=frames,interval=100,blit=False);movie.save(root/'figures/wake-preview.gif',writer=PillowWriter(fps=10),dpi=95);plt.close(fig)
print('Generated MAC schematic, POD modes/spectrum, and a data-backed wake preview.')
