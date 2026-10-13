from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation,LinearTriInterpolator
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
root=Path(__file__).resolve().parents[1];d=np.load(root/'results/phase4/fine.npz');points=d['points'];uv=d['vertex_velocity'];tri=Triangulation(*points,d['triangles'].T)
navy='#081D2C';pale='#D6F0E8';teal='#32A69F';gold='#F1BB80'
cm=LinearSegmentedColormap.from_list('flow',[navy,'#145369',teal,'#AFE2CA',gold])
fig,ax=plt.subplots(figsize=(14,4.4),facecolor=navy,constrained_layout=True);ax.set_facecolor(navy)
im=ax.tripcolor(tri,np.linalg.norm(uv,axis=0),shading='gouraud',cmap=cm,vmin=0,vmax=.45)
x=np.linspace(0,2.2,360);y=np.linspace(0,.41,90);X,Y=np.meshgrid(x,y)
ux=LinearTriInterpolator(tri,uv[0])(X,Y);uy=LinearTriInterpolator(tri,uv[1])(X,Y)
ax.streamplot(x,y,ux,uy,density=[2.4,.85],color=pale,linewidth=.5,arrowsize=.65)
ax.add_patch(Circle((.2,.2),.05,facecolor=navy,edgecolor=gold,lw=1.3,zorder=5))
ax.set(xlim=(0,2.2),ylim=(0,.41),xlabel='x',ylabel='y',aspect='equal');ax.tick_params(colors=pale);ax.xaxis.label.set_color(pale);ax.yaxis.label.set_color(pale)
for s in ax.spines.values():s.set_color('#4B777D')
cb=fig.colorbar(im,ax=ax,shrink=.55,pad=.02);cb.set_label('Speed',color=pale);cb.ax.tick_params(colors=pale)
fig.suptitle('Computed cylinder flow · DFG Re = 20 · fitted P2/P1 finite elements',color=pale,fontsize=14)
for ext in ('png','svg','pdf'):fig.savefig(root/f'figures/phase4-flow-styled.{ext}',dpi=200,facecolor=navy)
print('Styled cylinder view generated; streamlines use vertex interpolation for display.')
