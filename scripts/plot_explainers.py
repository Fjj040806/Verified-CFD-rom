"""Coordinated scientific explainers; synthetic projection uses actual operators."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Circle
from matplotlib.colors import LinearSegmentedColormap
from flowrom import PeriodicMAC
out=Path(__file__).resolve().parents[1]/'figures/explainers';out.mkdir(exist_ok=True)
NAVY='#243746';TEAL='#318B88';CLAY='#CC8868';MUTED='#71808A';LINE='#DDE5E7';PAPER='#FBFCFB';PALE='#EDF5F3'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':NAVY,'mathtext.fontset':'dejavusans','svg.fonttype':'none'})
def canvas(number,title,subtitle,height=6.8):
 fig=plt.figure(figsize=(12,height),facecolor=PAPER)
 fig.text(.055,.937,'FLOW / NUMERICAL METHODS',fontsize=9,color=TEAL,weight='bold')
 fig.text(.945,.937,number,fontsize=10,color=MUTED,ha='right')
 fig.text(.055,.866,title,fontsize=23,weight='bold')
 fig.text(.055,.816,subtitle,fontsize=10.5,color=MUTED)
 fig.add_artist(plt.Line2D([.055,.945],[.785,.785],transform=fig.transFigure,color=LINE,lw=1))
 return fig
def save(fig,name):
 for ext in ('png','svg','pdf'):fig.savefig(out/f'{name}.{ext}',dpi=220,facecolor=PAPER)
 plt.close(fig)
def footer(fig,text):fig.text(.055,.04,text,fontsize=9,color=MUTED)
# 1: graphic emphasis on one cell, restrained neighbouring grid.
fig=canvas('01','Where the flow variables live','MAC staggered grid · pressure at cell centres, velocity components on cell faces')
ax=fig.add_axes([.055,.17,.43,.56]);ax.set(xlim=(-.2,3.2),ylim=(-.2,3.2),aspect='equal');ax.axis('off')
for i in range(4):ax.plot([0,3],[i,i],color=LINE,lw=1);ax.plot([i,i],[0,3],color=LINE,lw=1)
ax.add_patch(FancyBboxPatch((1,1),1,1,boxstyle='square,pad=0',facecolor=PALE,edgecolor=TEAL,lw=1.7,zorder=2))
for j in range(3):
 for i in range(3):ax.plot(i+.5,j+.5,'o',color=NAVY,alpha=1 if (i,j)==(1,1) else .25,ms=5,zorder=3)
for j in range(3):
 for i in range(4):ax.plot(i,j+.5,'>',color=TEAL,alpha=.3,ms=6)
for j in range(4):
 for i in range(3):ax.plot(i+.5,j,'^',color=CLAY,alpha=.3,ms=6)
ax.text(1.6,1.62,r'$p$',fontsize=18)
for a,b,label,col,offset in [((1,1.5),(.55,1.5),r'$u_L$',TEAL,(0,.16)),((2,1.5),(2.45,1.5),r'$u_R$',TEAL,(0,.16)),((1.5,1),(1.5,.55),r'$v_B$',CLAY,(.18,0)),((1.5,2),(1.5,2.45),r'$v_T$',CLAY,(.18,0))]:
 ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'-|>','color':col,'lw':1.6,'mutation_scale':13},zorder=4)
 ax.text(b[0]+offset[0],b[1]+offset[1],label,color=col,fontsize=13,ha='center',va='center')
ax.text(1.5,-.12,'Arrows indicate outward face normals',ha='center',fontsize=9,color=MUTED)
ax=fig.add_axes([.56,.17,.385,.56]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
ax.text(0,.93,'01 / BALANCE',fontsize=9,color=TEAL,weight='bold')
ax.text(0,.79,'Every cell balances its flux.',fontsize=17,weight='bold')
ax.text(0,.59,r'$D\mathbf{u}=\frac{u_R-u_L}{h_x}+\frac{v_T-v_B}{h_y}$',fontsize=19)
ax.text(0,.44,'Incompressibility means net outward flow is zero.',fontsize=10.5,color=MUTED)
ax.plot([0,1],[.34,.34],color=LINE,lw=1,transform=ax.transAxes)
ax.text(0,.23,'02 / PRESSURE COUPLING',fontsize=9,color=TEAL,weight='bold')
ax.text(0,.1,r'$G p\ \longrightarrow\ \mathrm{face\ correction}$',fontsize=18,color=TEAL)
ax.text(0,-.04,r'Compatible operators: $L_p=DG$',fontsize=11,color=MUTED)
footer(fig,'INCOMPRESSIBLE FLOW + ROM  /  Grid design schematic');save(fig,'01-mac-cell')
# 2: controlled synthetic projection.
g=PeriodicMAC(28,28);x,y=g.coordinates('u');u0=np.sin(x)*np.cos(y);x,y=g.coordinates('v');v0=-np.cos(x)*np.sin(y)
x,y=g.coordinates();gx,gy=g.gradient(.32*np.cos(x)*np.cos(y));us=u0+gx;vs=v0+gy;un,vn,phi,d=g.project(us,vs,1.)
fig=canvas('02','Pressure restores incompressibility','Subtract a gradient field to remove the predictor’s nonzero divergence',height=7.2)
cmap=LinearSegmentedColormap.from_list('teal-paper-clay',['#6EAAA7',PAPER,'#D7A083'])
fields=[(us,vs,'PREDICT',r'$\mathbf{u}^{*}$'),(gx,gy,'SUBTRACT',r'$\Delta t\,G\phi$'),(un,vn,'PROJECT',r'$\mathbf{u}^{n+1}$')]
xc,yc=g.coordinates();limit=max(np.max(np.abs(g.divergence(a,b))) for a,b,*_ in fields)
for k,(u,v,label,formula) in enumerate(fields):
 left=.055+k*.31
 fig.text(left,.728,f'0{k+1} / {label}',fontsize=9,color=TEAL,weight='bold')
 fig.text(left,.685,formula,fontsize=18)
 ax=fig.add_axes([left,.245,.27,.405]);div=g.divergence(u,v)
 im=ax.pcolormesh(xc,yc,div,cmap=cmap,vmin=-limit,vmax=limit,shading='auto',rasterized=True)
 uc=.5*(u+np.roll(u,-1,axis=1));vc=.5*(v+np.roll(v,-1,axis=0));skip=(slice(1,None,2),slice(1,None,2))
 ax.quiver(xc[skip],yc[skip],uc[skip],vc[skip],color=NAVY,scale=18,width=.0035,headwidth=3.4)
 ax.set(xlim=(0,g.lx),ylim=(0,g.ly),aspect='equal');ax.set_xticks([]);ax.set_yticks([])
 for spine in ax.spines.values():spine.set_color(LINE);spine.set_linewidth(.8)
 fig.text(left,.199,'DIVERGENCE RMS',fontsize=8.5,color=MUTED)
 fig.text(left,.155,f'{np.sqrt(np.mean(div**2)):.2e}',fontsize=17,color=TEAL if k==2 else NAVY)
for xloc,symbol in [(.345,'−'),(.655,'=')]:fig.text(xloc,.445,symbol,fontsize=23,color=MUTED,ha='center')
cax=fig.add_axes([.725,.092,.20,.014]);cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[-limit,0,limit]);cb.ax.set_xticklabels(['negative','zero','positive'],fontsize=8);cb.outline.set_visible(False)
fig.text(.055,.092,'Arrows: velocity  ·  Colour: divergence',fontsize=9,color=MUTED)
footer(fig,'Synthetic periodic projection computed with the project solver · velocities interpolated to cell centres');save(fig,'02-pressure-projection')
# 3: coherent cards and thinner connectors.
fig=canvas('03','One step of the flow solver','Incremental pressure correction · periodic boundaries · first-order time integration',height=7.8)
ax=fig.add_axes([.04,.13,.92,.61]);ax.set(xlim=(0,12),ylim=(0,7));ax.axis('off')
def card(x,y,n,title,body,formula,col):
 w,h=5.1,2.7
 ax.add_patch(FancyBboxPatch((x+.035,y-.04),w,h,boxstyle='round,pad=.05,rounding_size=.14',facecolor='#E7ECEC',edgecolor='none',zorder=0))
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.05,rounding_size=.14',facecolor='white',edgecolor=LINE,lw=.9))
 ax.add_patch(Circle((x+.42,y+h-.48),.21,facecolor=col,edgecolor='none'))
 ax.text(x+.42,y+h-.48,str(n),ha='center',va='center',color='white',fontsize=10,weight='bold')
 ax.text(x+.8,y+h-.48,title,fontsize=14,weight='bold',va='center')
 ax.text(x+.3,y+h-1.02,body,fontsize=10.5,color=MUTED,va='top',linespacing=1.65)
 ax.text(x+.3,y+.4,formula,fontsize=15,color=col,va='center')
def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'-|>','color':'#97A6AD','lw':1.2,'mutation_scale':12})
card(.35,3.95,1,'Predict velocity','Explicit convection; implicit viscosity\nInclude the previous pressure',r'$(I-\Delta t\,\nu L)\mathbf{u}^{*}=\cdots$',TEAL)
card(6.55,3.95,2,'Solve pressure correction','Measure the predictor divergence\nSolve the pressure Poisson equation',r'$L_p\phi=D\mathbf{u}^{*}/\Delta t$',CLAY)
card(6.55,.35,3,'Correct the flow','Update velocity and pressure\nSet the pressure mean to zero',r'$\mathbf{u}^{n+1}=\mathbf{u}^{*}-\Delta t\,G\phi$',TEAL)
card(.35,.35,4,'Check and record','Monitor divergence, residual and CFL\nTrack energy and validation errors',r'$D\mathbf{u}^{n+1}=\Delta t\,r$',NAVY)
arrow((5.6,5.3),(6.35,5.3));arrow((9.1,3.75),(9.1,3.25));arrow((6.35,1.7),(5.6,1.7));arrow((2.9,3.25),(2.9,3.75))
ax.text(3.13,3.5,'next step',fontsize=9,color=MUTED,va='center')
footer(fig,'INCOMPRESSIBLE FLOW + ROM  /  Diagnose numerical error before advancing');save(fig,'03-time-step')
print('Refined figures generated in PNG, SVG and PDF.')
