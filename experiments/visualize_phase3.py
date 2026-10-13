from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from flowrom.cavity import Cavity
root=Path(__file__).resolve().parents[1];out=root/'results/phase3';d=np.load(out/'cavity-128-base.npz');n=128;c=Cavity(n)
x=(np.arange(n)+.5)/n;X,Y=np.meshgrid(x,x);uc,vc=d['uc'],d['vc'];omega=np.gradient(vc,c.h,axis=1)-np.gradient(uc,c.h,axis=0)
# Refined presentation of genuinely computed flow; no imagegen data.
fig,ax=plt.subplots(figsize=(8,7),facecolor='#081D2C',constrained_layout=True);ax.set_facecolor('#081D2C')
from matplotlib.colors import LinearSegmentedColormap
cm=LinearSegmentedColormap.from_list('flow',['#102838','#1B5D6A','#2EA7A0','#9AE3CE','#F3BC81'])
im=ax.pcolormesh(X,Y,np.hypot(uc,vc),cmap=cm,shading='auto',vmin=0,vmax=1)
ax.streamplot(x,x,uc,vc,density=2.,color='#CCEDE5',linewidth=.55,arrowsize=.65)
ax.annotate('',xy=(.85,1.025),xytext=(.15,1.025),arrowprops={'arrowstyle':'->','color':'#F3BC81','lw':2},annotation_clip=False)
ax.text(.5,1.065,'MOVING LID  /  U = 1',color='#F3BC81',ha='center',fontsize=10)
ax.set(xlim=(0,1),ylim=(0,1),xlabel='x / L',ylabel='y / L',aspect='equal');ax.tick_params(colors='#AFCDC9');ax.xaxis.label.set_color('#AFCDC9');ax.yaxis.label.set_color('#AFCDC9')
for s in ax.spines.values():s.set_color('#577F85')
cb=fig.colorbar(im,ax=ax,pad=.03,shrink=.8);cb.set_label('Computed speed / U lid',color='#AFCDC9');cb.ax.tick_params(colors='#AFCDC9')
fig.suptitle('Lid-driven cavity · Re = 100 · 128² cells',color='#E5F4EF',fontsize=15)
for ext in ('png','svg','pdf'):fig.savefig(root/f'figures/phase3-flow-styled.{ext}',dpi=200,facecolor=fig.get_facecolor())
plt.close(fig)
frames=np.load(out/'transient-64.npz');fig,ax=plt.subplots(figsize=(6,5),constrained_layout=True);ax.set(xlabel='x / L',ylabel='y / L',aspect='equal')
im=ax.imshow(np.hypot(frames['uc'][0],frames['vc'][0]),origin='lower',extent=(0,1,0,1),cmap=cm,vmin=0,vmax=1);fig.colorbar(im,ax=ax,label='Speed / U lid')
def update(k):
 im.set_data(np.hypot(frames['uc'][k],frames['vc'][k]));ax.set_title(f'Computed cavity startup · t = {frames["t"][k]:.2f}');return [im]
FuncAnimation(fig,update,frames=len(frames['t'])).save(root/'figures/phase3-cavity-startup.gif',writer=PillowWriter(fps=10));plt.close(fig)
# Single steady VTI: cell-centred samples as points, native divergence retained.
vdir=out/'paraview';vdir.mkdir(exist_ok=True);extent=f'0 {n-1} 0 {n-1} 0 0'
r=ET.Element('VTKFile',type='ImageData',version='0.1',byte_order='LittleEndian');a=ET.SubElement(r,'ImageData',WholeExtent=extent,Origin=f'{c.h/2} {c.h/2} 0',Spacing=f'{c.h} {c.h} 1');piece=ET.SubElement(a,'Piece',Extent=extent);pd=ET.SubElement(piece,'PointData',Scalars='speed',Vectors='velocity')
for name,field,nc in [('speed',np.hypot(uc,vc),1),('pressure',d['p'],1),('velocity',np.stack([uc,vc,np.zeros_like(uc)],axis=-1),3),('vorticity',omega,1),('divergence',c.div(d['u'],d['v']),1)]:
 item=ET.SubElement(pd,'DataArray',type='Float64',Name=name,NumberOfComponents=str(nc),format='ascii');item.text=' '.join(f'{v:.16g}' for v in field.ravel())
ET.SubElement(piece,'CellData');ET.ElementTree(r).write(vdir/'cavity-steady.vti',encoding='utf-8',xml_declaration=True)
from vtkmodules.vtkIOXML import vtkXMLImageDataReader
reader=vtkXMLImageDataReader();reader.SetFileName(str(vdir/'cavity-steady.vti'));reader.Update();assert reader.GetOutput().GetNumberOfPoints()==n*n
print('Styled physical flow, startup GIF and VTK-validated ParaView file generated.')
