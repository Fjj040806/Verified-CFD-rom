from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.tri import Triangulation
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation,PillowWriter
from skfem import MeshTri,Basis,ElementVector,ElementTriP2,ElementTriP1,LinearForm,asm
from skfem.models.poisson import mass
from scipy.sparse.linalg import splu
import meshio,xml.etree.ElementTree as ET
root=Path(__file__).resolve().parents[1];out=root/'results/phase4b';analysis=json.loads((out/'analysis.json').read_text());name=analysis['baseline']
d=np.load(out/f'{name}-snapshots.npz');m=np.load(out/f'{name}-mesh.npz');mesh=MeshTri(m['points'],m['triangles']);basis=Basis(mesh,ElementVector(ElementTriP2()),intorder=5);tri=Triangulation(*mesh.p,mesh.t.T)
indices=np.flatnonzero(d['t']>=d['t'][-1]-2.)[::2];t=d['t'][indices];U=d['u'][indices].astype(float);P=d['p'][indices].astype(float)
omega=[];omega_vertices=[]
pressure_basis=Basis(mesh,ElementTriP1(),intorder=5);projection_mass=asm(mass,pressure_basis);projection_solve=splu(projection_mass.tocsc())
@LinearForm
def vorticity_rhs(v,w):return w.omega*v
# P2 gradients are linear inside each affine triangle. Recover their exact
# reference-coordinate coefficients; duplicate vertices retain gradient jumps.
design=np.c_[np.ones(basis.X.shape[1]),basis.X.T];inverse=np.linalg.pinv(design)
coords=mesh.p[:,mesh.t.T.ravel()];display_tri=Triangulation(*coords,np.arange(mesh.nelements*3).reshape(-1,3))
for u in U:
 field=basis.interpolate(u);w=field.grad[1,0]-field.grad[0,1];omega.append(np.sum(w*basis.dx,axis=1)/basis.dx.sum(axis=1))
 coefficients=w@inverse.T
 assert np.allclose(coefficients@design.T,w,atol=1e-9,rtol=1e-9)
 rhs=asm(vorticity_rhs,pressure_basis,omega=w);projected=projection_solve.solve(rhs)
 assert np.linalg.norm(projection_mass@projected-rhs)<1e-10
 omega_vertices.append(projected)
omega=np.array(omega);omega_vertices=np.array(omega_vertices);history=np.genfromtxt(out/f'{name}-history.csv',delimiter=',',names=True)
navy='#081D2C';pale='#D6F0E8';gold='#F1BB80';cmap=LinearSegmentedColormap.from_list('vorticity',['#2EAAA7',navy,'#E4AA7F'])
fig=plt.figure(figsize=(12,5.4),facecolor=navy);ax=fig.add_axes([.07,.41,.80,.36]);ax.set_facecolor(navy)
im=ax.tripcolor(tri,omega_vertices[0],shading='gouraud',cmap=cmap,vmin=-25,vmax=25,rasterized=True);ax.add_patch(Circle((.2,.2),.05,facecolor=navy,edgecolor=gold,lw=1.2))
ax.set(xlim=(0,2.2),ylim=(0,.41),aspect='equal',xlabel='x',ylabel='y');ax.tick_params(colors=pale);ax.xaxis.label.set_color(pale);ax.yaxis.label.set_color(pale)
cax=fig.add_axes([.90,.43,.015,.29]);cb=fig.colorbar(im,cax=cax);cb.set_label('Vorticity (display clipped ±25)',color=pale);cb.ax.tick_params(colors=pale)
force=fig.add_axes([.07,.15,.80,.17]);force.set_facecolor(navy);late=history[history['t']>=t[0]];force.plot(late['t'],late['cl'],color='#64C6BB',lw=1);line=force.axvline(t[0],color=gold,lw=1.3);force.set(xlim=(t[0],t[-1]),xlabel='Physical time',ylabel='Lift coefficient');force.tick_params(colors=pale);force.xaxis.label.set_color(pale);force.yaxis.label.set_color(pale)
for axes in [ax,force]:
 for spine in axes.spines.values():spine.set_color('#517983')
title=fig.text(.07,.9,f'Computed vortex shedding · Re = 100 · t = {t[0]:.3f}',color=pale,fontsize=16)
fig.text(.07,.02,'P2/P1 finite-element solution  ·  L2-projected vorticity for display  ·  Gold cursor links flow and lift',color=pale,fontsize=9)
def update(k):
 im.set_array(omega_vertices[k]);line.set_xdata([t[k],t[k]]);title.set_text(f'Computed vortex shedding · Re = 100 · t = {t[k]:.3f}');return [im,line,title]
update(-1)
for ext in ('png','svg','pdf'):fig.savefig(root/f'figures/phase4b-vortex-shedding.{ext}',dpi=170,facecolor=navy)
FuncAnimation(fig,update,frames=len(t)).save(root/'figures/phase4b-vortex-shedding.gif',writer=PillowWriter(fps=12));plt.close(fig)
# ParaView viewing exports; complete coefficient arrays stay in NPZ.
vdir=out/'paraview';vdir.mkdir(exist_ok=True);refs=[]
for k in range(0,len(t),max(1,len(t)//30)):
 nodal=U[k][basis.nodal_dofs];filename=f'flow-{k:03d}.vtu'
 meshio.write(vdir/filename,meshio.Mesh(np.c_[mesh.p.T,np.zeros(mesh.nvertices)],[('triangle',mesh.t.T)],point_data={'velocity':np.c_[nodal.T,np.zeros(mesh.nvertices)],'pressure':P[k],'projected_vorticity':omega_vertices[k]},cell_data={'vorticity':[omega[k]]}))
 refs.append((float(t[k]),filename))
r=ET.Element('VTKFile',type='Collection',version='0.1',byte_order='LittleEndian');collection=ET.SubElement(r,'Collection')
for time,file in refs:ET.SubElement(collection,'DataSet',timestep=str(time),group='',part='0',file=file)
ET.ElementTree(r).write(vdir/'shedding.pvd',encoding='utf-8',xml_declaration=True)
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader
for _,file in refs:
 reader=vtkXMLUnstructuredGridReader();reader.SetFileName(str(vdir/file));reader.Update();assert reader.GetOutput().GetNumberOfCells()==mesh.nelements
print('GIF, scientific figure and',len(refs),'VTK-verified ParaView frames generated.')
