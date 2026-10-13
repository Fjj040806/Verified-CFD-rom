from pathlib import Path
import json,time
import numpy as np
import meshio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from flowrom.cylinder_fem import solve_cylinder
root=Path(__file__).resolve().parents[1];out=root/'results/phase4';out.mkdir(exist_ok=True)
ref=dict(cd=5.57953523384,cl=.010618948146,pressure_difference=.11752016697)
results=[]
for name,h,hc in [('coarse',.065,.009),('medium',.045,.006),('fine',.032,.004)]:
 start=time.perf_counter();r=solve_cylinder(h,hc);s=r['summary'];s['seconds']=time.perf_counter()-start;s['name']=name
 for key,value in ref.items():s[key+'_relative_error']=abs(s[key]-value)/abs(value)
 results.append(s);print(json.dumps(s),flush=True)
 mesh=r['mesh'];uv=r['u'][r['ub'].nodal_dofs];p=r['p'];tri=Triangulation(mesh.p[0],mesh.p[1],mesh.t.T)
 np.savez_compressed(out/f'{name}.npz',points=mesh.p,triangles=mesh.t,u=r['u'],p=p,vertex_velocity=uv,history=r['history'],velocity_doflocs=r['ub'].doflocs)
 meshio.write(out/f'{name}.vtu',meshio.Mesh(np.c_[mesh.p.T,np.zeros(mesh.nvertices)], [('triangle',mesh.t.T)],point_data={'velocity':np.c_[uv.T,np.zeros(mesh.nvertices)],'pressure':p,'speed':np.linalg.norm(uv,axis=0)}))
 if name=='fine':
  fig,axs=plt.subplots(3,1,figsize=(13,9),constrained_layout=True);fig.set_facecolor('#FBFCFB')
  for ax in axs:ax.set_aspect('equal');ax.set(xlim=(0,2.2),ylim=(0,.41),xlabel='x',ylabel='y')
  axs[0].triplot(tri,color='#718B95',lw=.22);axs[0].set_title('Fitted cylinder mesh · P2 velocity / P1 pressure')
  im=axs[1].tripcolor(tri,np.linalg.norm(uv,axis=0),shading='gouraud',cmap='viridis');axs[1].set_title('Computed steady speed · DFG Re = 20');fig.colorbar(im,ax=axs[1],label='Speed')
  im=axs[2].tripcolor(tri,p,shading='gouraud',cmap='RdBu_r');axs[2].set_title('Computed pressure · natural traction outlet');fig.colorbar(im,ax=axs[2],label='Pressure (density = 1)')
  for ext in ('png','svg','pdf'):fig.savefig(root/f'figures/phase4-cylinder.{ext}',dpi=180)
  plt.close(fig)
(out/'summary.json').write_text(json.dumps({'reference':ref,'results':results},indent=2))
fig,axs=plt.subplots(1,3,figsize=(12,3.6),constrained_layout=True)
for ax,key in zip(axs,ref):
 ax.plot([r['cylinder_facets'] for r in results],[r[key] for r in results],'o-',color='#318B88',label='computed');ax.axhline(ref[key],color='#CC8868',ls='--',label='DFG reference');ax.set(xlabel='Cylinder boundary facets',ylabel=key,title=key);ax.legend(fontsize=8)
for ext in ('png','svg','pdf'):fig.savefig(root/f'figures/phase4-benchmark.{ext}',dpi=180)
print('Phase 4 steady benchmark calculations complete.')
