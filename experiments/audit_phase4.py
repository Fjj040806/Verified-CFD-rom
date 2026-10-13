from pathlib import Path
import json
import numpy as np
from skfem import MeshTri,Basis,FacetBasis,Functional,ElementVector,ElementTriP2,ElementTriP1,asm
from skfem.helpers import dot,mul
root=Path(__file__).resolve().parents[1];out=root/'results/phase4';summary=json.loads((out/'summary.json').read_text())
for row in summary['results']:
 d=np.load(out/f"{row['name']}.npz");mesh=MeshTri(d['points'],d['triangles']);u,p=d['u'],d['p'];boundary=mesh.boundary_facets();mid=mesh.p[:,mesh.facets[:,boundary]].mean(axis=1)
 radial=np.linalg.norm(mesh.p[:,mesh.facets[:,boundary]]-np.array([.2,.2])[:,None,None],axis=0)
 tags=dict(cylinder=boundary[np.all(np.isclose(radial,.05,atol=1e-8),axis=0)],inlet=boundary[np.isclose(mid[0],0)],outlet=boundary[np.isclose(mid[0],2.2)],walls=boundary[np.isclose(mid[1],0)|np.isclose(mid[1],.41)])
 fb=FacetBasis(mesh,ElementVector(ElementTriP2()),facets=tags['cylinder'],intorder=5);fp=FacetBasis(mesh,ElementTriP1(),facets=tags['cylinder'],intorder=5)
 uf,pf=fb.interpolate(u),fp.interpolate(p)
 for axis,key in enumerate(['surface_cd','surface_cl']):
  @Functional
  def force(w):return -(.001*mul(w.uh.grad,w.n)[axis]-w.ph*w.n[axis])
  row[key]=float(asm(force,fb,uh=uf,ph=pf))*500
 row['boundary_fluxes']={}
 @Functional
 def flux(w):return dot(w.uh,w.n)
 for key,facets in tags.items():
  b=FacetBasis(mesh,ElementVector(ElementTriP2()),facets=facets,intorder=5)
  row['boundary_fluxes'][key]=float(asm(flux,b,uh=b.interpolate(u)))
 # Roundtrip VTU: vertex-only visual interpolation, not complete P2 DOFs.
 from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader
 reader=vtkXMLUnstructuredGridReader();reader.SetFileName(str(out/f"{row['name']}.vtu"));reader.Update()
 assert reader.GetOutput().GetNumberOfCells()==mesh.nelements
 print(row['name'],row['surface_cd'],row['surface_cl'],row['boundary_fluxes'])
(out/'summary.json').write_text(json.dumps(summary,indent=2))
