"""DFG Re20 cylinder: fitted polygonal geometry, P2/P1 FEM and Picard iteration.
Third-party assembly: scikit-fem; mesh: Gmsh. Not a from-scratch FEM library.
"""
from pathlib import Path
import numpy as np
from scipy.sparse import bmat,csr_matrix
from scipy.sparse.linalg import spsolve
from skfem import MeshTri,Basis,ElementVector,ElementTriP2,ElementTriP1,BilinearForm,FacetBasis,Functional,asm
from skfem.helpers import dot,mul,grad
from skfem.models.poisson import vector_laplace
from skfem.models.general import divergence


def cylinder_mesh(h=.045,hcircle=.006):
 import gmsh
 gmsh.initialize();gmsh.option.setNumber('General.Terminal',0)
 try:
  gmsh.model.add('dfg-cylinder');occ=gmsh.model.occ
  rect=occ.addRectangle(0,0,0,2.2,.41);disk=occ.addDisk(.2,.2,0,.05,.05)
  surface,_=occ.cut([(2,rect)],[(2,disk)]);occ.synchronize()
  gmsh.option.setNumber('Mesh.MeshSizeMin',hcircle);gmsh.option.setNumber('Mesh.MeshSizeMax',h)
  # Distance-based refinement around the actual CAD circle.
  curves=gmsh.model.getBoundary(surface,oriented=False)
  circle=[]
  for dim,tag in curves:
   box=gmsh.model.getBoundingBox(dim,tag)
   if box[0]>.1 and box[3]<.3:circle.append(tag)
  dist=gmsh.model.mesh.field.add('Distance');gmsh.model.mesh.field.setNumbers(dist,'CurvesList',circle)
  gmsh.model.mesh.field.setNumber(dist,'Sampling',100)
  threshold=gmsh.model.mesh.field.add('Threshold');gmsh.model.mesh.field.setNumber(threshold,'InField',dist)
  for key,value in [('SizeMin',hcircle),('SizeMax',h),('DistMin',.04),('DistMax',.35)]:gmsh.model.mesh.field.setNumber(threshold,key,value)
  gmsh.model.mesh.field.setAsBackgroundMesh(threshold);gmsh.model.mesh.generate(2)
  tags,coords,_=gmsh.model.mesh.getNodes();p=np.asarray(coords).reshape(-1,3)[:,:2]
  _,tri=gmsh.model.mesh.getElementsByType(2);tri=np.asarray(tri).reshape(-1,3)
  lookup={int(tag):k for k,tag in enumerate(tags)}
  t=np.array([[lookup[int(x)] for x in row] for row in tri])
  mesh=MeshTri(p.T,t.T)
 finally:gmsh.finalize()
 boundary=mesh.boundary_facets();mid=mesh.p[:,mesh.facets[:,boundary]].mean(axis=1)
 radial=np.linalg.norm(mesh.p[:,mesh.facets[:,boundary]]-np.array([.2,.2])[:,None,None],axis=0)
 cylinder=boundary[np.all(np.isclose(radial,.05,atol=1e-8),axis=0)]
 inlet=boundary[np.isclose(mid[0],0)];outlet=boundary[np.isclose(mid[0],2.2)]
 walls=boundary[np.isclose(mid[1],0)|np.isclose(mid[1],.41)]
 if len(inlet)==0 or len(outlet)==0 or len(cylinder)<12:raise RuntimeError('Boundary tagging failed')
 return mesh,dict(inlet=inlet,outlet=outlet,walls=walls,cylinder=cylinder)


@BilinearForm
def convection(u,v,w):return dot(mul(grad(u),w.wind),v)


def solve_cylinder(h=.045,hcircle=.006,nu=.001,umax=.3,tol=1e-9,maxiter=100):
 mesh,tags=cylinder_mesh(h,hcircle)
 ub=Basis(mesh,ElementVector(ElementTriP2()),intorder=5);pb=Basis(mesh,ElementTriP1(),intorder=5)
 A=nu*asm(vector_laplace,ub);B=asm(divergence,ub,pb)
 fixed=np.unique(np.concatenate([ub.get_dofs(facets=tags[k]).all() for k in ('inlet','walls','cylinder')]))
 prescribed=np.zeros(ub.N+pb.N);xin=ub.get_dofs(facets=tags['inlet']).all();ix=np.intersect1d(xin,ub.split_indices()[0]);y=ub.doflocs[1,ix]
 prescribed[ix]=4*umax*y*(.41-y)/.41**2
 free=np.setdiff1d(np.arange(ub.N+pb.N),fixed)
 x=prescribed.copy();history=[]
 for k in range(maxiter):
  C=asm(convection,ub,wind=ub.interpolate(x[:ub.N]));K=bmat([[A+C,-B.T],[B,csr_matrix((pb.N,pb.N))]],format='csr')
  new=prescribed.copy();new[free]=spsolve(K[free][:,free],-K[free][:,fixed]@prescribed[fixed])
  # Relax the Picard iterate, keeping essential boundary values exact.
  xnew=.7*new+.3*x;xnew[fixed]=prescribed[fixed]
  delta=np.linalg.norm(xnew-x)/max(np.linalg.norm(xnew),1e-15);x=xnew
  history.append(float(delta))
  if delta<tol:break
 else:raise RuntimeError('Picard iteration did not converge')
 C=asm(convection,ub,wind=ub.interpolate(x[:ub.N]));K=bmat([[A+C,-B.T],[B,csr_matrix((pb.N,pb.N))]],format='csr')
 residual=K@x;reaction=[];cdofs=ub.get_dofs(facets=tags['cylinder']).all()
 for indices in ub.split_indices():reaction.append(-float(np.sum(residual[np.intersect1d(cdofs,indices)])))
 mean=2*umax/3;factor=2/(mean**2*.1)
 velocity=x[:ub.N];pressure=x[ub.N:]
 # Query pressure from the fluid side immediately outside cardinal CAD points.
 q=np.array([[.15-1e-8,.25+1e-8],[.2,.2]])
 dp=float(np.diff(pb.probes(q)@pressure)[0]*-1)
 divfield=ub.interpolate(velocity).grad
 divq=divfield[0,0]+divfield[1,1]
 l2div=float(np.sqrt(np.sum(divq**2*ub.dx)))
 weak=float(np.linalg.norm(B@velocity))
 fb=FacetBasis(mesh,ElementVector(ElementTriP2()),facets=tags['cylinder'],intorder=5)
 fp=FacetBasis(mesh,ElementTriP1(),facets=tags['cylinder'],intorder=5)
 uf,pf=fb.interpolate(velocity),fp.interpolate(pressure)
 surface=[]
 for axis in (0,1):
  @Functional
  def force(w):return -(nu*mul(w.uh.grad,w.n)[axis]-w.ph*w.n[axis])
  surface.append(float(asm(force,fb,uh=uf,ph=pf))*factor)
 fluxes={}
 @Functional
 def flux(w):return dot(w.uh,w.n)
 for key in ('inlet','outlet','walls','cylinder'):
  boundary_basis=FacetBasis(mesh,ElementVector(ElementTriP2()),facets=tags[key],intorder=5)
  fluxes[key]=float(asm(flux,boundary_basis,uh=boundary_basis.interpolate(velocity)))
 coords=mesh.p;tri=mesh.t;facets=mesh.facets[:,tags['cylinder']]
 perimeter=float(np.sum(np.linalg.norm(coords[:,facets[0]]-coords[:,facets[1]],axis=0)))
 return dict(mesh=mesh,tags=tags,ub=ub,pb=pb,u=velocity,p=pressure,history=history,
  summary=dict(h=h,hcircle=hcircle,triangles=int(mesh.nelements),velocity_dofs=int(ub.N),pressure_dofs=int(pb.N),
  surface_cd=surface[0],surface_cl=surface[1],boundary_fluxes=fluxes,cylinder_facets=len(tags['cylinder']),perimeter=perimeter,cd=reaction[0]*factor,cl=reaction[1]*factor,pressure_difference=dp,
  free_momentum_residual=float(np.linalg.norm(residual[free])/max(np.linalg.norm(K[free][:,fixed]@prescribed[fixed]),1e-15)),
  weak_divergence=weak,divergence_l2=l2div,boundary_max_error=float(np.max(np.abs(x[fixed]-prescribed[fixed]))),picard_iterations=len(history)))
