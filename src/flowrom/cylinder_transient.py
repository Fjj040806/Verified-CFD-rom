"""Fully implicit BDF2 cylinder dynamics, cached-Jacobian nonlinear Newton correction.
Cached Jacobians are refreshed on slowdown; final nonlinear residual is checked.
"""
import numpy as np
from scipy.sparse import bmat,csr_matrix
from scipy.sparse.linalg import splu
from skfem import Basis,ElementVector,ElementTriP2,ElementTriP1,LinearForm,BilinearForm,asm
from skfem.helpers import dot,mul
from skfem.models.poisson import vector_laplace
from skfem.models.general import divergence
from .cylinder_fem import cylinder_mesh,convection

@LinearForm
def nonlinear(v,w):return dot(mul(w.uh.grad,w.uh),v)
@BilinearForm
def nonlinear_jacobian(u,v,w):return dot(mul(u.grad,w.wind)+mul(w.wind.grad,u),v)
@LinearForm
def mass_rhs(v,w):return dot(w.uh,v)

class TransientCylinder:
 def __init__(self,h=.065,hcircle=.009,nu=.001,umax=1.5,dt=.005,mesh=None,tags=None):
  self.mesh,self.tags=cylinder_mesh(h,hcircle) if mesh is None else (mesh,tags)
  self.dt=dt;self.nu=nu;self.umax=umax
  self.ub=Basis(self.mesh,ElementVector(ElementTriP2()),intorder=5);self.pb=Basis(self.mesh,ElementTriP1(),intorder=5)
  self.N=self.ub.N;self.total=self.N+self.pb.N
  self.phi=np.stack([basis[0].value for basis in self.ub.basis])
  self.phigrad=np.stack([basis[0].grad for basis in self.ub.basis])
  self.edofs=self.ub.element_dofs
  self.A=nu*asm(vector_laplace,self.ub)
  from skfem import BilinearForm
  @BilinearForm
  def mass(u,v,w):return dot(u,v)
  self.M=asm(mass,self.ub);self.B=asm(divergence,self.ub,self.pb)
  self.fixed=np.unique(np.concatenate([self.ub.get_dofs(facets=self.tags[k]).all() for k in ('inlet','walls','cylinder')]))
  self.free=np.setdiff1d(np.arange(self.total),self.fixed);self.prescribed=np.zeros(self.total)
  ix=np.intersect1d(self.ub.get_dofs(facets=self.tags['inlet']).all(),self.ub.split_indices()[0]);y=self.ub.doflocs[1,ix]
  self.prescribed[ix]=4*umax*y*(.41-y)/.41**2
  background=np.zeros(self.N);ix=self.ub.split_indices()[0];y=self.ub.doflocs[1,ix]
  background[ix]=4*umax*y*(.41-y)/.41**2
  self.C0=asm(convection,self.ub,wind=self.ub.interpolate(background))
  self.factors={};self.calls=0
  for alpha in (1.,1.5):
   K=bmat([[alpha/dt*self.M+self.A+self.C0,-self.B.T],[self.B,csr_matrix((self.pb.N,self.pb.N))]],format='csc')
   self.factors[alpha]=splu(K[self.free][:,self.free].tocsc())
  cdofs=self.ub.get_dofs(facets=self.tags['cylinder']).all()
  self.force_dofs=[np.intersect1d(cdofs,indices) for indices in self.ub.split_indices()]
  self.pressure_probe=self.pb.probes(np.array([[.15-1e-8,.25+1e-8],[.2,.2]]))
 def initial(self):return self.prescribed.copy()
 def nvector(self,u):
  coefficients=u[self.edofs]
  values=np.einsum('le,lieq->ieq',coefficients,self.phi,optimize=True)
  gradients=np.einsum('le,lijeq->ijeq',coefficients,self.phigrad,optimize=True)
  advected=np.einsum('ijeq,jeq->ieq',gradients,values,optimize=True)
  local=np.einsum('ieq,lieq,eq->le',advected,self.phi,self.ub.dx,optimize=True)
  return np.bincount(self.edofs.ravel(),weights=local.ravel(),minlength=self.N)
 def residual(self,x,alpha,rhs):
  u,p=x[:self.N],x[self.N:]
  momentum=alpha/self.dt*(self.M@u)+self.A@u+self.nvector(u)-self.B.T@p-rhs
  return np.r_[momentum,self.B@u]
 def refresh(self,x,alpha):
  J=asm(nonlinear_jacobian,self.ub,wind=self.ub.interpolate(x[:self.N]))
  K=bmat([[alpha/self.dt*self.M+self.A+J,-self.B.T],[self.B,csr_matrix((self.pb.N,self.pb.N))]],format='csc')
  self.factors[alpha]=splu(K[self.free][:,self.free].tocsc())
 def step(self,current,previous=None,tol=1e-8,maxiter=30,forcing=None,boundary_scale=1.):
  alpha=1. if previous is None else 1.5
  rhs=self.M@(current[:self.N]/self.dt if previous is None else (2*current[:self.N]-.5*previous[:self.N])/self.dt)
  if forcing is not None:rhs=rhs+forcing
  x=current.copy() if previous is None else 2*current-previous;x[self.fixed]=boundary_scale*self.prescribed[self.fixed]
  scale=max(np.linalg.norm(rhs),1.)
  self.calls+=1
  if self.calls<=2 or self.calls%20==0:self.refresh(x,alpha)
  last=np.inf
  for k in range(maxiter):
   r=self.residual(x,alpha,rhs);relative=np.linalg.norm(r[self.free])/scale
   if not np.isfinite(relative):raise RuntimeError('Nonfinite nonlinear state')
   if relative<tol:break
   if k>=2 and (relative>last*.75 or k%6==0):self.refresh(x,alpha)
   delta=self.factors[alpha].solve(-r[self.free]);x[self.free]+=delta
   last=relative
  else:raise RuntimeError(f'Nonlinear solve failed: residual={relative:g}')
  # Includes the inertial term: steady reaction formula would be incorrect here.
  factor=2/((2*self.umax/3)**2*.1)
  cd,cl=[-float(np.sum(r[dofs]))*factor for dofs in self.force_dofs]
  pressures=self.pressure_probe@x[self.N:]
  weak=np.linalg.norm(self.B@x[:self.N])
  return x,dict(cd=cd,cl=cl,dp=float(pressures[0]-pressures[1]),nonlinear_residual=float(relative),iterations=k,weak_divergence=float(weak))
