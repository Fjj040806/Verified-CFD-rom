"""Closed-wall MAC flow; spectral Neumann pressure and implicit wall diffusion."""
import numpy as np
from scipy.fft import dctn,idctn,dst,idst

class Cavity:
 def __init__(self,n,re=100.,lid=1.):
  self.n=n;self.h=1/n;self.nu=1/re;self.lid=lid
  k=np.arange(n);self.ep=-4*np.sin(np.pi*k/(2*n))**2/self.h**2
  self.lp=self.ep[:,None]+self.ep[None,:];self.lp[0,0]=1.
  a=-4*np.sin(np.pi*np.arange(1,n)/(2*n))**2/self.h**2
  b=-4*np.sin(np.pi*np.arange(1,n+1)/(2*n))**2/self.h**2
  self.lu=b[:,None]+a[None,:];self.lv=a[:,None]+b[None,:]
 def initial(self):
  n=self.n;return np.zeros((n,n+1)),np.zeros((n+1,n)),np.zeros((n,n))
 def div(self,u,v):return (np.diff(u,axis=1)+np.diff(v,axis=0))/self.h
 def grad(self,p):
  u,v,_=self.initial();u[:,1:-1]=np.diff(p,axis=1)/self.h;v[1:-1,:]=np.diff(p,axis=0)/self.h
  return u,v
 def lap(self,p):return self.div(*self.grad(p))
 def poisson(self,b):
  scale=max(np.max(np.abs(b)),1.)
  if abs(b.mean())>1e-11*scale:raise ValueError('Nonzero net wall flux')
  b=b-b.mean();z=dctn(b,type=2,norm='ortho')/self.lp;z[0,0]=0
  return idctn(z,type=2,norm='ortho')
 def project(self,u,v,dt):
  b=self.div(u,v)/dt;phi=self.poisson(b);gx,gy=self.grad(phi)
  return u-dt*gx,v-dt*gy,phi
 def extended(self,u,v):
  # Tangential ghost reflections impose boundary values at half-cell distance.
  ue=np.empty((self.n+2,self.n+1));ue[1:-1]=u;ue[0]=-u[0];ue[-1]=2*self.lid-u[-1]
  ve=np.empty((self.n+1,self.n+2));ve[:,1:-1]=v;ve[:,0]=-v[:,0];ve[:,-1]=-v[:,-1]
  return ue,ve
 def convection(self,u,v):
  h=self.h;ue,ve=self.extended(u,v)
  vu=.25*(v[:-1,:-1]+v[1:,:-1]+v[:-1,1:]+v[1:,1:])
  uv=.25*(u[:-1,:-1]+u[:-1,1:]+u[1:,:-1]+u[1:,1:])
  cu=u[:,1:-1]*(u[:,2:]-u[:,:-2])/(2*h)+vu*(ue[2:,1:-1]-ue[:-2,1:-1])/(2*h)
  cv=uv*(ve[1:-1,2:]-ve[1:-1,:-2])/(2*h)+v[1:-1,:]*(v[2:,:]-v[:-2,:])/(2*h)
  return cu,cv
 def diffuse(self,rhs,dt,component):
  if component=='u':
   z=dst(dst(rhs,type=1,axis=1,norm='ortho'),type=2,axis=0,norm='ortho')/(1-dt*self.nu*self.lu)
   return idst(idst(z,type=2,axis=0,norm='ortho'),type=1,axis=1,norm='ortho')
  z=dst(dst(rhs,type=2,axis=1,norm='ortho'),type=1,axis=0,norm='ortho')/(1-dt*self.nu*self.lv)
  return idst(idst(z,type=1,axis=0,norm='ortho'),type=2,axis=1,norm='ortho')
 def step(self,u,v,p,dt):
  if dt*(np.max(np.abs(u))+np.max(np.abs(v)))/self.h>.5:raise ValueError('CFL guard exceeded')
  cu,cv=self.convection(u,v);gx,gy=self.grad(p)
  ru=u[:,1:-1]-dt*(cu+gx[:,1:-1]);rv=v[1:-1,:]-dt*(cv+gy[1:-1,:])
  ru[-1]+=dt*self.nu*2*self.lid/self.h**2
  us,vs,_=self.initial();us[:,1:-1]=self.diffuse(ru,dt,'u');vs[1:-1]=self.diffuse(rv,dt,'v')
  un,vn,phi=self.project(us,vs,dt);pn=p+phi;pn-=pn.mean()
  return un,vn,pn
 def centered(self,u,v):return .5*(u[:,:-1]+u[:,1:]),.5*(v[:-1]+v[1:])
