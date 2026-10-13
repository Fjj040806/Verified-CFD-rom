"""First-order periodic incremental pressure-correction Navier-Stokes solver."""
import numpy as np
from .mac import PeriodicMAC


def exact_taylor_green(g,t,nu,background=(0.,0.)):
    a=np.exp(-2*nu*t); U,V=background
    x,y=g.coordinates('u'); u=U+a*np.sin(x-U*t)*np.cos(y-V*t)
    x,y=g.coordinates('v'); v=V-a*np.cos(x-U*t)*np.sin(y-V*t)
    x,y=g.coordinates('p'); p=a*a*(np.cos(2*(x-U*t))+np.cos(2*(y-V*t)))/4
    return u,v,p-p.mean()


class PeriodicFlow:
    def __init__(self,grid,nu,poisson_method='fft',rtol=1e-10):
        if nu<=0: raise ValueError('nu must be positive')
        self.g=grid;self.nu=nu;self.method=poisson_method;self.rtol=rtol
        kx=np.arange(grid.nx);ky=np.arange(grid.ny)
        self.eigen=(-4*np.sin(np.pi*kx/grid.nx)**2/grid.hx**2)[None,:]+(-4*np.sin(np.pi*ky/grid.ny)**2/grid.hy**2)[:,None]

    def convection(self,u,v):
        g=self.g;u,v=g.check(u),g.check(v)
        dx=lambda a:(np.roll(a,-1,axis=1)-np.roll(a,1,axis=1))/(2*g.hx)
        dy=lambda a:(np.roll(a,-1,axis=0)-np.roll(a,1,axis=0))/(2*g.hy)
        # Four-point cross-velocity interpolation at each target face.
        vu=.25*(v+np.roll(v,1,axis=1)+np.roll(v,-1,axis=0)+np.roll(np.roll(v,1,axis=1),-1,axis=0))
        uv=.25*(u+np.roll(u,-1,axis=1)+np.roll(u,1,axis=0)+np.roll(np.roll(u,-1,axis=1),1,axis=0))
        return u*dx(u)+vu*dy(u),uv*dx(v)+v*dy(v)

    def step(self,u,v,p,dt,forcing=None):
        g=self.g;p=g.check(p)
        if dt<=0:raise ValueError('dt must be positive')
        cfl=dt*(np.max(np.abs(u))/g.hx+np.max(np.abs(v))/g.hy)
        if cfl>.5:raise ValueError(f'Advective CFL {cfl:g} exceeds configured safety limit .5')
        nu,nv=self.convection(u,v);gx,gy=g.gradient(p)
        fu,fv=(0.,0.) if forcing is None else forcing
        solve=lambda rhs:np.fft.ifft2(np.fft.fft2(rhs)/(1-dt*self.nu*self.eigen)).real
        us=solve(u+dt*(-nu-gx+fu));vs=solve(v+dt*(-nv-gy+fv))
        un,vn,phi,d=g.project(us,vs,dt,method=self.method,rtol=self.rtol)
        pn=p+phi;pn-=pn.mean()
        d.update(cfl=float(cfl),kinetic_energy=float(.5*np.mean(un*un+vn*vn)))
        return un,vn,pn,d


def centered_fields(g,u,v):
    uc=.5*(u+np.roll(u,-1,axis=1));vc=.5*(v+np.roll(v,-1,axis=0))
    omega=(np.roll(vc,-1,axis=1)-np.roll(vc,1,axis=1))/(2*g.hx)-(np.roll(uc,-1,axis=0)-np.roll(uc,1,axis=0))/(2*g.hy)
    return uc,vc,omega
