from dataclasses import dataclass
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg


@dataclass(frozen=True)
class PeriodicMAC:
    """Unique periodic DOFs: p,u,v all (ny,nx); duplicate boundary faces omitted."""
    nx: int
    ny: int
    lx: float = 2*np.pi
    ly: float = 2*np.pi

    def __post_init__(self):
        if self.nx < 3 or self.ny < 3 or self.lx <= 0 or self.ly <= 0:
            raise ValueError('Positive domain and at least three cells per axis required')

    @property
    def hx(self): return self.lx/self.nx
    @property
    def hy(self): return self.ly/self.ny

    def coordinates(self, location='p'):
        offsets={'p':(.5,.5), 'u':(0,.5), 'v':(.5,0)}
        ox,oy=offsets[location]
        return np.meshgrid((np.arange(self.nx)+ox)*self.hx,
                           (np.arange(self.ny)+oy)*self.hy)

    def check(self, a):
        a=np.asarray(a,dtype=float)
        if a.shape != (self.ny,self.nx) or not np.all(np.isfinite(a)):
            raise ValueError('Expected finite array of shape (ny,nx)')
        return a

    def gradient(self,p):
        p=self.check(p)
        return ((p-np.roll(p,1,axis=1))/self.hx,
                (p-np.roll(p,1,axis=0))/self.hy)

    def divergence(self,u,v):
        u,v=self.check(u),self.check(v)
        return ((np.roll(u,-1,axis=1)-u)/self.hx+
                (np.roll(v,-1,axis=0)-v)/self.hy)

    def laplacian(self,p):
        return self.divergence(*self.gradient(p))

    def poisson(self,b,method='fft',rtol=1e-10,atol=0.,maxiter=None,compatibility_scale=None):
        """Solve L phi=b with zero-mean gauge. Reject non-roundoff incompatibility."""
        b=self.check(b)
        scale=np.max(np.abs(b))
        if compatibility_scale is not None:
            scale=max(scale,compatibility_scale)
        threshold=100*np.finfo(float).eps*max(scale,np.finfo(float).tiny)
        mean=float(b.mean())
        if abs(mean)>threshold:
            raise ValueError(f'Incompatible periodic RHS mean={mean:g}')
        b=b-mean
        iterations=0
        if method=='fft':
            kx=np.arange(self.nx);ky=np.arange(self.ny)
            eigen=(-4*np.sin(np.pi*kx/self.nx)**2/self.hx**2)[None,:]+(-4*np.sin(np.pi*ky/self.ny)**2/self.hy**2)[:,None]
            eigen[0,0]=1.
            transformed=np.fft.fft2(b)/eigen
            transformed[0,0]=0.
            phi=np.fft.ifft2(transformed).real
        elif method=='cg':
            # P(-L)P + constant-mode identity is SPD on the full space.
            def matvec(x):
                a=x.reshape(self.ny,self.nx)
                return (-self.laplacian(a-a.mean())+a.mean()).ravel()
            operator=LinearOperator((b.size,b.size),matvec=matvec,dtype=float)
            def callback(_):
                nonlocal iterations
                iterations+=1
            x,info=cg(operator,-b.ravel(),rtol=rtol,atol=atol,
                      maxiter=maxiter,callback=callback)
            if info!=0: raise RuntimeError(f'CG failed: info={info}')
            phi=x.reshape(b.shape)
        else: raise ValueError('method must be fft or cg')
        phi-=phi.mean()
        residual=b-self.laplacian(phi)
        absolute=float(np.linalg.norm(residual))
        rhs_norm=float(np.linalg.norm(b))
        return phi,{'iterations':iterations,'rhs_mean_before':mean,
                    'absolute_residual':absolute,
                    'relative_residual':absolute/rhs_norm if rhs_norm else 0.}

    def project(self,u,v,dt,**kwargs):
        if dt<=0: raise ValueError('dt must be positive')
        u,v=self.check(u),self.check(v)
        b=self.divergence(u,v)/dt
        # Cancellation scale comes from input face fluxes, not the tiny remainder.
        scale=(2*np.max(np.abs(u))/self.hx+2*np.max(np.abs(v))/self.hy)/dt
        phi,diagnostics=self.poisson(b,compatibility_scale=scale,**kwargs)
        gx,gy=self.gradient(phi)
        un,vn=u-dt*gx,v-dt*gy
        residual=b-self.laplacian(phi)
        div=self.divergence(un,vn)
        diagnostics.update(divergence_rms=float(np.sqrt(np.mean(div**2))),
            identity_max_error=float(np.max(np.abs(div-dt*residual))),
            energy_before=float(.5*np.mean(u*u+v*v)),
            energy_after=float(.5*np.mean(un*un+vn*vn)))
        return un,vn,phi,diagnostics
