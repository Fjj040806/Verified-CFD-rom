import unittest
import numpy as np
from flowrom import PeriodicMAC
from flowrom.navier_stokes import PeriodicFlow,exact_taylor_green


class TestFlow(unittest.TestCase):
    def test_uniform_flow_invariant(self):
        g=PeriodicMAC(24,20); s=PeriodicFlow(g,.1)
        u=np.full((20,24),.3);v=np.full((20,24),-.2);p=np.zeros_like(u)
        un,vn,pn,_=s.step(u,v,p,.02)
        np.testing.assert_allclose(un,u,atol=1e-14)
        np.testing.assert_allclose(vn,v,atol=1e-14)
        np.testing.assert_allclose(pn,p,atol=1e-14)

    def test_discrete_taylor_green_decay(self):
        g=PeriodicMAC(32,32);s=PeriodicFlow(g,.1)
        u,v,p=exact_taylor_green(g,0,.1);u0=u.copy();v0=v.copy()
        dt=.005
        for _ in range(20):u,v,p,d=s.step(u,v,p,dt)
        rate=8*np.sin(g.hx/2)**2/g.hx**2
        factor=(1+dt*.1*rate)**(-20)
        np.testing.assert_allclose(u,u0*factor,atol=1e-12)
        np.testing.assert_allclose(v,v0*factor,atol=1e-12)
        self.assertLess(d['divergence_rms'],1e-13)

    def test_convection_accuracy(self):
        errors=[]
        for n in (16,32,64):
            g=PeriodicMAC(n,n);s=PeriodicFlow(g,.1)
            u,v,_=exact_taylor_green(g,0,.1,background=(.3,.2))
            cu,cv=s.convection(u,v)
            x,y=g.coordinates('u')
            eu=.5*np.sin(2*x)+.3*np.cos(x)*np.cos(y)-.2*np.sin(x)*np.sin(y)
            x,y=g.coordinates('v')
            ev=.5*np.sin(2*y)+.3*np.sin(x)*np.sin(y)-.2*np.cos(x)*np.cos(y)
            errors.append(np.sqrt(np.mean((cu-eu)**2+(cv-ev)**2)))
        self.assertTrue(all(np.log2(a/b)>1.9 for a,b in zip(errors,errors[1:])))
