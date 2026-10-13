import unittest
import numpy as np
from flowrom import PeriodicMAC


class TestMAC(unittest.TestCase):
    def test_periodic_adjoint_and_rectangular_poisson(self):
        g=PeriodicMAC(31,24,3.,5.)
        rng=np.random.default_rng(7)
        p,u,v=[rng.normal(size=(24,31)) for _ in range(3)]
        gx,gy=g.gradient(p)
        self.assertAlmostEqual(float(np.sum(p*g.divergence(u,v))+np.sum(gx*u+gy*v)),0.,places=10)
        b=g.laplacian(p)
        phi,_=g.poisson(b)
        np.testing.assert_allclose(phi,p-p.mean(),atol=1e-12)

    def test_projection_contract(self):
        g=PeriodicMAC(32,24)
        rng=np.random.default_rng(3)
        u,v=[rng.normal(size=(24,32)) for _ in range(2)]
        un,vn,_,d=g.project(u,v,.03,method='cg',rtol=1e-9)
        self.assertLess(d['identity_max_error'],1e-12)
        self.assertLess(d['divergence_rms'],1e-7)
        self.assertLessEqual(d['energy_after'],d['energy_before']+1e-13)
        un2,vn2,_,_=g.project(un,vn,.03)
        np.testing.assert_allclose(un2,un,atol=1e-7)
        np.testing.assert_allclose(vn2,vn,atol=1e-7)

    def test_compatibility_and_zero_rhs(self):
        g=PeriodicMAC(8,8)
        with self.assertRaises(ValueError): g.poisson(np.ones((8,8)))
        for method in ('fft','cg'):
            phi,d=g.poisson(np.zeros((8,8)),method=method)
            self.assertEqual(np.max(np.abs(phi)),0.)
            self.assertEqual(d['absolute_residual'],0.)

    def test_second_order_poisson(self):
        errors=[]
        for n in (16,32,64):
            g=PeriodicMAC(n,n);x,y=g.coordinates()
            exact=np.sin(x)*np.cos(2*y)
            phi,_=g.poisson(-5*exact)
            errors.append(np.sqrt(np.mean((phi-exact)**2)))
        self.assertTrue(all(np.log2(a/b)>1.95 for a,b in zip(errors,errors[1:])))


if __name__=='__main__': unittest.main()
