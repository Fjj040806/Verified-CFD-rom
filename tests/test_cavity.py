import unittest
import numpy as np
from flowrom.cavity import Cavity
class TestCavity(unittest.TestCase):
 def test_neumann_inverse(self):
  c=Cavity(20);p=np.random.default_rng(7).normal(size=(20,20));q=c.poisson(c.lap(p))
  np.testing.assert_allclose(q,p-p.mean(),atol=1e-12)
 def test_projection_and_walls(self):
  c=Cavity(16);u,v,p=c.initial();rng=np.random.default_rng(4);u[:,1:-1]=rng.normal(size=(16,15));v[1:-1]=rng.normal(size=(15,16))
  un,vn,_=c.project(u,v,.01);self.assertLess(np.max(np.abs(c.div(un,vn))),1e-12)
  self.assertEqual(np.max(np.abs(un[:,[0,-1]])),0);self.assertEqual(np.max(np.abs(vn[[0,-1]])),0)
 def test_implicit_wall_operator(self):
  c=Cavity(18);rng=np.random.default_rng(5);dt=.01
  for component,shape in [('u',(18,17)),('v',(17,18))]:
   rhs=rng.normal(size=shape);a=c.diffuse(rhs,dt,component)
   # Homogeneous tangential ghost reflection and normal zero faces.
   q=np.pad(a,1);q[0,1:-1]=-a[0] if component=='u' else 0;q[-1,1:-1]=-a[-1] if component=='u' else 0
   q[1:-1,0]=-a[:,0] if component=='v' else 0;q[1:-1,-1]=-a[:,-1] if component=='v' else 0
   lap=(q[2:,1:-1]+q[:-2,1:-1]+q[1:-1,2:]+q[1:-1,:-2]-4*a)/c.h**2
   np.testing.assert_allclose(a-dt*c.nu*lap,rhs,atol=1e-12)
