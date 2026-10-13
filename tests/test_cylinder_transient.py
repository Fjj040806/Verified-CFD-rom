import unittest
import numpy as np
class TestTransient(unittest.TestCase):
 def test_jacobian_and_vector_assembly(self):
  try:
   from flowrom.cylinder_transient import TransientCylinder,nonlinear,nonlinear_jacobian
   from skfem import asm
   import gmsh
  except (ImportError,OSError):self.skipTest('Optional cylinder dependencies unavailable')
  s=TransientCylinder(.1,.015,umax=.3)
  rng=np.random.default_rng(8);u=rng.normal(size=s.N)*.2;direction=rng.normal(size=s.N);epsilon=1e-6
  expected=asm(nonlinear,s.ub,uh=s.ub.interpolate(u));np.testing.assert_allclose(s.nvector(u),expected,atol=1e-13)
  J=asm(nonlinear_jacobian,s.ub,wind=s.ub.interpolate(u))
  fd=(s.nvector(u+epsilon*direction)-s.nvector(u-epsilon*direction))/(2*epsilon)
  np.testing.assert_allclose(J@direction,fd,atol=1e-10,rtol=1e-7)
  current=s.initial();new,d=s.step(current)
  self.assertLess(d['nonlinear_residual'],1e-8)
  self.assertLess(d['weak_divergence'],1e-10)
  np.testing.assert_allclose(new[s.fixed],s.prescribed[s.fixed],atol=1e-14)
