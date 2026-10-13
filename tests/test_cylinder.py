import unittest
import numpy as np
class TestCylinder(unittest.TestCase):
 def test_mesh_and_steady_contracts(self):
  try:
   from flowrom.cylinder_fem import solve_cylinder
   import gmsh
  except (ImportError,OSError):self.skipTest('Optional cylinder FEM/Gmsh dependencies unavailable')
  r=solve_cylinder(.085,.012,tol=1e-8);s=r['summary']
  self.assertEqual(s['boundary_max_error'],0.)
  self.assertLess(s['weak_divergence'],1e-10)
  self.assertLess(s['free_momentum_residual'],1e-7)
  self.assertLess(abs(s['perimeter']-np.pi*.1),.001)
  self.assertGreater(s['cd'],0.)
  self.assertTrue(np.isfinite(r['u']).all())
