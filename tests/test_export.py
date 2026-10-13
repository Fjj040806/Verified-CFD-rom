import tempfile
from pathlib import Path
import unittest
import numpy as np
from flowrom import PeriodicMAC
from flowrom.navier_stokes import exact_taylor_green,centered_fields
from flowrom.export import write_vti


class TestExport(unittest.TestCase):
    def test_vtk_roundtrip(self):
        try:
            from vtkmodules.vtkIOXML import vtkXMLImageDataReader
            from vtkmodules.util.numpy_support import vtk_to_numpy
        except ImportError:self.skipTest('Optional VTK reader unavailable')
        g=PeriodicMAC(12,10);u,v,p=exact_taylor_green(g,.2,.1)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'flow.vti';write_vti(path,g,u,v,p)
            reader=vtkXMLImageDataReader();reader.SetFileName(str(path));reader.Update()
            data=reader.GetOutput()
            self.assertEqual(data.GetNumberOfPoints(),120)
            np.testing.assert_allclose(data.GetOrigin(),(g.hx/2,g.hy/2,0))
            np.testing.assert_allclose(vtk_to_numpy(data.GetPointData().GetArray('pressure')),p.ravel())
            expected=centered_fields(g,u,v)[0]
            np.testing.assert_allclose(vtk_to_numpy(data.GetPointData().GetArray('velocity'))[:,0],expected.ravel())
