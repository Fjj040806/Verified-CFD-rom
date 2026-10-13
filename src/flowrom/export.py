"""ParaView-readable VTI/PVD; point coordinates are MAC cell centres.
Velocity is interpolated for visualization, not the original face DOFs.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from .navier_stokes import centered_fields


def write_vti(path,g,u,v,p):
    extent=f'0 {g.nx-1} 0 {g.ny-1} 0 0'
    root=ET.Element('VTKFile',type='ImageData',version='0.1',byte_order='LittleEndian')
    data=ET.SubElement(root,'ImageData',WholeExtent=extent,Origin=f'{g.hx/2} {g.hy/2} 0',Spacing=f'{g.hx} {g.hy} 1')
    piece=ET.SubElement(data,'Piece',Extent=extent)
    points=ET.SubElement(piece,'PointData',Scalars='pressure',Vectors='velocity')
    uc,vc,w=centered_fields(g,u,v)
    vel=np.stack([uc,vc,np.zeros_like(uc)],axis=-1)
    for name,a,ncomp in [('pressure',p,1),('velocity',vel,3),('vorticity',w,1),('divergence',g.divergence(u,v),1)]:
        arr=ET.SubElement(points,'DataArray',type='Float64',Name=name,NumberOfComponents=str(ncomp),format='ascii')
        arr.text=' '.join(f'{x:.16g}' for x in a.ravel())
    ET.SubElement(piece,'CellData')
    ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)


def write_pvd(path,frames):
    root=ET.Element('VTKFile',type='Collection',version='0.1',byte_order='LittleEndian')
    collection=ET.SubElement(root,'Collection')
    for t,filename in frames:ET.SubElement(collection,'DataSet',timestep=str(t),group='',part='0',file=filename)
    ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)
