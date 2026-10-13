# ParaView flow viewer

Open results/phase2/paraview/flow.pvd and select Apply. Play the time sequence.
Colour by vorticity and use a fixed range [-2,2] for comparable frames, or colour
by velocity magnitude / pressure. View along the z axis.
For arrows, use Glyph with velocity and reduce glyph density.
For streamlines, use Stream Tracer with a line seed across the plane, if supported
by your ParaView setup. The periodic boundary is not joined in the visualization
mesh; traces stop at the exported domain edge.

Exported VTI point coordinates are the MAC cell centres. Velocity is averaged
from neighbouring faces for display, so it is a second-order visualization
interpolation, not the raw face field. The scalar divergence is computed from
raw face DOFs; do not replace solver diagnostics with ParaView's collocated
gradient filter. Original staggered snapshots are retained in snapshots.npz.

This environment validated VTI files using VTK's native XML reader. No claim of
interactive ParaView GUI inspection is made. Python is sufficient for present
2D quantitative verification; ParaView becomes more useful for later cylinder
geometry, larger data and exploratory streamline views.
