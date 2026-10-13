from pathlib import Path
import numpy as np,json
from flowrom.cylinder_transient import TransientCylinder
root=Path(__file__).resolve().parents[1];out=root/'results/phase4b';checkpoint=np.load(out/'coarse-checkpoint25.npz')
a=TransientCylinder(dt=.005);b=TransientCylinder(dt=.005);xa=checkpoint['x'].copy();xb=xa.copy();pa=pb=None;records=[]
for k in range(1,67):
 na,da=a.step(xa,pa,tol=1e-6);nb,db=b.step(xb,pb,tol=1e-8);pa,xa=xa,na;pb,xb=xb,nb
 diff=xa[:a.N]-xb[:b.N];scale=xb[:b.N]
 error=float(np.sqrt(diff@(a.M@diff)/(scale@(a.M@scale))))
 records.append(dict(t=25+k*.005,relative_mass_velocity_difference=error,cd_difference=abs(da['cd']-db['cd']),cl_difference=abs(da['cl']-db['cl']),loose_residual=da['nonlinear_residual'],tight_residual=db['nonlinear_residual']))
summary={key:max(r[key] for r in records) for key in records[0] if key!='t'}
(out/'nonlinear-tolerance-audit.json').write_text(json.dumps(dict(dt=.005,start=25,end=25.33,summary=summary,records=records),indent=2));print(json.dumps(summary,indent=2))
