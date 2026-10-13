"""Semidiscrete manufactured solution: isolates nonlinear time-integration error."""
from pathlib import Path
import numpy as np,json
from flowrom.cylinder_fem import solve_cylinder
from flowrom.cylinder_transient import TransientCylinder
root=Path(__file__).resolve().parents[1];out=root/'results/phase4b'
r=solve_cylinder(.1,.015,umax=.3);ubar,pbar=r['u'],r['p'];rows=[]
for dt in (.04,.02,.01,.005):
 s=TransientCylinder(.1,.015,umax=.3,dt=dt,mesh=r['mesh'],tags=r['tags']);base=np.r_[ubar,pbar];current=base.copy();previous=None
 Mu=s.M@ubar;Au=s.A@ubar;Nu=s.nvector(ubar);Bp=s.B.T@pbar
 for k in range(1,round(.4/dt)+1):
  t=k*dt;a=np.exp(-t);f=-a*Mu+a*Au+a*a*Nu-a*Bp
  new,d=s.step(current,previous,tol=1e-10,forcing=f,boundary_scale=a);previous,current=current,new
 error=current[:s.N]-np.exp(-.4)*ubar
 relative=float(np.sqrt(error@(s.M@error)/(ubar@(s.M@ubar))))
 rows.append(dict(dt=dt,relative_mass_velocity_error=relative,residual=d['nonlinear_residual']))
for k,row in enumerate(rows):row['order']=None if k==0 else float(np.log2(rows[k-1]['relative_mass_velocity_error']/row['relative_mass_velocity_error']))
(out/'bdf2-time-verification.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
