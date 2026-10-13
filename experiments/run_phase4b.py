from pathlib import Path
import argparse,time,json,csv
import numpy as np
from flowrom.cylinder_transient import TransientCylinder
root=Path(__file__).resolve().parents[1];out=root/'results/phase4b';out.mkdir(exist_ok=True)
a=argparse.ArgumentParser();a.add_argument('--name',default='coarse');a.add_argument('--h',type=float,default=.065);a.add_argument('--hc',type=float,default=.009);a.add_argument('--dt',type=float,default=.005);a.add_argument('--end',type=float,default=30.);a.add_argument('--restart');a.add_argument('--transfer',action='store_true');a.add_argument('--tol',type=float,default=1e-8);a.add_argument('--snapshot-dt',type=float,default=.02);args=a.parse_args()
s=TransientCylinder(args.h,args.hc,dt=args.dt);x=s.initial();previous=None;t0=0.
if args.restart:
 d=np.load(args.restart);x=d['x'];previous=None;t0=float(d['time'])
 same=s.mesh.p.shape==d['points'].shape and np.allclose(s.mesh.p,d['points']) and np.array_equal(s.mesh.t,d['triangles'])
 if not same:
  if not args.transfer:raise RuntimeError('Restart mesh mismatch')
  from skfem import MeshTri,Basis,ElementVector,ElementTriP2,ElementTriP1
  from scipy.sparse import bmat,csr_matrix
  from scipy.sparse.linalg import spsolve
  oldmesh=MeshTri(d['points'],d['triangles']);ob=Basis(oldmesh,ElementVector(ElementTriP2()));op=Basis(oldmesh,ElementTriP1())
  coords=s.ub.doflocs.copy();coords[0]=np.clip(coords[0],1e-10,2.2-1e-10);coords[1]=np.clip(coords[1],1e-10,.41-1e-10)
  radial=coords-np.array([.2,.2])[:,None];radius=np.linalg.norm(radial,axis=0);near=radius<.0500001
  coords[:,near]=np.array([.2,.2])[:,None]+radial[:,near]/radius[near]*.0500001
  values=ob.interpolator(x[:ob.N])(coords);target=np.zeros(s.N)
  for component,indices in enumerate(s.ub.split_indices()):target[indices]=values[component,indices]
  # Mass-metric constrained projection enforces weak continuity on the new mesh.
  K=bmat([[s.M,-s.B.T],[s.B,csr_matrix((s.pb.N,s.pb.N))]],format='csr');rhs=np.r_[s.M@target,np.zeros(s.pb.N)]
  projected=s.prescribed.copy();projected[s.free]=spsolve(K[s.free][:,s.free],rhs[s.free]-K[s.free][:,s.fixed]@projected[s.fixed])
  coords=s.pb.doflocs.copy();coords[0]=np.clip(coords[0],1e-10,2.2-1e-10);coords[1]=np.clip(coords[1],1e-10,.41-1e-10)
  radial=coords-np.array([.2,.2])[:,None];radius=np.linalg.norm(radial,axis=0);near=radius<.0500001
  coords[:,near]=np.array([.2,.2])[:,None]+radial[:,near]/radius[near]*.0500001
  projected[s.N:]=op.interpolator(x[ob.N:])(coords);x=projected
steps=round((args.end-t0)/args.dt);dt=args.dt;start=time.perf_counter();history=[];times=[];snapshots=[];pressure=[];checkpoint_t=25.
from scipy.sparse import save_npz
save_npz(out/f'{args.name}-velocity-mass.npz',s.M);save_npz(out/f'{args.name}-divergence.npz',s.B)
np.savez_compressed(out/f'{args.name}-mesh.npz',points=s.mesh.p,triangles=s.mesh.t,velocity_doflocs=s.ub.doflocs,nodal_dofs=s.ub.nodal_dofs)
import signal
# Recoverable checkpoints allow interruption without discarding a long rollout.
def save_live_checkpoint(signum,frame):
 np.savez_compressed(out/f'{args.name}-live-checkpoint.npz',x=x,time=t0 if 't' not in globals() else t,points=s.mesh.p,triangles=s.mesh.t)
signal.signal(signal.SIGUSR1,save_live_checkpoint)
for k in range(1,steps+1):
 new,d=s.step(x,previous,tol=args.tol);previous,x=x,new;t=t0+k*dt
 history.append(dict(t=t,**d))
 if abs(t-checkpoint_t)<dt/3:np.savez_compressed(out/f'{args.name}-checkpoint25.npz',x=x,time=t,points=s.mesh.p,triangles=s.mesh.t)
 if t>=max(25.,args.end-5) and k%max(1,round(args.snapshot_dt/dt))==0:
  times.append(t);snapshots.append(x[:s.N].copy());pressure.append(x[s.N:].copy())
 if k%max(1,round(.5/dt))==0:
  print(f'{args.name} t={t:.3f} cd={d["cd"]:.5f} cl={d["cl"]:.5f} residual={d["nonlinear_residual"]:.2e} iter={d["iterations"]} elapsed={time.perf_counter()-start:.1f}s',flush=True)
  with (out/f'{args.name}-history.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
if snapshots:np.savez_compressed(out/f'{args.name}-snapshots.npz',t=np.array(times),u=np.array(snapshots),p=np.array(pressure))
np.savez_compressed(out/f'{args.name}-final.npz',x=x,time=args.end,points=s.mesh.p,triangles=s.mesh.t)
(out/f'{args.name}-metadata.json').write_text(json.dumps(dict(dt=dt,start=t0,end=args.end,h=args.h,hcircle=args.hc,nu=.001,umax=1.5,triangles=int(s.mesh.nelements),velocity_dofs=int(s.N),pressure_dofs=int(s.pb.N),elapsed_seconds=time.perf_counter()-start,nonlinear_tolerance=args.tol,transfer=args.transfer,snapshot_dtype='float64',snapshot_dt=args.snapshot_dt,scheme='fully implicit BDF2 with backward-Euler startup',initial='restart checkpoint with BE re-start' if args.restart else 'zero interior velocity, prescribed inlet from t=0'),indent=2))
print('COMPLETE',args.name,flush=True)
