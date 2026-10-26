"""Audit saved-mesh ROM operators and exact probes; no neural training.
Run from the repository root after installing requirements-operators.txt.
"""
from pathlib import Path
import argparse,json,sys,time,hashlib,platform
import numpy as np
from scipy.sparse import load_npz,save_npz,coo_matrix
from scipy.sparse.linalg import norm as sparse_norm
from scipy.integrate import solve_ivp
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[2]);args=parser.parse_args();root=args.project_root.resolve();here=Path(__file__).resolve().parent;out=here/'phase1-results';out.mkdir(exist_ok=True);start=time.perf_counter()
sys.path.insert(0,str(root/'src'))
from skfem import MeshTri,Basis,ElementTriP2,asm
import skfem,scipy
from flowrom.rom import WeightedPOD
from flowrom.cylinder_transient import TransientCylinder,nonlinear_jacobian
config=json.loads((here/'pilot-config.json').read_text());manifest=json.loads((root/'RELEASE-MANIFEST.json').read_text());hashes={f['path']:f['sha256'] for f in manifest['files']}
inputs=['data/demo/re100-settled.npz','data/demo/velocity-mass.npz','results/phase4b/coarse-half-mesh.npz','results/phase4b/coarse-half-divergence.npz','src/flowrom/rom.py','src/flowrom/cylinder_fem.py','src/flowrom/cylinder_transient.py']
for name in inputs:assert hashlib.sha256((root/name).read_bytes()).hexdigest()==hashes[name],name
raw=np.load(root/inputs[0]);sel=(raw['t']>=27-1e-9)&(raw['t']<=28.79+1e-9);t=raw['t'][sel];U=raw['u'][sel].T
assert U.shape[1]==180 and t[-1]<28.8
M=load_npz(root/inputs[1]);saved=np.load(root/inputs[2]);B=load_npz(root/inputs[3]);mesh=MeshTri(saved['points'],saved['triangles'])
bd=mesh.boundary_facets();mid=mesh.p[:,mesh.facets[:,bd]].mean(axis=1)
tags={'inlet':bd[np.isclose(mid[0],0,atol=1e-12)],'outlet':bd[np.isclose(mid[0],2.2,atol=1e-12)],'walls':bd[np.isclose(mid[1],0,atol=1e-12)|np.isclose(mid[1],.41,atol=1e-12)]}
tags['cylinder']=np.setdiff1d(bd,np.concatenate(list(tags.values())))
assert len(np.unique(np.concatenate(list(tags.values()))))==len(bd)
rad=np.linalg.norm(mesh.p[:,mesh.facets[:,tags['cylinder']]]-np.array([.2,.2])[:,None,None],axis=0);assert np.max(abs(rad-.05))<1e-10
solver=TransientCylinder(nu=.001,umax=1.5,dt=.005,mesh=mesh,tags=tags)
coord_err=float(np.max(abs(solver.ub.doflocs-saved['velocity_doflocs'])));mass_err=float(sparse_norm(solver.M-M)/sparse_norm(M));b_err=float(sparse_norm(solver.B-B)/sparse_norm(B))
assert coord_err==0;assert mass_err<1e-12;assert b_err<1e-12
pod=WeightedPOD(U,M);Phi=pod.modes[:,:20];mean=pod.mean;Mr=Phi.T@(M@Phi);scales=pod.coefficients(U,20).std(axis=1)
assert np.linalg.eigvalsh(Mr).min()>.999
boundary_error=float(np.max(abs(mean[solver.fixed]-solver.prescribed[solver.fixed])));trace_rel=float(np.linalg.norm(Phi[solver.fixed])/np.linalg.norm(Phi));weak_scaled=float(np.linalg.norm(B@Phi)/(sparse_norm(B)*np.linalg.norm(Phi)))
assert boundary_error<1e-12;assert trace_rel<1e-10;assert weak_scaled<1e-10
scalar=Basis(mesh,ElementTriP2(),intorder=5);split=solver.ub.split_indices();assert np.array_equal(solver.ub.doflocs[:,split[0]],scalar.doflocs)
def probe(points):
 points=np.asarray(points);P=scalar.probes(points.T).tocoo();H=coo_matrix((np.r_[P.data,P.data],(np.r_[2*P.row,2*P.row+1],np.r_[split[0][P.col],split[1][P.col]])),shape=(2*len(points),solver.N)).tocsr()
 values=np.zeros(solver.N);x,y=solver.ub.doflocs[:,split[0]];values[split[0]]=1+x+y;x,y=solver.ub.doflocs[:,split[1]];values[split[1]]=2-x+2*y*y
 truth=np.empty(2*len(points));truth[0::2]=1+points[:,0]+points[:,1];truth[1::2]=2-points[:,0]+2*points[:,1]**2
 error=float(np.max(abs(H@values-truth)));assert error<1e-12*(1+np.max(abs(truth)));return H,error
points=config['sensors'];hidden=[[x,y] for x in [.5,.85,1.2,1.7] for y in [.08,.16,.25,.33]];assert set(map(tuple,points)).isdisjoint(set(map(tuple,hidden)))
H,probe_error=probe(points);Hv,hidden_probe_error=probe(hidden);save_npz(out/'sensor-probes.npz',H);save_npz(out/'hidden-probes.npz',Hv)
A=solver.A;c=-Phi.T@(A@mean+solver.nvector(mean));J=asm(nonlinear_jacobian,solver.ub,wind=solver.ub.interpolate(mean));L=-Phi.T@((A+J)@Phi);Q=np.zeros((20,20,20));diag=[solver.nvector(Phi[:,j]) for j in range(20)]
for j in range(20):
 Q[:,j,j]=-Phi.T@diag[j]
 for k in range(j+1,20):
  q=-.5*Phi.T@(solver.nvector(Phi[:,j]+Phi[:,k])-diag[j]-diag[k]);Q[:,j,k]=q;Q[:,k,j]=q
np.savez_compressed(out/'reduced-operators.npz',mean=mean,modes=Phi,mass=Mr,constant=c,linear=L,quadratic_symmetric=Q,offline_times=t,offline_scales=scales,viscosity=.001,sensors=np.array(points),hidden_sensors=np.array(hidden))
full_projected=-Phi.T@(A@U+np.column_stack([solver.nvector(U[:,j]) for j in range(U.shape[1])]))
rows=[];rng=np.random.default_rng(0);curves=[]
for rank in [6,10,20]:
 P=Phi[:,:rank];G=Mr[:rank,:rank];Gi=np.linalg.inv(G);cr=c[:rank];lr=L[:rank,:rank];qr=Q[:rank,:rank,:rank]
 def F(a):return cr+lr@a+np.einsum('ijk,j,k->i',qr,a,a)
 def rhs(time,a):return Gi@F(a)
 errors=[]
 for j in range(10):
  a=rng.normal(size=rank)*scales[:rank];u=mean+P@a;direct=-P.T@(A@u+solver.nvector(u));errors.append(float(np.linalg.norm(F(a)-direct)/(1+np.linalg.norm(direct))))
 assert max(errors)<1e-10
 a=rng.normal(size=rank)*scales[:rank];v=rng.normal(size=rank);eps=1e-6
 analytic=(lr+2*np.einsum('ijk,k->ij',qr,a))@v;finite=(F(a+eps*v)-F(a-eps*v))/(2*eps);jac_error=float(np.linalg.norm(analytic-finite)/(1+np.linalg.norm(analytic)));assert jac_error<1e-7
 coeff=Gi@(P.T@(M@(U-mean[:,None])));projected=mean[:,None]+P@coeff
 direct_resolved=-P.T@(A@projected+np.column_stack([solver.nvector(projected[:,j]) for j in range(projected.shape[1])]))
 closure=Gi@(full_projected[:rank]-direct_resolved);full_rate=Gi@full_projected[:rank];closure_ratio=float(np.linalg.norm(closure)/np.linalg.norm(full_rate))
 # Common interior points: compare temporal stencils without labeling this pure time error.
 dt=float(np.median(np.diff(t)));n=len(t)
 d3=(coeff[:,4:n-2]-coeff[:,2:n-4])/(2*dt)
 d5=(coeff[:,1:n-5]-8*coeff[:,2:n-4]+8*coeff[:,4:n-2]-coeff[:,5:n-1])/(12*dt)
 d7=(-coeff[:,0:n-6]+9*coeff[:,1:n-5]-45*coeff[:,2:n-4]+45*coeff[:,4:n-2]-9*coeff[:,5:n-1]+coeff[:,6:n])/(60*dt)
 ref_rate=full_rate[:,3:n-3];rate_norm=np.linalg.norm(ref_rate)
 temporal=dict(centered3_mismatch_over_rate=float(np.linalg.norm(d3-ref_rate)/rate_norm),centered5_mismatch_over_rate=float(np.linalg.norm(d5-ref_rate)/rate_norm),centered7_mismatch_over_rate=float(np.linalg.norm(d7-ref_rate)/rate_norm),fd5_vs_fd7_over_rate=float(np.linalg.norm(d5-d7)/rate_norm),scope='Combined saved-trajectory/derivative/solver consistency diagnostic, not a pure time-error estimate')
 sv=np.linalg.svd(H@P,compute_uv=False);nrank=int(np.sum(sv>sv[0]*1e-10));condition=float(sv[0]/sv[-1]) if nrank==rank else None
 # An offline, full-state-initialized diagnostic, not a sparse-observation result.
 coarse=solve_ivp(rhs,(float(t[0]),float(t[-1])),coeff[:,0],t_eval=t,rtol=1e-8,atol=1e-10)
 fine=solve_ivp(rhs,(float(t[0]),float(t[-1])),coeff[:,0],t_eval=t,rtol=1e-10,atol=1e-12)
 ode=dict(status='failed',message=coarse.message+'; '+fine.message)
 if coarse.success and fine.success and coarse.y.shape==coeff.shape and fine.y.shape==coeff.shape:
  diff=coarse.y-fine.y;ode_change=float(np.sqrt(np.sum(diff*(G@diff))/np.sum(fine.y*(G@fine.y))));pred=mean[:,None]+P@fine.y
  ode=dict(status='completed',relative_mass_trajectory_tolerance_change=ode_change,offline_total_error=pod.errors(pred,U)['total_relative'],offline_fluctuation_error=pod.errors(pred,U)['fluctuation_relative'],evaluation_role='offline full-state-initialized diagnostic only')
  assert ode_change<1e-5
  pointwise=np.sqrt(np.sum((pred-U)*(M@(pred-U)),axis=0));curves.append((rank,pointwise))
 rows.append(dict(rank=rank,tensor_max_scaled_error=max(errors),tensor_jacobian_scaled_error=jac_error,sensor_map_rank=nrank,sensor_nullity=rank-nrank,sensor_condition_full_column=condition,sensor_singular_values=sv.tolist(),offline_closure_over_projected_rate=closure_ratio,temporal_consistency=temporal,weak_mode_absolute_norm=float(np.linalg.norm(B@P)),galerkin_diagnostic=ode))
summary=dict(stage='Phase 1A: saved-mesh physics and sensor audit',no_pinn_trained=True,base_data_commit=config['baseline_commit'],offline_interval=[float(t[0]),float(t[-1])],offline_frames=len(t),complete_online_or_future_fields_used=False,velocity_dofs=int(solver.N),mesh_coordinate_max_error=coord_err,mass_relative_frobenius_error=mass_err,divergence_matrix_relative_error=b_err,mean_boundary_max_error=boundary_error,mode_boundary_relative_error=trace_rel,mode_weak_divergence_matrix_scaled=weak_scaled,sensor_polynomial_max_error=probe_error,hidden_polynomial_max_error=hidden_probe_error,reduced_mass_max_identity_error=float(np.max(abs(Mr-np.eye(20)))),ranks=rows,autodiff_gate='pending: no neural-library AD claimed; tensor derivative checked analytically versus finite differences',runtime_seconds=time.perf_counter()-start,software=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,skfem=skfem.__version__))
(out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout='constrained');axs=axes.flat;teal='#318B88';clay='#CC8868'
palette={6:teal,10:clay,20:'#788393'}
for row in rows:axs[0].semilogy(np.arange(1,len(row['sensor_singular_values'])+1),row['sensor_singular_values'],'o-',color=palette[row['rank']],label=f"r={row['rank']}")
axs[0].set(xlabel='Singular-value index',ylabel='Singular value of H Phi',title='A · Eight two-component sensors');axs[0].legend(fontsize=8)
axs[1].bar([str(row['rank']) for row in rows],[row['offline_closure_over_projected_rate'] for row in rows],color=teal);axs[1].set(xlabel='POD rank',ylabel='Spatial closure / projected rate',title='B · Offline operator discrepancy')
axs[1].set_yscale('log')
axs[2].semilogy([str(row['rank']) for row in rows],[row['temporal_consistency']['centered7_mismatch_over_rate'] for row in rows],'o-',color=clay,label='7-point derivative vs full projected rate')
axs[2].semilogy([str(row['rank']) for row in rows],[row['temporal_consistency']['fd5_vs_fd7_over_rate'] for row in rows],'s:',color=teal,label='5- vs 7-point stencil sensitivity')
axs[2].set(xlabel='POD rank',ylabel='Relative rate discrepancy',title='C · Temporal consistency (not pure time error)');axs[2].legend(fontsize=7)
for rank,error in curves:axs[3].semilogy(t-t[0],np.maximum(error,1e-15),color=palette[rank],label=f'r={rank}')
axs[3].set(xlabel='Time since offline initial frame',ylabel='Instantaneous mass velocity error',title='D · Full-state-initialized Galerkin');axs[3].legend(fontsize=8)
for ax in axes.flat:ax.grid(alpha=.15)
fig.suptitle('Phase 1A · Offline operator diagnostics, not a trained PINN or sparse-reconstruction result',fontsize=11)
for ext in ['png','svg','pdf']:fig.savefig(out/f'operator-audit.{ext}',dpi=180)
print(json.dumps(summary,indent=2,allow_nan=False))
