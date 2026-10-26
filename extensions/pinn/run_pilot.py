"""Frozen nine-fit sparse-observation PINN pilot. All outputs use final iterates."""
import argparse, json, time, hashlib
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);args=p.parse_args()
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
 out=HERE/'phase2-results';out.mkdir(exist_ok=True)
 op=np.load(HERE/'phase1-results/reduced-operators.npz');r=10
 phi=op['modes'][:,:r];mean=op['mean'];G=op['mass'][:r,:r];Gi=np.linalg.inv(G)
 c=op['constant'][:r];L=op['linear'][:r,:r];Q=op['quadratic_symmetric'][:r,:r,:r]
 H=load_npz(HERE/'phase1-results/sensor-probes.npz');HH=load_npz(HERE/'phase1-results/hidden-probes.npz')
 C=H@phi;ym=H@mean
 data=np.load(args.baseline/'data/demo/re100-settled.npz');M=load_npz(args.baseline/'data/demo/velocity-mass.npz')
 # Only offline full fields and online sensor observations enter fitting.
 off=data['u'][:180];aoff=np.linalg.solve(G,phi.T@(M@(off-mean).T)).T
 D=np.maximum(aoff.std(axis=0),.001*aoff.std(axis=0)[0]);T=.59
 Y=(H@data['u'][180:240].T).T
 tau=torch.linspace(0,1,60)[:,None];coll=torch.linspace(0,1,128)[:,None]
 ct,lt,qt,git,dt,cd,yt,ymt=[torch.tensor(x) for x in (c,L,Q,Gi,D,C*D,Y,ym)]
 def network(seed):
  torch.manual_seed(seed)
  return nn.Sequential(nn.Linear(1,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,r))
 def residual(net,x):
  z,dz=torch.func.jvp(net,(x,),(torch.ones_like(x),))
  a=z*dt;f=ct+a@lt.T+torch.einsum('ijk,nj,nk->ni',qt,a,a)
  return dz-T*(f@git.T)/dt
 def losses(net):
  ld=((ymt+net(tau)@cd.T-yt)**2).mean();lf=(residual(net,coll)**2).mean();return ld,lf
 # Differentiation and time scaling checks precede all optimization.
 net=network(0);x=coll[::16];eps=1e-6
 _,jvp=torch.func.jvp(net,(x,),(torch.ones_like(x),))
 fd=(net(x+eps)-net(x-eps))/(2*eps)
 checks={'torch_version':torch.__version__,'jvp_max_abs_error':float((jvp-fd).abs().max().detach())}
 testx=torch.tensor([[.23],[.71]])
 _,der=torch.func.jvp(lambda v:torch.cat((v**2,torch.sin(v)),dim=1),(testx,),(torch.ones_like(testx),))
 checks['analytic_physical_time_derivative_error']=float((der/T-torch.cat((2*testx,torch.cos(testx)),dim=1)/T).abs().max())
 loss=sum(losses(net));loss.backward();w=net[0].weight;analytic=float(w.grad[0,0]);old=float(w[0,0].detach())
 with torch.no_grad():w[0,0]=old+eps
 plus=float(sum(losses(net)).detach())
 with torch.no_grad():w[0,0]=old-eps
 minus=float(sum(losses(net)).detach())
 with torch.no_grad():w[0,0]=old
 checks['mixed_parameter_gradient_relative_error']=abs(analytic-(plus-minus)/(2*eps))/max(1,abs(analytic))
 assert checks['jvp_max_abs_error']<1e-7 and checks['mixed_parameter_gradient_relative_error']<1e-6
 (out/'autodiff-checks.json').write_text(json.dumps(checks,indent=2))
 preds={};histories={};fits=[]
 query=torch.linspace(0,1.2/T,121)[:,None]
 for seed in [0,1,2]:
  for weight in [0.,.1,1.]:
   net=network(seed);initial=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in net.parameters())).hexdigest()
   opt=torch.optim.Adam(net.parameters(),lr=.001);history=[];start=time.perf_counter()
   for step in range(2000):
    opt.zero_grad();ld,lf=losses(net);loss=ld+weight*lf;loss.backward();opt.step()
    if step%100==0 or step==1999:history.append([step+1,float(ld.detach()),float(lf.detach())])
   key=f'nn_w{weight:g}_s{seed}';ld,lf=losses(net)
   preds[key]=(net(query).detach().numpy()*D)
   torch.save(net.state_dict(),out/(key+'.pt'));histories[key]=history
   row={'model':key,'seed':seed,'physics_weight':weight,'updates':2000,'initial_sha256':initial,'sensor_mse':float(ld.detach()),'physics_mse':float(lf.detach()),'seconds':time.perf_counter()-start}
   fits.append(row);(out/'training.json').write_text(json.dumps(fits,indent=2));print(row,flush=True)
 np.savez_compressed(out/'neural-predictions.npz',**preds)
 # Nonneural fits use exactly the same sparse online observations.
 gap=np.linalg.lstsq(C,(Y-ym).T,rcond=None)[0].T;preds['gappy_pod']=gap
 X=np.column_stack((aoff[:-1],np.ones(179)));ab=np.linalg.lstsq(X,aoff[1:],rcond=1e-10)[0];A=ab[:r].T;b=ab[r]
 Ak=np.eye(r);d=np.zeros(r);obs=[];rhs=[]
 for k in range(60):
  obs.append(C@Ak@np.diag(D));rhs.append(Y[k]-ym-C@d);Ak=A@Ak;d=A@d+b
 O=np.vstack(obs);q0=np.linalg.lstsq(O,np.concatenate(rhs),rcond=1e-10)[0];a=D*q0;seq=[]
 for k in range(121):seq.append(a.copy());a=A@a+b
 preds['affine_dmd_assimilation']=np.array(seq)
 def f(t,a):return Gi@(c+L@a+np.einsum('ijk,j,k->i',Q,a,a))
 times=np.arange(60)*.01
 def integrate(q,sensitivity=True,times=times):
  def aug(t,v):
   a=v[:r];S=v[r:].reshape(r,r);J=Gi@(L+2*np.einsum('ijk,k->ij',Q,a));return np.r_[f(t,a),(J@S).ravel()]
  init=np.r_[D*q,np.diag(D).ravel()]
  sol=solve_ivp(aug,(0,float(times[-1])),init,t_eval=times,rtol=1e-8,atol=1e-10,method='DOP853')
  if not sol.success:raise RuntimeError(sol.message)
  return sol.y[:r].T,sol.y[r:].T.reshape(-1,r,r)
 cache={}
 def evaluate(q):
  if 'q' not in cache or not np.array_equal(q,cache['q']):
   a,S=integrate(q);cache.update(q=q.copy(),res=(a@C.T+ym-Y).ravel(),jac=np.einsum('ij,njk->nik',C,S).reshape(-1,r))
  return cache
 start=time.perf_counter();fit=least_squares(lambda q:evaluate(q)['res'],gap[0]/D,jac=lambda q:evaluate(q)['jac'],max_nfev=40,ftol=1e-8,xtol=1e-8,gtol=1e-8)
 preds['galerkin_assimilation']=integrate(fit.x,times=np.arange(121)*.01)[0]
 baselines={'galerkin':{'success':bool(fit.success),'message':fit.message,'nfev':fit.nfev,'sensor_mse':float(np.mean(fit.fun**2)),'seconds':time.perf_counter()-start},'dmd_observability_condition':float(np.linalg.cond(O)),'gappy_sensor_condition':float(np.linalg.cond(C))}
 (out/'baseline-fit.json').write_text(json.dumps(baselines,indent=2));print(baselines,flush=True)
 # Evaluation begins only after all fitting is complete.
 truth=data['u'][180:];oracle=np.linalg.solve(G,phi.T@(M@(truth-mean).T)).T
 preds['projection_floor']=oracle;preds['offline_mean']=np.zeros((121,r))
 def energy(v):return np.einsum('ij,ji->',v,M@v.T)
 metrics=[]
 for key,a in preds.items():
  for label,sl in [('online',slice(0,60)),('future',slice(60,121))]:
   if len(a)<121 and label=='future':continue
   aa=a[sl];u=mean+aa@phi.T;target=truth[sl];err=u-target
   metrics.append({'model':key,'window':label,'relative_mass_error':float(np.sqrt(energy(err)/energy(target))),'relative_fluctuation_error':float(np.sqrt(energy(err)/energy(target-mean))),'hidden_probe_rmse':float(np.sqrt(np.mean((HH@err.T)**2)))})
 (out/'metrics.json').write_text(json.dumps(metrics,indent=2));np.savez_compressed(out/'predictions.npz',**preds)
 (out/'histories.json').write_text(json.dumps(histories,indent=2))
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})
 colors=['#7b93a6','#24a69a','#a16aab'];fig,axs=plt.subplots(1,3,figsize=(13,3.7),constrained_layout=True)
 for j,w in enumerate([0.,.1,1.]):
  for seed in [0,1,2]:
   h=np.array(histories[f'nn_w{w:g}_s{seed}']);axs[0].semilogy(h[:,0],h[:,1],color=colors[j],alpha=.7,label=f'lambda={w:g}' if seed==0 else None)
   row=next(m for m in metrics if m['model']==f'nn_w{w:g}_s{seed}' and m['window']=='online');fitrow=next(f for f in fits if f['model']==row['model'])
   axs[1].scatter(fitrow['physics_mse'],100*row['relative_fluctuation_error'],color=colors[j],s=45)
  for window,offset in [('online',-.12),('future',.12)]:
   vals=[100*m['relative_fluctuation_error'] for m in metrics if m['model'].startswith(f'nn_w{w:g}_') and m['window']==window]
   axs[2].scatter(np.full(3,j+offset),vals,color=colors[j],marker='o' if window=='online' else '^',label=window if j==0 else None)
 axs[0].set(xlabel='Adam updates',ylabel='Sensor MSE',title='All nine fixed-budget fits');axs[0].legend(frameon=False)
 axs[1].set(xscale='log',xlabel='Dimensionless physics MSE',ylabel='Online fluctuation error (%)',title='Residual versus reconstruction')
 axs[2].set(xticks=[0,1,2],xticklabels=['0','0.1','1'],xlabel='Physics weight',ylabel='Fluctuation error (%)',title='Spatial reconstruction / extrapolation');axs[2].legend(frameon=False)
 for ext in ['png','svg','pdf']:fig.savefig(out/f'pilot-comparison.{ext}')
 plt.close(fig)
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
