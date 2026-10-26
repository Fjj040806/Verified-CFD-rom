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
 out=HERE/'phase3-results';out.mkdir(exist_ok=True)
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
 # Predeclared continuation: fixed final iterate, no truth-based selection.
 config={'seeds':[0,1,2],'beta':[0,.1,1],'additional_updates':4000,'ramp_updates':2000,'learning_rate':.0003,'normalization':'each initial checkpoint sensor/physics MSE, fixed thereafter','optimizer':'fresh Adam; all arms restart optimizer','evaluation_selection':'all final iterates; no full-field selection'}
 (out/'config.json').write_text(json.dumps(config,indent=2))
 preds={};fits=[];histories={};query=torch.linspace(0,1.2/T,121)[:,None]
 for seed in config['seeds']:
  for beta in config['beta']:
   net=network(seed);net.load_state_dict(torch.load(HERE/'phase2-results'/f'nn_w0_s{seed}.pt',weights_only=True))
   ld0,lf0=losses(net);sd=max(float(ld0.detach()),1e-12);sf=max(float(lf0.detach()),1e-12)
   opt=torch.optim.Adam(net.parameters(),lr=config['learning_rate']);history=[];start=time.perf_counter()
   for step in range(config['additional_updates']):
    ramp=min((step+1)/config['ramp_updates'],1.);opt.zero_grad();ld,lf=losses(net)
    loss=ld/sd+beta*ramp*lf/sf;loss.backward();opt.step()
    if step%100==0 or step==3999:history.append([step+1,float(ld.detach()),float(lf.detach()),beta*ramp])
   key=f'continuation_b{beta:g}_s{seed}';ld,lf=losses(net);preds[key]=net(query).detach().numpy()*D
   torch.save(net.state_dict(),out/(key+'.pt'));histories[key]=history
   row={'model':key,'seed':seed,'beta':beta,'sensor_mse':float(ld.detach()),'physics_mse':float(lf.detach()),'initial_sensor_mse':sd,'initial_physics_mse':sf,'final_effective_raw_lambda':beta*sd/sf,'additional_updates':4000,'seconds':time.perf_counter()-start};fits.append(row)
   (out/'training.json').write_text(json.dumps(fits,indent=2));print(row,flush=True)
 # Complete field evaluation is deferred until all checkpoints are frozen.
 truth=data['u'][180:];metrics=[]
 def energy(v):return np.einsum('ij,ji->',v,M@v.T)
 for key,a in preds.items():
  for window,sl in [('online',slice(0,60)),('future',slice(60,121))]:
   err=mean+a[sl]@phi.T-truth[sl]
   metrics.append({'model':key,'window':window,'relative_fluctuation_error':float(np.sqrt(energy(err)/energy(truth[sl]-mean))),'relative_mass_error':float(np.sqrt(energy(err)/energy(truth[sl]))),'hidden_probe_rmse':float(np.sqrt(np.mean((HH@err.T)**2)))})
 (out/'metrics.json').write_text(json.dumps(metrics,indent=2));(out/'histories.json').write_text(json.dumps(histories));np.savez_compressed(out/'predictions.npz',**preds)
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
