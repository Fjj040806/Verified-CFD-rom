"""Phase 0 plans and an analytic mismatch example. No trained PINN or CFD result."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Rectangle
root=Path(__file__).resolve().parent;out=root/'figures';out.mkdir(exist_ok=True)
cfg=json.loads((root/'pilot-config.json').read_text());teal='#318B88';clay='#CC8868';ink='#243746';gray='#788393'
plt.rcParams.update({'font.size':10,'axes.labelcolor':ink,'text.color':ink,'axes.titlecolor':ink})
fig=plt.figure(figsize=(12,7.8),layout='constrained');gs=fig.add_gridspec(3,1,height_ratios=[2.0,1.15,2.25]);ax=fig.add_subplot(gs[0]);ax.add_patch(Rectangle((0,0),2.2,.41,facecolor='#F0F5F3',edgecolor=ink,lw=1));ax.add_patch(Circle((.2,.2),.05,facecolor='white',edgecolor=ink,lw=1.2))
points=np.array(cfg['sensors']);ax.scatter(*points.T,color=teal,s=48,zorder=3)
for i,(x,y) in enumerate(points):ax.annotate(str(i+1),(x,y),xytext=(5,5),textcoords='offset points',fontsize=9,color=teal)
ax.text(.75,.205,'8 planned locations = 16 velocity channels',color=ink,fontsize=11);ax.text(.03,.35,'Parabolic inlet',fontsize=9);ax.text(1.9,.35,'Natural outlet',fontsize=9);ax.set_aspect('equal');ax.set(xlim=(-.04,2.24),ylim=(-.02,.45),xlabel='x',ylabel='y',title='A · Planned sparse observations in the cylinder channel (not optimized)');ax.spines[['top','right']].set_visible(False)
ax=fig.add_subplot(gs[1]);ax.set(xlim=(27,30),ylim=(0,1),yticks=[],xlabel='Simulation time',title='B · Separate the offline prior, online observations, and future evaluation')
for lo,hi,color,label in [(27,28.79,teal,'Offline complete fields\nPOD basis and operators'),(28.8,29.39,clay,'Sparse online data\nFit coefficient function'),(29.4,30,gray,'Future fields\nEvaluation only')]:
 ax.axvspan(lo,hi,color=color,alpha=.18);ax.text((lo+hi)/2,.5,label,ha='center',va='center',fontsize=10)
ax.set_xticks([27,28,28.8,29.4,30]);ax.spines[['top','right','left']].set_visible(False)
ax=fig.add_subplot(gs[2]);ax.axis('off');ax.set_title('C · A coefficient-function PINN constrained by the reduced FEM equations')
ax.text(.015,.81,r'$\widehat{u}(t)=\overline{u}+\Phi D z_\theta(\tau),\quad \tau=(t-t_A)/T_A$',fontsize=17,transform=ax.transAxes)
ax.text(.015,.52,r'$r_z=\partial_\tau z_\theta-T_A D^{-1}M_r^{-1}F_r(Dz_\theta;\nu)$',fontsize=17,transform=ax.transAxes)
ax.text(.015,.22,r'$\mathcal{L}=\mathcal{L}_{\mathrm{sensor}}+\lambda_f\mathcal{L}_{\mathrm{physics}}$',fontsize=17,transform=ax.transAxes)
ax.text(.63,.77,'Fixed offline basis and mean\nSame sensor information for all baselines',fontsize=10,transform=ax.transAxes,color=gray)
ax.text(.63,.39,'Initial state inferred from sparse data\nNo complete online truth in the loss\nNo future observations or collocation',fontsize=10,transform=ax.transAxes,color=gray)
fig.suptitle('PINN extension · Phase 0 design schematic, not a trained-model result',fontsize=15)
for ext in ['png','svg','pdf']:fig.savefig(out/f'phase0-principle.{ext}',dpi=180)
plt.close(fig)
lam=np.r_[0.,np.logspace(-3,3,121)];alpha=1/(1+lam);data_error=abs(alpha-1);physics_residual=abs(alpha)
# Exact first-order stationarity for a strictly convex analytic loss.
stationarity=2*(alpha-1)+2*lam*alpha
assert np.max(np.abs(stationarity))<1e-12
assert np.all(np.diff(data_error)>=0);assert np.all(np.diff(physics_residual)<=0)
tq=np.linspace(0,2*np.pi,8192,endpoint=False);xq=(np.cos(tq)+np.sin(tq))/2;dxq=(np.cos(tq)-np.sin(tq))/2
assert np.max(abs(dxq+xq-np.cos(tq)))<1e-12
for a in [0.,.1,.5,1.]:
 assert abs(4*np.mean((a*xq-xq)**2)-(a-1)**2)<1e-12
 assert abs(2*np.mean((a*(dxq+xq))**2)-a*a)<1e-12
t=np.linspace(0,2*np.pi,401);truth=(np.cos(t)+np.sin(t))/2
fig,axs=plt.subplots(1,2,figsize=(11,4.2),layout='constrained');ax=axs[0];ax.plot(t,truth,color=ink,lw=2,label='True x(t)')
for weight,color in [(.1,teal),(1,clay),(10,gray)]:ax.plot(t,truth/(1+weight),'--',color=color,label=fr'Constant amplitude fit, $\lambda={weight:g}$')
ax.set(xlabel='Time',ylabel='Resolved coordinate x',title='A · Omitted coordinates can bias a reduced physics constraint');ax.legend(fontsize=8);ax.grid(alpha=.15)
ax=axs[1];ax.semilogx(lam[1:],data_error[1:],color=clay,lw=2,label='Normalized observation error');ax.semilogx(lam[1:],physics_residual[1:],color=teal,lw=2,label='Normalized reduced-physics residual');ax.set(xlabel=r'Physics weight $\lambda$',ylabel='Normalized magnitude',title=r'B · Exact minimizer: $\alpha^*=1/(1+\lambda)$',ylim=(-.03,1.06));ax.legend(fontsize=8);ax.grid(alpha=.15)
fig.suptitle('Analytic truncation illustration · restricted trial family · not CFD or PINN training',fontsize=13)
for ext in ['png','svg','pdf']:fig.savefig(out/f'phase0-analytic-mismatch.{ext}',dpi=180)
plt.close(fig)
(root/'phase0-analytic-data.json').write_text(json.dumps({'status':'analytic illustration; not empirical CFD or PINN results','system':'xdot=-x+y; ydot=-z; zdot=y; initial=(1/2,1,0)','true_resolved_solution':'(cos(t)+sin(t))/2','restricted_trial':'alpha times true resolved solution','loss':'(alpha-1)^2+lambda*alpha^2','lambda':lam.tolist(),'optimal_alpha':alpha.tolist(),'normalized_data_error':data_error.tolist(),'normalized_reduced_residual':physics_residual.tolist()},indent=2))
# Check the design contract, not a training implementation.
assert len(cfg['sensors'])==8 and len(set(map(tuple,cfg['sensors'])))==8
assert cfg['offline_interval'][1]<cfg['online_interval'][0]<cfg['online_interval'][1]<cfg['future_evaluation_interval'][0]
assert cfg['pinn_fits']==len(cfg['physics_weights'])*len(cfg['paired_seeds'])
assert not cfg['full_state_initial_condition_labels'] and not cfg['future_physics_collocation']
for x,y in cfg['sensors']:assert 0<x<2.2 and 0<y<.41 and (x-.2)**2+(y-.2)**2>.05**2
(root/'phase0-checks.json').write_text(json.dumps({'scope':'analytic and configuration checks only; no mesh probes, FEM tensors or trained model verified','pilot_design_consistent':True,'nominal_sensor_coordinates_inside_analytic_fluid_domain':True,'time_intervals_disjoint':True,'future_information_disabled':True,'analytic_stationarity_max_abs':float(np.max(abs(stationarity))),'analytic_error_monotonicity':True,'phase0_complete':True,'next_gate':'saved-mesh operator and sensor validation'},indent=2))
print('Phase 0: configuration and analytic checks passed; two figure sets generated.')
