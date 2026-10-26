"""Plot predeclared seed-zero examples, never a truth-selected best run."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent/'phase2-results';z=np.load(p/'predictions.npz');t=28.8+np.arange(121)*.01
fig,ax=plt.subplots(2,1,figsize=(9,5),sharex=True,constrained_layout=True)
for j,a in enumerate(ax):
 for key,label,color,style in [('projection_floor','Full-field projection (evaluation only)','#253b50','-'),('affine_dmd_assimilation','Sensor-assimilated DMD','#dc9b45','--'),('nn_w0_s0','Data-only, seed 0','#7b93a6','-'),('nn_w0.1_s0','PINN, lambda=0.1, seed 0','#24a69a','-'),('nn_w1_s0','PINN, lambda=1, seed 0','#a16aab',':')]:
  a.plot(t,z[key][:,j],label=label,color=color,ls=style,lw=1.5)
 a.axvline(29.39,color='#6f7780',lw=1,ls=':');a.axvspan(29.39,30,color='#eef1f4',zorder=-2);a.set_ylabel(f'POD coefficient {j+1}');a.spines[['top','right']].set_visible(False)
ax[0].legend(ncol=2,fontsize=8,frameon=False);ax[0].set_title('Sparse reconstruction followed by native future extrapolation');ax[1].set_xlabel('Time');
for ext in ['png','svg','pdf']:fig.savefig(p/f'coefficient-trajectories.{ext}',dpi=180)
