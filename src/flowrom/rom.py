"""Mass-weighted POD and reduced-coordinate DMD, with explicit held-out forecasts."""
import numpy as np
from scipy.linalg import eigh

class WeightedPOD:
 def __init__(self,training,mass):
  self.mass=mass;self.mean=training.mean(axis=1);X=training-self.mean[:,None]
  gram=X.T@(mass@X);values,vectors=eigh((gram+gram.T)/2)
  order=np.argsort(values)[::-1];values=values[order];vectors=vectors[:,order]
  keep=values>max(values[0]*1e-12,1e-25);self.values=values[keep]
  self.modes=(X@vectors[:,keep])/np.sqrt(self.values)[None,:]
 def coefficients(self,data,rank):return self.modes[:,:rank].T@(self.mass@(data-self.mean[:,None]))
 def reconstruct(self,data,rank):return self.mean[:,None]+self.modes[:,:rank]@self.coefficients(data,rank)
 def errors(self,predicted,actual):
  e=predicted-actual;fluct=actual-self.mean[:,None]
  norm=lambda a:np.sqrt(np.sum(a*(self.mass@a)))
  return dict(total_relative=float(norm(e)/norm(actual)),fluctuation_relative=float(norm(e)/norm(fluct)))

class ReducedDMD:
 def __init__(self,pod,training,rank,dt,affine=True):
  self.pod=pod;self.rank=rank;self.dt=dt
  c=pod.coefficients(training,rank)
  if affine:
   fit=c[:,1:]@np.linalg.pinv(np.vstack([c[:,:-1],np.ones(c.shape[1]-1)]),rcond=1e-10)
   self.A=fit[:,:rank];self.bias=fit[:,rank]
  else:self.A=c[:,1:]@np.linalg.pinv(c[:,:-1],rcond=1e-10);self.bias=np.zeros(rank)
  self.eigenvalues,self.eigenvectors=np.linalg.eig(self.A.astype(complex))
  modal=np.linalg.pinv(self.eigenvectors)@c
  activity=np.mean(np.abs(modal)**2,axis=1)*np.sum(np.abs(self.eigenvectors)**2,axis=0)
  self.frequencies=np.imag(np.log(self.eigenvalues))/(2*np.pi*dt)
  positive=np.flatnonzero(self.frequencies>1e-6)
  self.dominant_frequency=float(self.frequencies[positive[np.argmax(activity[positive])]]) if len(positive) else None
 def forecast(self,last_observed,steps):
  c=self.pod.coefficients(last_observed[:,None],self.rank)[:,0];sequence=[]
  for _ in range(steps):c=self.A@c+self.bias;sequence.append(c.copy())
  return self.pod.mean[:,None]+self.pod.modes[:,:self.rank]@np.array(sequence).T
