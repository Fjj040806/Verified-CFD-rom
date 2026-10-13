import unittest
import numpy as np
from scipy.sparse import diags
from flowrom.rom import WeightedPOD,ReducedDMD
class TestROM(unittest.TestCase):
 def test_weighted_harmonic_forecast(self):
  dt=.01;t=np.arange(100)*dt;frequency=2.5
  a=np.array([1.,2.,0.,-.5]);b=np.array([0.,1.,2.,.5]);mean=np.array([2.,-1.,.4,1.])
  U=mean[:,None]+a[:,None]*np.cos(2*np.pi*frequency*t)+b[:,None]*np.sin(2*np.pi*frequency*t)
  mass=diags([1.,3.,.5,2.]);pod=WeightedPOD(U[:,:80],mass)
  np.testing.assert_allclose(pod.modes.T@(mass@pod.modes),np.eye(2),atol=1e-12)
  np.testing.assert_allclose(pod.reconstruct(U[:,80:],2),U[:,80:],atol=1e-12)
  model=ReducedDMD(pod,U[:,:80],2,dt)
  self.assertAlmostEqual(model.dominant_frequency,frequency,places=10)
  np.testing.assert_allclose(model.forecast(U[:,79],20),U[:,80:],atol=1e-11)
 def test_training_mean_and_rank_error(self):
  rng=np.random.default_rng(3);train=rng.normal(size=(7,30));test=rng.normal(size=(7,20))+4
  pod=WeightedPOD(train,diags(np.arange(1.,8.)))
  np.testing.assert_allclose(pod.mean,train.mean(axis=1))
  low=pod.errors(pod.reconstruct(test,2),test)['total_relative'];high=pod.errors(pod.reconstruct(test,6),test)['total_relative']
  self.assertLessEqual(high,low)

 def test_affine_forecast_incomplete_cycle(self):
  dt=.01;t=np.arange(100)*dt
  data=np.vstack([3+np.cos(2*np.pi*2.5*t),-2+np.sin(2*np.pi*2.5*t)])
  train=data[:,:53];pod=WeightedPOD(train,diags([1.,2.]))
  affine=ReducedDMD(pod,train,2,dt).forecast(train[:,-1],47)
  linear=ReducedDMD(pod,train,2,dt,affine=False).forecast(train[:,-1],47)
  np.testing.assert_allclose(affine,data[:,53:],atol=1e-10)
  self.assertGreater(np.linalg.norm(linear-data[:,53:]),.1)
