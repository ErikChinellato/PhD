import torch

from Experiments import *

#DATA ASSIMILATION IN TIME

#Initial vector
InitialVector = torch.tensor([-5.6866,-8.4929,17.8452])

#Setup SDE structures
Params = Parameters(dt=0.01)

#D.A. integration params
NSteps = 100
dt = 0.001
t = 0.

#Setup error standard deviations
std_P = 0.05
std_Q = 0.1
std_R = 1.

#Setup vector fields
TrueVectorField = Lorenz63(eps=1e-2)
ApproxVectorField = Lorenz63(eps=1e-2)

#Create experiment instance
DA = DataAssimilationInTime_Conditional1D(Params,NSteps,dt,t,InitialVector,
                                          TrueVectorField,ApproxVectorField,TruePredictorMethod='dopri5',ApproxPredictorMethod='rk4',ObserverMethod='id',
                                          std_P=std_P,std_Q=std_Q,std_R=std_R,
                                          ParticlesNum=1000,TrainPerc=0.9,BatchSize=256,Epochs=200,LR=1e-4,
                                          Warmstart=True,LoadWarmstart=False,WarmstartParticlesNum=10000,WarmstartBatchSize=256,WarmstartEpochs=500,WarmstartLR=1e-4,WarmstartSampleOffset=0,WarmstartSampleScale=1,
                                          ShowSteps=True,ShowMeasurement=True,ShowOutput=True,ShowSlice=[0,1],ShowConfidence=0.95)

#Run data assimilation
DA.run()
EmpiricalMean, EmpiricalCovariance, TrueTrajectory, NoisyTrajectory, Measurements = DA.collect()