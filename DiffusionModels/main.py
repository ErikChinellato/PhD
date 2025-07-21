import torch
from UserInputs import *
from IntegrateSDE import *
from ScoreMatching import *
from DataManagement import *

from predictor_rk4 import *

import matplotlib.pyplot as plt

DummyTest = False #You should be able to run this with no errors

if DummyTest:
    Params = Parameters(T=1,dt=0.001,
                 mu=2,alpha=1,
                 gamma=lambda t: 12**(2*t),
                 beta=lambda t: 0.01 + t*(20-0.01),
                 VarianceMode='VP')
    Integrate = IntegrateSDE(Params)
    Loss = ScoreLoss()  
    ScoreModelInit = DUMMYConditionalScoreNetwork1D()

    #Dummy training data
    Particles = torch.rand((10000,8))
    Observations = torch.rand((10000,4))

    training_data = ConditionalDiffusionDataset1D(Particles[0:9000,...],Observations[0:9000,...])
    test_data = ConditionalDiffusionDataset1D(Particles[9000:,...],Observations[9000:,...])

    Score = ScoreMatching(Params,ScoreModelInit,training_data,test_data,Loss,batch_size=50,learning_rate=1e-4,epochs=10)
    Score.Train()
    ScoreModel = Score.ScoreModel

    sigmaTSq = Params.m_sigma(Params.T*torch.ones((1,1)))[1]**2
    Particles = sigmaTSq*torch.randn((10000,8))

    HistParticles = Integrate.EvolveSDEParticles(Particles,ScoreModel)
    torch.save(HistParticles, 'HistParticles.pth')


#DATA ASSIMILATION IN TIME

#Set-up structures
Params = Parameters()
Integrate = IntegrateSDE(Params)
Loss = ScoreLoss()
ScoreModelInit = ConditionalScoreNetwork1D()

#Particles sampled from initial (given, maybe gaussian around a set value) distribution
Batch = 1000
TrainPerc = 0.9
TrainSplit = round(Batch*TrainPerc)

InitialVector = torch.tensor([1,1,1])
input_d = InitialVector.shape[0]
C = torch.eye(input_d)

TrueState = InitialVector.detach().clone().unsqueeze(0)
Particles = InitialVector + torch.randn((Batch,input_d))

#Integration params
NSteps = 5
dt = 0.01
t = 0.

#Begin Data Assimilation
for step in range(NSteps):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, title=f"Step: {step:>3d}")

    #Plot particles before data assimilation
    ax.scatter(Particles[:,0],Particles[:,1],color='red',label='Old particles')

    #Update particles with forward mechanics and collect observations
    t, Particles = predict_rk4(Particles, t, dt, lorenz63_batch)
    Observations = Particles@(torch.transpose(C,dim0=0,dim1=1))

    #Plot particles after predictor
    plt.scatter(Particles[:,0],Particles[:,1],color='orange',label='After predictor')

    #Create train and test datasets
    training_data = ConditionalDiffusionDataset1D(Particles[0:TrainSplit,...],Observations[0:TrainSplit,...])
    test_data = ConditionalDiffusionDataset1D(Particles[TrainSplit:,...],Observations[TrainSplit:,...])

    #Train score function
    Score = ScoreMatching(Params,ScoreModelInit,training_data,test_data,Loss,batch_size=256,learning_rate=1e-3,epochs=1000)
    Score.Train()
    ScoreModel = Score.ScoreModel

    #NEW OBSERVATION ARRIVES (from PDE/ODE solver!)
    _, TrueState = predict_rk4(TrueState, t, dt, lorenz63_batch)
    NewObservation = TrueState@(torch.transpose(C,dim0=0,dim1=1))

    #Resample particles as gaussian noise and evolve them using the SDE
    sigmaT = Params.m_sigma(Params.T*torch.ones((1,1)))[1]
    Particles = sigmaT*torch.randn((Batch,input_d))
    Observations = NewObservation.repeat((Batch,1))
    X = [Particles, Observations]
    Particles = Integrate.EvolveSDEParticles(X,ScoreModel)[...,-1]

    #Plot particles after data assimilation
    ax.scatter(Particles[:,0],Particles[:,1],color='green',label='After D.A.')
    ax.scatter(NewObservation[0,0],NewObservation[0,1],color='blue',label='Measurement')
    ax.legend()
    plt.show()



#TODO LIST
#Numerical integrator for lorenz63 (DONE)
#Predictor (DONE)
#Score function (DONE)

#Figure out if the network is training properly, try different architectures
#Take a closer look at the Euler-Maruyama integrator: is it working as it should?

#Assume an inaccurate predictor: on line 68 the particles are updated using an inaccurate predictor, 
# but on line 85 we use the real, accurate one to generate the observations. What changes? Compute empirical mean and covariance of particles
# at each D.A. step and compare them in a case where the predictor is accurate vs inaccurate.


