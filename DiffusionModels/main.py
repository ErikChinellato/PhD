import torch
from UserInputs import *
from IntegrateSDE import *
from ScoreMatching import *
from DataManagement import *

DummyTest = True #You should be able to run this with no errors

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
#Params = Parameters()
#Integrate = IntegrateSDE(Params)
#Loss = ScoreLoss()

#ScoreModelInit = ...

#Particles sampled from initial (given, maybe gaussian around a set value) distribution
#Particles = ...

#for loop 
    #Update particles with forward mechanics and collect observations
    #Particles = F(Particles) #Explicit or implicit euler
    #Observations = C(Particles)

    #Create train and test datasets
    #training_data = fun(Particles,Obervations)
    #test_data = fun(Particles,Observations)

    #Train score function
    #Score = ScoreMatching(Params,ScoreModelInit,training_data,test_data,Loss)
    #Score.Train()
    #ScoreModel = Score.ScoreModel

    #NEW OBSERVATION ARRIVES (from PDE/ODE solver!)
    #Observation = ...

    #Resample particles as gaussian noise and evolve them using the SDE
    #Particles = torch.randn(...)
    #Particles = Integrate.EvolveSDEParticles(Particles,Observation,ScoreModel)[...,-1]



#TODO LIST
#Numerical integrator for lorenz63
#Predictor
#Score function

