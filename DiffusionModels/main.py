import torch
from UserInputs import *
from IntegrateSDE import *
from ScoreMatching import *
from DataManagement import *

from predictor_rk4 import *
from Auxiliary import *

import matplotlib.pyplot as plt
import os.path

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

#Model dimensions setup
Batch = 1000
TrainPerc = 0.9
TrainSplit = round(Batch*TrainPerc)

InitialVector = torch.tensor([1,1,1])
state_d = InitialVector.shape[0]
C = torch.eye(state_d)
observation_d = C.shape[0]

#Error covariances
ToggleNoise = 1

std_P = 0.05
P_sqrt = std_P*torch.eye(state_d)

std_Q = 0.05*ToggleNoise
Q_sqrt = std_Q*torch.eye(state_d)

std_R = 0.05*ToggleNoise
R_sqrt = std_R*torch.eye(observation_d)

#Initial population
TrueState = InitialVector.detach().clone().unsqueeze(0)
NoisyState = InitialVector.detach().clone() + torch.randn((1,state_d))@( P_sqrt.t() )
Particles = InitialVector + torch.randn((Batch,state_d))@( P_sqrt.t() )

#Integration params
NSteps = 100
dt = 0.01
t = 0.

#Plot containers & related params
EmpiricalMean = torch.zeros((NSteps,state_d))
EmpiricalCovariance = torch.zeros((NSteps,state_d,state_d))
TrueTrajectory = torch.zeros((NSteps,state_d))
Confidence = 0.95
Slice = [0,1] #x,y
ShowSteps = True

#Set-up structures
Params = Parameters(dt=0.01)
Integrate = IntegrateSDE(Params)
Loss = ScoreLoss()

#Warmstart training
if os.path.isfile(os.path.dirname(os.path.abspath(__file__))+'/Models/model_warmstart.pth'):
    ScoreModelInit = torch.load(os.path.dirname(os.path.abspath(__file__))+'/Models/model_warmstart.pth', weights_only=False)
else:
    ScoreModelInit = ConditionalScoreNetwork1D(state_d=state_d,observation_d=observation_d,temb_d=4)

#Begin Data Assimilation
for step in range(NSteps):
    ParticlesOld = Particles.clone() #Save particles before data assimilation

    #Update particles with forward mechanics and collect observations
    t, Particles = predict_rk4(Particles, t, dt, lorenz63_batch)
    Particles = Particles + torch.randn((Batch,state_d))@( Q_sqrt.t() ) #Add model noise

    Observations = Particles@( C.t() )
    Observations = Observations + torch.randn((Batch,observation_d))@( R_sqrt.t() ) #Add measurement noise

    ParticlesPredicted = Particles.clone() #Save particles after predictor

    #Create train and test datasets
    training_data = ConditionalDiffusionDataset1D(Particles[0:TrainSplit,...],Observations[0:TrainSplit,...])
    test_data = ConditionalDiffusionDataset1D(Particles[TrainSplit:,...],Observations[TrainSplit:,...])

    #Train score function
    Score = ScoreMatching(Params,ScoreModelInit,training_data,test_data,Loss,batch_size=256,learning_rate=1e-4,epochs=200)
    Score.Train()
    ScoreModel = Score.ScoreModel

    #NEW OBSERVATION ARRIVES (from PDE/ODE solver!)
    _, TrueState = predict_rk4(TrueState, t, dt, lorenz63_batch)
    TrueTrajectory[step,:] = TrueState

    _, NoisyState = predict_rk4(NoisyState, t, dt, lorenz63_batch)
    NoisyState = NoisyState + torch.randn((1,state_d))@( Q_sqrt.t() ) #Add model noise
    TrueTrajectory[step,:] = TrueState
    NewObservation = NoisyState@( C.t() )
    NewObservation = NewObservation + torch.randn((1,observation_d))@( R_sqrt.t() ) #Add measurement noise

    #Resample particles as gaussian noise and evolve them using the SDE
    sigmaT = Params.m_sigma(Params.T*torch.ones((1,1)))[1]
    Particles = sigmaT*torch.randn((Batch,state_d))
    Observations = NewObservation.repeat((Batch,1))
    X = [Particles, Observations]
    Particles = Integrate.EvolveSDEParticles(X,ScoreModel)[...,-1]

    ParticlesNew = Particles.clone() #Save particles after data assimilation

    #Assemble empirical mean and covariance
    EmpiricalMean[step,:], EmpiricalCovariance[step,:,:] = ComputeMeanAndCov(Particles)
    Ellipsoid = GetCovEllipsoid(EmpiricalMean[step,Slice],EmpiricalCovariance[step][np.ix_(Slice, Slice)],Confidence)

    if ShowSteps:
        #Plot particles
        fig = plt.figure(1,figsize=(6,6))
        plt.draw()
        plt.pause(0.01)
        plt.clf()
        ax = fig.add_subplot(title=f"Step: {step:>3d}")
    
        ax.scatter(*(ParticlesOld[:,Slice].t()),color='red',marker=".",alpha=0.5,label='Old particles') #Before data assimilation
        ax.scatter(*(ParticlesPredicted[:,Slice].t()),color='orange',marker=".",alpha=0.5,label='After predictor') #After predictor
        ax.scatter(*(ParticlesNew[:,Slice].t()),color='green',marker=".",alpha=0.5,label='After D.A.') #After data assimilation

        ax.scatter(*NewObservation[0,Slice],color='blue',label='Measurement')
        ax.scatter(*TrueState[0,Slice],color='black',label='True State')
        ax.scatter(*EmpiricalMean[step,Slice],color='magenta',label='Empirical Mean')

        PlotLegend = ax.legend()
        plt.gca().add_artist(PlotLegend)
        ax.add_artist(Ellipsoid)
        plt.legend([Ellipsoid], [str(100 * Confidence) + "% " + "confidence"],loc="upper right")
        ax.axis('equal')
        ax.set_axisbelow(True)
        ax.grid(color='gray',linestyle='dashed',alpha=0.5)
        plt.draw()
        plt.pause(0.01)


fig = plt.figure(2)
ax = fig.add_subplot(projection='3d',title="Trajectory + Uncertainty Quantification")
ax.plot(*(EmpiricalMean.t()),color="magenta",label='Empirical Mean')
ax.plot(*(TrueTrajectory.t()),color="black",label='True Trajectory')
PlotLegend = ax.legend()
for step in range(NSteps):
    Ellipsoid = GetCovEllipsoid(EmpiricalMean[step,:],EmpiricalCovariance[step,:,:],Confidence)
    ax.plot_surface(*Ellipsoid, rstride=4, cstride=4, color='magenta', alpha=0.1)
ax.axis('equal')
plt.show()


#TODO LIST
#Numerical integrator for lorenz63 (DONE)
#Predictor (DONE)
#Score function (DONE)

#Figure out if the network is training properly, try different architectures (D)
#Take a closer look at the Euler-Maruyama integrator: is it working as it should? (E)

#Assume an inaccurate predictor: on line 75 the particles are updated using an inaccurate predictor, 
# but on line 91 we use the real, accurate one to generate the observations. What changes? Compute empirical mean and covariance of particles
# at each D.A. step and compare them in a case where the predictor is accurate vs inaccurate.


#Code a way to display the mean particle at each step w/ covariance @ 1 std (some kind of uncertainty quantification) (!) (E)

#Experiments to try
#What happens when we increase noise levels? (Empirical covariance matrices at each DA step) (D)
#What happens when we measure only a subset of the state variables? (D)

#Comparison with DKF (plot estimated trajectories together)
