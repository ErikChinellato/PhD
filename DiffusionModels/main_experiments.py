import torch
from UserInputs import *
from IntegrateSDE import *
from ScoreMatching import *
from DataManagement import *

from Auxiliary import *
from ModelComponents import *

import matplotlib.pyplot as plt
import os.path
import scipy.io as sio

#DATA ASSIMILATION IN TIME

#Initial population setup
Batch = 1000
TrainPerc = 0.9
TrainSplit = round(Batch*TrainPerc)

InitialVector = torch.tensor([-5.6866,-8.4929,17.8452])#torch.tensor([1,1,1])
state_d = InitialVector.shape[0]

std_P = 0.05
P_sqrt = std_P*torch.eye(state_d)

#D.A. integration params
NSteps = 100
dt = 0.001
t = 0.

#Setup SDE structures
Params = Parameters(dt=0.01)
Integrate = IntegrateSDE(Params)

std_Q_Vals = [0.,0.15,0.3]
std_R_Vals = [0.,1./3.,2./3.,1.]

for ParticleInd in range(11,15):
    for IndQ in range(len(std_Q_Vals)):
        for IndR in range(len(std_R_Vals)):
            TrueState = InitialVector.detach().clone().unsqueeze(0) #Used as a reference
            NoisyState = InitialVector.detach().clone() + torch.randn((1,state_d))@( P_sqrt.t() ) #Used to generate measurements for D.A. at each time-step
            Particles = InitialVector + torch.randn((Batch,state_d))@( P_sqrt.t() )

            Meas = torch.tensor(sio.loadmat("DiffusionModels/Meas_"+str(IndQ)+"_"+str(IndR)+".mat")['Meas'][:,1:],dtype=torch.float).transpose(0,1)

            #Setup predictors and observers
            std_Q = std_Q_Vals[IndQ]
            std_R = std_R_Vals[IndR]

            TrueVectorField = Lorenz63(eps=1e-2)
            TruePredictor = Predictors(state_d=state_d,dt=dt,std_Q=std_Q,VectorFieldClass=TrueVectorField,method='dopri5')

            ApproxVectorField = Lorenz63(eps=1e-2)
            ApproxPredictor = Predictors(state_d=state_d,dt=dt,std_Q=std_Q,VectorFieldClass=ApproxVectorField,method='rk4')

            Observer = Observers(default_d=state_d,std_R=std_R,method='id')
            observation_d = Observer.get_dimension()

            #Plot containers & related params
            EmpiricalMean = torch.zeros((NSteps,state_d))
            EmpiricalCovariance = torch.zeros((NSteps,state_d,state_d))
            TrueTrajectory = torch.zeros((NSteps,state_d))
            Confidence = 0.95
            Slice = [0,1] #x,y
            ShowSteps = False
            ShowOutput = False

            #Setup score model
            Warmstart = True
            Loss = ScoreLoss()

            if Warmstart:
                if 0:#os.path.isfile(os.path.dirname(os.path.abspath(__file__))+'/Models/model_warmstart.pth'):
                    ScoreModelInit = torch.load(os.path.dirname(os.path.abspath(__file__))+'/Models/model_warmstart.pth',weights_only=False)
                else:
                    ScoreModelInit = ConditionalScoreNetwork1D(state_d=state_d,observation_d=observation_d,temb_d=4)

                    WarmstartBatch = 10000
                    WarmstartTrainSplit = round(WarmstartBatch*TrainPerc)

                    WarmstartRange = [-20,20]
                    
                    WarmstartParticles = 40*torch.rand(WarmstartBatch,state_d) + torch.tensor([-20,-20,0])

                    _, WarmstartParticlesTmp = ApproxPredictor.forward(0,WarmstartParticles,AddNoise=True)
                    WarmstartObservations = Observer.forward(WarmstartParticlesTmp,AddNoise=True)
                    _, WarmstartParticles = ApproxPredictor.forward(0,WarmstartParticles,AddNoise=True)

                    training_data = ConditionalDiffusionDataset1D(WarmstartParticles[0:WarmstartTrainSplit,...],WarmstartObservations[0:WarmstartTrainSplit,...])
                    test_data = ConditionalDiffusionDataset1D(WarmstartParticles[WarmstartTrainSplit:,...],WarmstartObservations[WarmstartTrainSplit:,...])

                    Score = ScoreMatching(Params,ScoreModelInit,training_data,test_data,Loss,batch_size=256,learning_rate=1e-3,epochs=500)
                    Score.Train()
                    ScoreModelInit = Score.ScoreModel
                    torch.save(ScoreModelInit,os.path.dirname(os.path.abspath(__file__))+'/Models/model_warmstart.pth')
            else:
                ScoreModelInit = ConditionalScoreNetwork1D(state_d=state_d,observation_d=observation_d,temb_d=4)


            #Begin Data Assimilation
            for step in range(NSteps):
                ParticlesOld = Particles.clone() #Save particles before data assimilation

                #Update particles with forward mechanics and collect observations
                _, ParticlesTmp = ApproxPredictor.forward(t,Particles,AddNoise=True)
                Observations = Observer.forward(ParticlesTmp,AddNoise=True)

                _, Particles = ApproxPredictor.forward(t,Particles)
                ParticlesPredicted = Particles.clone() #Save particles after predictor

                #Create train and test datasets
                training_data = ConditionalDiffusionDataset1D(Particles[0:TrainSplit,...],Observations[0:TrainSplit,...])
                test_data = ConditionalDiffusionDataset1D(Particles[TrainSplit:,...],Observations[TrainSplit:,...])

                #Train score function
                Score = ScoreMatching(Params,ScoreModelInit,training_data,test_data,Loss,batch_size=256,learning_rate=1e-4,epochs=200)
                Score.Train()
                ScoreModel = Score.ScoreModel

                #NEW OBSERVATION ARRIVES (from PDE/ODE solver!)
                _, TrueState = TruePredictor.forward(t,TrueState)
                TrueTrajectory[step,:] = TrueState

                t, NoisyState = TruePredictor.forward(t,NoisyState,AddNoise=True)
                NewObservation = Observer.forward(NoisyState,AddNoise=True)
                NewObservation = Meas[step,:].unsqueeze(0)#Overwrite measurement for comparison
                
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

            if ShowOutput:
                fig = plt.figure(2)
                plt.draw()
                plt.pause(0.01)
                plt.clf()
                ax = fig.add_subplot(projection='3d',title="Trajectory + Uncertainty Quantification")
                ax.plot(*(EmpiricalMean.t()),color="magenta",label='Empirical Mean')
                ax.plot(*(TrueTrajectory.t()),color="black",label='True Trajectory')
                PlotLegend = ax.legend()
                for step in range(NSteps):
                    Ellipsoid = GetCovEllipsoid(EmpiricalMean[step,:],EmpiricalCovariance[step,:,:],Confidence)
                    ax.plot_surface(*Ellipsoid,rstride=4,cstride=4,color='magenta',alpha=0.1)
                ax.axis('equal')
                plt.draw()
                plt.pause(0.01)

            torch.save(EmpiricalMean,'DiffusionModels/DiffModels_Accurate/EmpMean_'+str(IndQ)+'_'+str(IndR)+'_'+str(ParticleInd)+'.pt')
            torch.save(EmpiricalCovariance,'DiffusionModels/DiffModels_Accurate/EmpCov_'+str(IndQ)+'_'+str(IndR)+'_'+str(ParticleInd)+'.pt')


#torch.save(TrueTrajectory,'TrueTraj.pt')

#TODO LIST
#Numerical integrator for lorenz63 (DONE)
#Predictor (DONE)
#Score function (DONE)

#Figure out if the network is training properly, try different architectures (D)
#Take a closer look at the Euler-Maruyama integrator: is it working as it should? (E)

#Assume an inaccurate predictor: on line 75 the particles are updated using an inaccurate predictor, 
# but on line 91 we use the real, accurate one to generate the observations. What changes? Compute empirical mean and covariance of particles
# at each D.A. step and compare them in a case where the predictor is accurate vs inaccurate.


#Code a way to display the mean particle at each step w/ covariance @ 1 std (some kind of uncertainty quantification) (!) (DONE)

#Experiments to try
#What happens when we increase noise levels? (Empirical covariance matrices at each DA step) (D)
#What happens when we measure only a subset of the state variables? (D)

#Comparison with DKF (plot estimated trajectories together)


#Try different architectures (general ablation study) avg test loss as a a metric

#RMSE - pick a parameter configuration (predictor/vector field/model and measurement noise std: std_Q = [0,0.15,0.3] std_R = linspace(0,1,4) std_P fixed.) and compute 10 trjctrs 


#E:incorrect vector field
#D:correct vector field 