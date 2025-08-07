import torch
import numpy as np
from UserInputs import *
from IntegrateSDE import *
from ScoreMatching import *
from DataManagement import *
from Auxiliary import *
from ModelComponents import *

import matplotlib.pyplot as plt
import os.path

class Experiment():
    def __init__(self):
        pass

class DataAssimilationInTime_Conditional1D(Experiment):
    def __init__(self,Params,NSteps,dt,t,InitialVector,
                 TrueVectorField,ApproxVectorField,TruePredictorMethod='dopri5',ApproxPredictorMethod='rk4',ObserverMethod='id',ObserverSlice=None,ObserverTransform=None,
                 std_P=0.,std_Q=0.,std_R=0.,P_sqrt=None,Q_sqrt=None,R_sqrt=None,
                 ParticlesNum=1000,TrainPerc=0.9,BatchSize=256,Epochs=200,LR=1e-4,
                 Warmstart=True,LoadWarmstart=False,WarmstartParticlesNum=10000,WarmstartBatchSize=256,WarmstartEpochs=500,WarmstartLR=1e-3,WarmstartSampleOffset=0,WarmstartSampleScale=1,
                 ShowSteps=False,ShowMeasurement=False,ShowOutput=False,ShowSlice=None,ShowConfidence=0.95):
        super(DataAssimilationInTime_Conditional1D,self).__init__()
        self.Params = Params
        self.Integrate = IntegrateSDE(Params)

        self.NSteps = NSteps
        self.dt = dt
        self.t0 = t
        self.t = t
        self.InitialVector = InitialVector
        self.state_d = len(InitialVector)

        self.P_sqrt = std_P*torch.eye(self.state_d) if P_sqrt is None else P_sqrt
        self.TruePredictor = Predictors(state_d=self.state_d,dt=dt,std_Q=std_Q,Q_sqrt=Q_sqrt,VectorFieldClass=TrueVectorField,method=TruePredictorMethod)
        self.ApproxPredictor = Predictors(state_d=self.state_d,dt=dt,std_Q=std_Q,Q_sqrt=Q_sqrt,VectorFieldClass=ApproxVectorField,method=ApproxPredictorMethod)
        self.Observer = Observers(default_d=self.state_d,slice=ObserverSlice,transform=ObserverTransform,std_R=std_R,R_sqrt=R_sqrt,method=ObserverMethod)
        self.observation_d = self.Observer.get_dimension()

        self.Loss = ScoreLoss()
        self.ParticlesNum = ParticlesNum
        self.TrainPerc = TrainPerc
        self.TrainSplit = round(ParticlesNum*TrainPerc)
        self.BatchSize = BatchSize
        self.Epochs = Epochs
        self.LR = LR

        self.Warmstart = Warmstart
        self.LoadWarmstart = LoadWarmstart
        self.WarmstartParticlesNum = WarmstartParticlesNum
        self.WarmstartBatchSize = WarmstartBatchSize
        self.WarmstartEpochs = WarmstartEpochs
        self.WarmstartLR = WarmstartLR
        self.WarmstartTrainSplit = round(WarmstartParticlesNum*TrainPerc)
        self.WarmstartSampleOffset = WarmstartSampleOffset
        self.WarmstartSampleScale = WarmstartSampleScale

        self.ShowSteps = ShowSteps
        self.ShowMeasurement = ShowMeasurement
        self.ShowOutput = ShowOutput
        if ShowSteps:
            assert ShowSlice is not None, "Slice required if ShowSteps = True."
        self.ShowSlice = ShowSlice
        self.ShowConfidence = ShowConfidence

        self.EmpiricalMean = torch.zeros((NSteps,self.state_d))
        self.EmpiricalCovariance = torch.zeros((NSteps,self.state_d,self.state_d))
        self.TrueTrajectory = torch.zeros((NSteps,self.state_d))
        self.NoisyTrajectory = torch.zeros((NSteps,self.state_d))
        self.Measurements = torch.zeros((NSteps,self.state_d))


    def run(self):
        ScoreModelInit = self.init_model()
        Particles, NoisyState, TrueState = self.init_particles()

        for StepInd in range(self.NSteps):
            Particles, ParticlesOld, ParticlesPredicted, ParticlesNew, NoisyState, TrueState, Measurement = self.step(Particles,NoisyState,TrueState,ScoreModelInit)
            self.manage_data(StepInd,Particles,ParticlesOld,ParticlesPredicted,ParticlesNew,NoisyState,TrueState,Measurement)

        self.plot_output()

    def collect(self):
        return self.EmpiricalMean, self.EmpiricalCovariance, self.TrueTrajectory, self.NoisyTrajectory, self.Measurements

    def manage_data(self,StepInd,Particles,ParticlesOld,ParticlesPredicted,ParticlesNew,NoisyState,TrueState,Measurement):
            self.EmpiricalMean[StepInd,:], self.EmpiricalCovariance[StepInd,:,:] = ComputeMeanAndCov(Particles)
            self.TrueTrajectory[StepInd,:] = TrueState
            self.NoisyTrajectory[StepInd,:] = NoisyState
            self.Measurements[StepInd,:] = Measurement

            if self.ShowSteps:
                self.plot_step(StepInd,ParticlesOld,ParticlesPredicted,ParticlesNew,TrueState,Measurement)

    def step(self,Particles,NoisyState,TrueState,ScoreModelInit):
        ParticlesOld = Particles.detach().clone() #Save particles before data assimilation

        #Update particles with forward mechanics and collect observations
        _, ParticlesTmp = self.ApproxPredictor.forward(self.t,Particles,AddNoise=True)
        Observations = self.Observer.forward(ParticlesTmp,AddNoise=True) #Realistic observations computed by adding noise

        _, Particles = self.ApproxPredictor.forward(self.t,Particles)
        ParticlesPredicted = Particles.detach().clone() #Save particles after predictor

        #Create train and test datasets
        training_data = ConditionalDiffusionDataset1D(Particles[0:self.TrainSplit,...],Observations[0:self.TrainSplit,...])
        test_data = ConditionalDiffusionDataset1D(Particles[self.TrainSplit:,...],Observations[self.TrainSplit:,...])

        #Train score function
        Score = ScoreMatching(self.Params,ScoreModelInit,training_data,test_data,self.Loss,batch_size=self.BatchSize,learning_rate=self.LR,epochs=self.Epochs)
        Score.Train()
        ScoreModel = Score.ScoreModel

        #NEW OBSERVATION ARRIVES
        _, TrueState = self.TruePredictor.forward(self.t,TrueState)

        self.t, NoisyState = self.TruePredictor.forward(self.t,NoisyState,AddNoise=True)
        Measurement = self.Observer.forward(NoisyState,AddNoise=True)

        #Resample particles as gaussian noise and evolve them using the SDE
        sigmaT = self.Params.m_sigma(self.Params.T*torch.ones((1,1)))[1]
        Particles = sigmaT*torch.randn((self.ParticlesNum,self.state_d))
        Observations = Measurement.repeat((self.ParticlesNum,1))
        X = [Particles, Observations]
        Particles = self.Integrate.EvolveSDEParticles(X,ScoreModel)[...,-1]

        ParticlesNew = Particles.detach().clone() #Save particles after data assimilation

        return Particles, ParticlesOld, ParticlesPredicted, ParticlesNew, NoisyState, TrueState, Measurement

    def init_particles(self):
        TrueState = self.InitialVector.detach().clone().unsqueeze(0) #Used as a reference
        NoisyState = self.InitialVector.detach().clone() + torch.randn((1,self.state_d))@( self.P_sqrt.t() ) #Used to generate measurements for D.A. at each time-step
        Particles = self.InitialVector + torch.randn((self.ParticlesNum,self.state_d))@( self.P_sqrt.t() )

        return Particles, NoisyState, TrueState

    def init_model(self):
        if self.Warmstart:
            if self.LoadWarmstart and os.path.isfile(os.path.dirname(os.path.abspath(__file__))+'/Models/model_warmstart_test.pth'):
                ScoreModelInit = torch.load(os.path.dirname(os.path.abspath(__file__))+'/Models/model_warmstart_test.pth',weights_only=False)
            else:
                ScoreModelInit = ConditionalScoreNetwork1D(state_d=self.state_d,observation_d=self.observation_d,temb_d=4)
                
                WarmstartParticles = self.WarmstartSampleScale*torch.rand(self.WarmstartParticlesNum,self.state_d) + self.WarmstartSampleOffset
                TimeBatchNum = self.WarmstartParticlesNum//self.NSteps
                BatchNum = TimeBatchNum*self.NSteps 

                WarmstartParticlesFinal = torch.zeros((BatchNum,self.observation_d))
                WarmstartObservationsFinal = torch.zeros((BatchNum,self.state_d))
                for StepInd in range(self.NSteps): #If the vector field is not sationary, we have to train using different times
                    WarmstartParticlesLocal = WarmstartParticles[StepInd*TimeBatchNum:(StepInd+1)*TimeBatchNum,:]
                    _, WarmstartParticlesLocalTmp = self.ApproxPredictor.forward(self.t0+self.dt*StepInd,WarmstartParticlesLocal,AddNoise=True)
                    WarmstartObservationsLocal = self.Observer.forward(WarmstartParticlesLocalTmp,AddNoise=True)
                    _, WarmstartParticlesLocal = self.ApproxPredictor.forward(self.t0+self.dt*StepInd,WarmstartParticlesLocal)

                    WarmstartParticlesFinal[StepInd*TimeBatchNum:(StepInd+1)*TimeBatchNum,:] = WarmstartParticlesLocal
                    WarmstartObservationsFinal[StepInd*TimeBatchNum:(StepInd+1)*TimeBatchNum,:] = WarmstartObservationsLocal

                WarmstartParticlesFinal = WarmstartParticlesFinal[torch.randperm(BatchNum)] #shuffle
                WarmstartObservationsFinal = WarmstartObservationsFinal[torch.randperm(BatchNum)] #shuffle

                training_data = ConditionalDiffusionDataset1D(WarmstartParticlesFinal[0:self.WarmstartTrainSplit,...],WarmstartObservationsFinal[0:self.WarmstartTrainSplit,...])
                test_data = ConditionalDiffusionDataset1D(WarmstartParticlesFinal[self.WarmstartTrainSplit:,...],WarmstartObservationsFinal[self.WarmstartTrainSplit:,...])

                Score = ScoreMatching(self.Params,ScoreModelInit,training_data,test_data,self.Loss,batch_size=self.WarmstartBatchSize,learning_rate=self.WarmstartLR,epochs=self.WarmstartEpochs)
                Score.Train()
                ScoreModelInit = Score.ScoreModel
                torch.save(ScoreModelInit, os.path.dirname(os.path.abspath(__file__))+'/Models/model_warmstart_test.pth')
        else:
            ScoreModelInit = ConditionalScoreNetwork1D(state_d=self.state_d,observation_d=self.observation_d,temb_d=4)
        return ScoreModelInit
    
    def plot_step(self,StepInd,ParticlesOld,ParticlesPredicted,ParticlesNew,TrueState,Measurement):
        Ellipsoid = GetCovEllipsoid(self.EmpiricalMean[StepInd,self.ShowSlice],self.EmpiricalCovariance[StepInd][np.ix_(self.ShowSlice, self.ShowSlice)],self.ShowConfidence)

        fig = plt.figure(1,figsize=(6,6))
        plt.draw()
        plt.pause(0.01)
        plt.clf()
        ax = fig.add_subplot(title=f"Step: {StepInd:>3d}")
    
        ax.scatter(*(ParticlesOld[:,self.ShowSlice].t()),color='red',marker=".",alpha=0.5,label='Old particles') #Before data assimilation
        ax.scatter(*(ParticlesPredicted[:,self.ShowSlice].t()),color='orange',marker=".",alpha=0.5,label='After predictor') #After predictor
        ax.scatter(*(ParticlesNew[:,self.ShowSlice].t()),color='green',marker=".",alpha=0.5,label='After D.A.') #After data assimilation

        if self.ShowMeasurement:
            ax.scatter(*Measurement[0,self.ShowSlice],color='blue',label='Measurement')
        ax.scatter(*TrueState[0,self.ShowSlice],color='black',label='True State')
        ax.scatter(*self.EmpiricalMean[StepInd,self.ShowSlice],color='magenta',label='Empirical Mean')

        PlotLegend = ax.legend()
        plt.gca().add_artist(PlotLegend)
        ax.add_artist(Ellipsoid)
        plt.legend([Ellipsoid], [str(100 * self.ShowConfidence) + "% " + "confidence"],loc="upper right")
        ax.axis('equal')
        ax.set_axisbelow(True)
        ax.grid(color='gray',linestyle='dashed',alpha=0.5)
        plt.draw()
        plt.pause(0.01)

    def plot_output(self):
        if self.ShowOutput:
            fig = plt.figure(2)
            ax = fig.add_subplot(projection='3d',title="Trajectory + Uncertainty Quantification")
            
            if self.ShowMeasurement:
                ax.plot(*(self.Measurements.t()),color="blue",label='Measurements')
            ax.plot(*(self.TrueTrajectory[:-1,:].t()),color="black",label='True Trajectory')
            ax.plot(*(self.EmpiricalMean.t()),color="magenta",label='Empirical Mean')

            for step in range(self.NSteps):
                Ellipsoid = GetCovEllipsoid(self.EmpiricalMean[step,:],self.EmpiricalCovariance[step,:,:],self.ShowConfidence)
                ax.plot_surface(*Ellipsoid,rstride=4,cstride=4,color='magenta',alpha=0.1)

            ax.axis('equal')
            PlotLegend = ax.legend()
            plt.gca().add_artist(PlotLegend)
            plt.show()