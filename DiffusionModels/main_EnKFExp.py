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

#Initial population setup
Batch = 1000

InitialVector = torch.tensor([-5.6866,-8.4929,17.8452])
state_d = InitialVector.shape[0]

std_P = 0.05    # Model noise
P_sqrt = std_P*torch.eye(state_d)

#D.A. integration params
NSteps = 100
dt = 0.001
t = np.arange(0.0, dt * NSteps, dt)
print(f"Time steps: {len(t)}")

std_Q_Vals = [0.,0.15,0.3]       # Predictor noise
std_R_Vals = [0.,1./3.,2./3.,1]  # Observer noise

for ParticleInd in range(15): 
    for IndQ in range(len(std_Q_Vals)):
        for IndR in range(len(std_R_Vals)):
            Particles = InitialVector + torch.randn((Batch,state_d))@( P_sqrt.t() )
            print(f"IndQ = {IndQ}")
            print(f"IndR = {IndR}")
            print(f"ParticleInd = {ParticleInd}")
            print("--------------------")
            #Setup predictors and observers
            std_Q = std_Q_Vals[IndQ] # Predictor noise
            std_R = std_R_Vals[IndR] # Observer noise

            TrueVectorField = Lorenz63(eps=1e-2)
            TruePredictor = Predictors(state_d=state_d,dt=dt,std_Q=std_Q,VectorFieldClass=TrueVectorField,method='dopri5')

            ApproxVectorField = Lorenz63(eps=1e-2)  # Change to eps=0 for inaccurate predictor
            ApproxPredictor = Predictors(state_d=state_d,dt=dt,std_Q=std_Q,VectorFieldClass=ApproxVectorField,method='rk4')

            Observer = Observers(default_d=state_d,std_R=std_R,method='id')
            observation_d = Observer.get_dimension()

            #Plot containers & related params
            EmpiricalMean_function = torch.zeros((NSteps,state_d))
            EmpiricalCovariance_function = torch.zeros((NSteps,state_d,state_d))
            EmpiricalMean_manual = torch.zeros((NSteps,state_d))
            EmpiricalCovariance_manual = torch.zeros((NSteps,state_d,state_d))
            TrueTrajectory = torch.load('DiffusionModels/Data/TrueTraj.pt')
            Confidence = 0.95
            Slice = [0,1] #x,y
            ShowSteps = True
            ShowOutput = True

            # True measurements
            Meas = torch.tensor(sio.loadmat("DiffusionModels/Data/Measurements/Meas_"+str(IndQ)+"_"+str(IndR)+".mat")['Meas'][:,1:],dtype=torch.float).transpose(0,1)

            # --------- EnKF DA ---------

            ############## METHOD 1: function call (lecture notes code) ##############

            #EnKF Functions
            # A is the predictor
            def A(x,t):
                # Convert numpy array x to torch tensor, shape (ensemble_size, state_dim)
                x_torch = torch.tensor(x.T, dtype=torch.float)
                # Get prediction without noise (it is added later in EnKF)
                _, xpred_torch = ApproxPredictor.forward(t,x_torch,AddNoise=False)
                # Convert back to numpy and transpose: (ensemble_size, obs_dim) -> (obs_dim, ensemble_size)
                xpred = xpred_torch.numpy().T
                return xpred
            
            # h is the observation rule
            def h(x):
                # Convert numpy array x to torch tensor, shape (ensemble_size, state_dim)
                x_torch = torch.tensor(x.T,dtype=torch.float)
                # Get observations without noise (it is added later in EnKF)
                y_torch = Observer.forward(x_torch,AddNoise=False)
                # Convert back to numpy and transpose: (ensemble_size, obs_dim) -> (obs_dim, ensemble_size)
                y = y_torch.numpy().T
                return y
            
            # EMBEDDINGS FOR THE FUNCTION
            Y = Meas.unsqueeze(0).numpy()           # Shape: (1, NSteps, obs_d)
            X0 = Particles.T.unsqueeze(0).numpy()   # Shape is (1, state_d, ensembles)

            # Call EnKF function
            X_EnKF = EnKF(Y,X0,A,h,t,std_Q,std_R)
            # Shape of X_EnKF: (1, time_steps, state_d, batch)

            # Assemble empirical mean and covariance
            for step in range(NSteps):
                particles_step = torch.tensor(X_EnKF[0, step, :, :]).t() # shape: (batch, state_d)
                EmpiricalMean_function[step, :], EmpiricalCovariance_function[step, :, :] = ComputeMeanAndCov(particles_step)


            ############## METHOD 2: manual call (adaptation of our own code) ##############
            for step in range(NSteps):
                #Begin Data Assimilation
                ParticlesOld = Particles.clone() #Save particles before data assimilation
                NewObservation = Meas[step,:].unsqueeze(0)

                # 1) Forecast via ApproxPredictor
                t, Particles = ApproxPredictor.forward(t,Particles,AddNoise=True) 
                ParticlesPredicted = Particles.clone() #Save particles after predictor

                # 2) (Re‐)compute ensemble mean & deviations
                x_mean = Particles.mean(dim=0,keepdim=True)              # (1, d)
                X_prime = Particles - x_mean                              # (B, d)

                # 3) Project to obs space
                Y_pred = Observer.forward(Particles,AddNoise=True)                   # (B, obs_d)
                y_mean = Y_pred.mean(dim=0,keepdim=True)                 # (1, obs_d)
                Y_prime = Y_pred - y_mean                                 # (B, obs_d)

                # 4) Cross‐covariances
                P_xy = (X_prime.T @ Y_prime) / Batch                   # (d, obs_d)
                P_yy = (Y_prime.T @ Y_prime) / Batch                    # (obs_d, obs_d)

                # 5) Kalman gain
                K = P_xy @ torch.linalg.inv(P_yy + torch.eye(observation_d)*1e-6)                         # (d, obs_d)

                # 6) Innovation
                innovation = NewObservation - Y_pred                       # (B, obs_d)

                # 7) Update particles
                Particles = Particles + innovation @ K.T                   # (B, d)
                ParticlesNew = Particles.clone()                        #Save particles after data assimilation
                
                # Assemble empirical mean and covariance
                EmpiricalMean_manual[step,:], EmpiricalCovariance_manual[step,:,:] = ComputeMeanAndCov(Particles)
                Ellipsoid = GetCovEllipsoid(EmpiricalMean_manual[step,Slice],EmpiricalCovariance_manual[step][np.ix_(Slice, Slice)],Confidence)

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
                    ax.scatter(*TrueTrajectory[step,Slice],color='black',label='True State')
                    ax.scatter(*EmpiricalMean_manual[step,Slice],color='magenta',label='Empirical Mean')

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
                ax.plot(*(EmpiricalMean_function.t()),color="magenta",marker="o",label='Empirical Mean (function)')
                ax.plot(*(EmpiricalMean_manual.t()),color="blue",marker="*",label='Empirical Mean (manual)')
                ax.plot(*(Meas.T),color="red",label='Measurements')

                ax.plot(*(TrueTrajectory.t()),color="black",label='True Trajectory',alpha=0.5)
                ax.plot(*(InitialVector), 'o', color="green", label="StartingPoint")
                PlotLegend = ax.legend()
                #for step in range(NSteps):
                #    Ellipsoid = GetCovEllipsoid(EmpiricalMean_manual[step,:],EmpiricalCovariance_manual[step,:,:],Confidence)
                #    ax.plot_surface(*Ellipsoid,rstride=4,cstride=4,color='magenta',alpha=0.1)
                ax.axis('equal')
                plt.draw()
                plt.pause(0.01)
                #plt.show()

            save = False
            if save:
                torch.save(EmpiricalMean_manual,'DiffusionModels/Data/EnKF_Inaccurate/EmpMean_'+str(IndQ)+'_'+str(IndR)+'_'+str(ParticleInd)+'.pt')
                torch.save(EmpiricalCovariance_manual,'DiffusionModels/Data/EnKF_Inaccurate/EmpCov_'+str(IndQ)+'_'+str(IndR)+'_'+str(ParticleInd)+'.pt')


