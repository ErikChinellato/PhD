import torch
import numpy as np
import matplotlib.pyplot as plt
from Auxiliary import *
import scipy.io as sio


TrueTraj = torch.load("DiffusionModels/Data/TrueTraj.pt", weights_only=True)

Folders = ["DKF_Learned", "DKF_Accurate", "DiffModels_Inaccurate", "DiffModels_Accurate", "EnKF_Inaccurate", "EnKF_Accurate"]


TrajNum = 15
StatesNum = 100
std_Q_Vals = [0.,0.15,0.3]
std_R_Vals = [0.,1./3.,2./3.,1.]


RMSE = torch.zeros((len(Folders),len(std_Q_Vals),len(std_R_Vals),StatesNum))
for IndFolder in range(len(Folders)):
    for IndQ in range(len(std_Q_Vals)):
        for IndR in range(len(std_R_Vals)):
            ErrorSq = 0
            for TrajInd in range(TrajNum):
                if IndFolder > 1:
                    Traj = torch.load("DiffusionModels/Data/"+Folders[IndFolder]+"/EmpMean_"+str(IndQ)+"_"+str(IndR)+"_"+str(TrajInd)+".pt", weights_only=True)
                    '''
                    fig = plt.figure(2,figsize=(8,8))
                    ax = fig.add_subplot(projection='3d',title="Trajectory + Uncertainty Quantification (inaccurate predictor) - RMSE:")
                    ax.plot(*(Traj.t()),color="magenta",label='Empirical Mean')
                    ax.plot(*(TrueTraj.t()),color="black",label='True Trajectory')
                    PlotLegend = ax.legend()
                    ax.axis('equal')
                    plt.draw()
                    plt.show()
                    '''
                else:
                    Traj = torch.tensor(sio.loadmat("DiffusionModels/Data/"+Folders[IndFolder]+"/CollectedStates_"+str(IndQ)+"_"+str(IndR)+"_"+str(TrajInd)+".mat")['CollectedStates'][:,1:],dtype=torch.float).transpose(0,1)

                ErrorSq += torch.sum( (TrueTraj - Traj)**2, dim=1) #sum over the state coordinates

            ErrorSq /= TrajNum #mean over trajectories
            ErrorSq = torch.sqrt(ErrorSq) #take square root
            RMSE[IndFolder,IndQ,IndR,:] = ErrorSq


ShowIndFolders = [0,1,2,3,4,5]
ShowIndQ = 2
ShowIndR = 2

plt.figure(1)
plt.title(f"RMSE over time-step: std_Q = {std_Q_Vals[ShowIndQ]:>2f}, std_R = {std_R_Vals[ShowIndR]:>2f}")
for ShowIndFolder in ShowIndFolders:
    plt.plot(RMSE[ShowIndFolder,ShowIndQ,ShowIndR,:],label = Folders[ShowIndFolder])
plt.legend()
plt.xlabel("time-step")
plt.ylabel("RMSE")
#plt.ylim((0,4))
plt.draw()

ShowIndQ = 2
plt.figure(2)
plt.title(f"Mean RMSE over noise level: std_Q = {std_Q_Vals[ShowIndQ]:>2f}")
for ShowIndFolder in ShowIndFolders:
    plt.plot(std_R_Vals,torch.mean(RMSE[ShowIndFolder,ShowIndQ,:,:],dim=-1),label = Folders[ShowIndFolder])
plt.legend()
plt.xlabel("std_R")
plt.ylabel("Mean RMSE")
#plt.ylim((0,1.5))
plt.draw()

ShowIndR = 3
plt.figure(3)
plt.title(f"Mean RMSE over noise level: std_R = {std_R_Vals[ShowIndR]:>2f}")
for ShowIndFolder in ShowIndFolders:
    plt.plot(std_Q_Vals,torch.mean(RMSE[ShowIndFolder,:,ShowIndR,:],dim=-1),label = Folders[ShowIndFolder])
plt.legend()
plt.xlabel("std_Q")
plt.ylabel("Mean RMSE")
#plt.ylim((0,1.5))
plt.show()