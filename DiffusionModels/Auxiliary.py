import torch
from matplotlib import patches
import numpy as np
from scipy.stats import chi2
import time

def ComputeMeanAndCov(Particles):
    Mean = torch.mean(Particles,dim=0)
    Cov = torch.cov(Particles.t())
    return Mean, Cov

def GetCovEllipsoid(Mean,Cov,Confidence):
    if len(Mean) == 2:
        #Mean = Mean.numpy()
        #Cov = Cov.numpy()
        L, V = np.linalg.eig(Cov)
        j_max = np.argmax(L)
        j_min = np.argmin(L)
        p = len(Mean) #DoF
        scale = chi2.isf(1-Confidence, p)
        ell = patches.Ellipse(
            (Mean[0], Mean[1]),
            2.0 * np.sqrt(scale * L[j_max]),
            2.0 * np.sqrt(scale * L[j_min]),
            angle=np.arctan2(V[1,j_max],V[0,j_max]) * 180 / np.pi,
            alpha=0.1,
            color="magenta",
        )

    elif len(Mean) == 3:
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)

        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))
        sphere = np.stack((x, y, z), axis=-1)[..., None]

        e, v = np.linalg.eig(Cov)

        p = len(Mean) #DoF
        scale = chi2.isf(1-Confidence, p)
        s = v @ np.diag(np.sqrt(scale*e)) @ v.T

        ell = (s @ sphere).squeeze(-1) + Mean.numpy()
        ell = ell.transpose(2, 0, 1)
    else:
        print('GetCovEllipsoid: unsupported data dimension.')
        ell = None

    return ell


def EnKF(Y,X0,A,h,t,std_Q,std_R):
    """
    Created on Mon Nov  4 14:21:47 2024

    @author: jarrah
    """

    #np.random.seed(0)
    # Y is AVG_SIM x N x dy
    # X0 is AVG_SIM x L x J

    AVG_SIM = X0.shape[0]
    L = X0.shape[1]
    J = X0.shape[2]
    
    N = Y.shape[1]
    dy = Y.shape[2]

    start_time = time.time()
    SAVE_X_EnKF =  np.zeros((AVG_SIM,N,J,L))
    
    for k in range(AVG_SIM):

        y = Y[k,]

        x_EnKF  = np.zeros((N,J,L))
        x_EnKF[0,] = X0[k,].T 
        
        SAVE_X_EnKF[k,0,:,:] = x_EnKF[0,]
        # EnKF & 3DVAR
        for i in range(N-1):

            # 1) Forecast step
            sai_EnKF = np.random.multivariate_normal(np.zeros(L),std_Q*std_Q * np.eye(L),J) #  J x L 
            x_hatEnKF = A(x_EnKF[i,].T,t[i]).T + sai_EnKF # J x L

            # 2) (Re‐)compute ensemble mean & deviations
            X_hat = x_hatEnKF.mean(axis=0,keepdims=True)
            a = (x_hatEnKF - X_hat)

            # 3) Project to obs space
            eta_EnKF = np.random.multivariate_normal(np.zeros(dy),std_R*std_R * np.eye(dy),J)  # J x dy 
            y_hatEnKF = h(x_hatEnKF.T).T + eta_EnKF # J x dy

            Y_hat = y_hatEnKF.mean(axis=0,keepdims=True)
            b = (y_hatEnKF - Y_hat)

            # 4) Cross‐covariances
            C_xy = 1/J * a.T@b #np.matmul(a.transpose(),b)/J
            C_yy = 1/J * b.T@b #np.matmul(b.transpose(),b)/J
            
            # 5) Kalman gain
            K = C_xy @ np.linalg.inv(C_yy + np.eye(dy)*1e-6)#gamma*gamma)
            x_EnKF[i+1,:,:] = x_hatEnKF + (K@ (y[i+1,:] - y_hatEnKF).T).T 
        
            SAVE_X_EnKF[k,i+1,:,:] = x_EnKF[i+1,]

    print("--- EnKF time : %s seconds ---" % (time.time() - start_time))
    return SAVE_X_EnKF.transpose(0,1,3,2)