import torch
from matplotlib import patches
import numpy as np
from scipy.stats import chi2

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
            angle=np.arctan2(V[j_max, 1], V[j_max, 0]) * 180 / np.pi,
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