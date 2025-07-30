import torch
from scipy.integrate import ode
import numpy as np


class Observers():
    def __init__(self,method,default_d,slice=None,transform=None,std_R=0.05,R_sqrt=None):
        self.method = method
        self.observation_d = default_d
        self.slice = slice
        self.transform = transform
        self.std_R = std_R
        self.R_sqrt = R_sqrt

    def projection(self,particles):
        if self.slice is not None:
            obs = particles[:,self.slice]
        else:
            obs = particles
        return obs

    def linear(self,particles):
        if self.transform is not None:
            obs = particles@self.transform
        else:
            obs = particles
        return obs

    def cubic(self,particles):
        if self.slice is not None:
            obs = particles[:,self.slice]**3
        else:
            obs = particles**3
        return obs
    
    def id(self,particles):
        obs = particles
        return obs
    

    def get_dimension(self):
        match self.method:
            case 'projection':
                self.observation_d = self.observation_d if self.slice is None else len(self.slice)
                if self.R_sqrt is None:
                    self.R_sqrt = self.std_R*torch.eye(self.observation_d)
                return self.observation_d
            case 'linear':
                self.observation_d = self.observation_d if self.transform is None else self.transform.shape[-1]
                if self.R_sqrt is None:
                    self.R_sqrt = self.std_R*torch.eye(self.observation_d)
                return self.observation_d
            case 'cubic':
                self.observation_d = self.observation_d if self.slice is None else len(self.slice)
                if self.R_sqrt is None:
                    self.R_sqrt = self.std_R*torch.eye(self.observation_d)
                return self.observation_d
            case 'id':
                #self.observation_d = self.observation_d
                if self.R_sqrt is None:
                    self.R_sqrt = self.std_R*torch.eye(self.observation_d)
                return self.observation_d
            case _:
                self.observation_d = self.observation_d if self.slice is None else len(self.slice)
                if self.R_sqrt is None:
                    self.R_sqrt = self.std_R*torch.eye(self.observation_d)
                return self.observation_d

    def noise_handler(self,obs,AddNoise):
        if AddNoise:
            out = obs + torch.randn_like(obs)@( self.R_sqrt.t() )
        else:
            out = obs
        return out

    def forward(self,particles,AddNoise=False):
        match self.method:
            case 'projection':
                obs = self.projection(particles)
            case 'linear':
                obs = self.linear(particles)
            case 'cubic':
                obs = self.cubic(particles)
            case 'id':
                obs = self.id(particles)
            case _:
                obs = self.projection(particles)
        return self.noise_handler(obs,AddNoise)

class Predictors():
    def __init__(self,method,state_d,dt,VectorFieldClass,std_Q=0.05,Q_sqrt=None):
        self.method = method
        self.state_d = state_d
        self.dt = dt
        self.VectorField = VectorFieldClass.VectorField
        self.std_Q = std_Q
        self.Q_sqrt = std_Q*torch.eye(state_d) if Q_sqrt is None else Q_sqrt

    def rk4(self,t,particles):
        k1 = self.VectorField(t,particles)
        k2 = self.VectorField(t + 0.5 * self.dt, particles + 0.5 * self.dt * k1)
        k3 = self.VectorField(t + 0.5 * self.dt, particles + 0.5 * self.dt * k2)
        k4 = self.VectorField(t + self.dt, particles + self.dt * k3)
        pred = particles + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return t+self.dt, pred

    def explicit_euler(self,t,particles):
        pred = particles + self.dt * self.VectorField(particles, t)
        return t+self.dt, pred

    def dopri5(self,t,particles): #solver only supports batches of size 1, must iterate over batch with for loop
        Batch = particles.shape[0]
        pred = torch.zeros_like(particles)

        solver = ode(self.VectorField).set_integrator("dopri5")

        for ParticleInd in range(Batch):
            solver.set_initial_value(particles[ParticleInd,:].numpy(), t)
            pred[ParticleInd,:] = torch.tensor(solver.integrate(t+self.dt),dtype=torch.float)
        if not solver.successful():
            raise RuntimeError("Could not integrate")
        return t+self.dt, pred
    
    def id(self,t,particles):
        pred = particles
        return t+self.dt, pred
    

    def noise_handler(self,pred,AddNoise):
        if AddNoise:
            out = pred + torch.randn_like(pred)@( self.Q_sqrt.t() )
        else:
            out = pred
        return out

    def forward(self,t,particles,AddNoise=False):
        match self.method:
            case 'rk4':
                t, pred = self.rk4(t,particles)
            case 'explicit_euler':
                t, pred = self.explicit_euler(t,particles)
            case 'dopri5':
                t, pred = self.dopri5(t,particles)
            case 'id':
                t, pred = self.id(t,particles)
            case _:
                t, pred = self.rk4(t,particles)
        return t, self.noise_handler(pred,AddNoise)

class VectorField():
    def __init__(self):
        pass #Global methods and options to be inherited go here, not needed atm

class Lorenz63(VectorField):
    def __init__(self,sigma=10.0,rho=28.0,beta=8.0/3.0,eps=0,vel=[10,10,10]):
        super().__init__()
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.eps = eps
        self.vel = vel
    
    def VectorField(self,t,states):
        CheckNumpy = isinstance(states, np.ndarray)

        if CheckNumpy:
            states = torch.tensor(states,dtype=torch.float).unsqueeze(0)
            
        x = states[..., 0]
        y = states[..., 1]
        z = states[..., 2]

        dxdt = self.vel[0]*( self.sigma * (y - x) + self.eps*(x**3) )
        dydt = self.vel[1]*( x * (self.rho - z) - y )
        dzdt = self.vel[2]*( x * y - self.beta * z )

        out = torch.cat((dxdt[:,None], dydt[:,None], dzdt[:,None]), dim=1)

        if CheckNumpy:
            out = out.numpy()

        return out