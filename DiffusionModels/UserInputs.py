import torch

class Parameters():
    def __init__(self,T=1,dt=0.001,
                 mu=2,alpha=1,
                 gamma=lambda t: 12**(2*t),
                 beta=lambda t: 0.01 + t*(20-0.01),
                 VarianceMode='VP'):
        self.T = T          #Time horizon
        self.dt = dt        #Integration step  
        self.N = round(self.T/self.dt)  
        self.mu = mu        #Variance preserving parameter
        self.alpha = alpha  #Diffusion equation extra parameter
        self.gamma = gamma
        self.beta = beta
        self.VarianceMode = VarianceMode

    def Integrate(self,fun,t):
        #Trapezoid rule
        StepNum = 100
        Step = t/StepNum
        tSamples = t.shape[0]

        ValsFun = torch.zeros((tSamples,StepNum+1))
        for tInd in range(tSamples):
            ValsFun[tInd,:] = fun(torch.linspace(0,t[tInd].item(),StepNum+1))

        IntFun = Step*( ( (ValsFun[:,0]+ValsFun[:,-1])/2 + torch.sum(ValsFun[:,1:-1],dim=1) ).unsqueeze(1) )

        return IntFun
    
    def b_g(self,t):
        if self.VarianceMode == 'VP':
            IntBeta = self.Integrate(self.beta,t)

            bt = self.beta(t)
            gt = self.beta(t)*( ( 1 - torch.exp( -(self.mu/2)*IntBeta ) )**( (2/self.mu) - 1 ) )

        elif self.VarianceMode == 'VE':
            bt = torch.zeros_like(t)
            gt = self.gamma(t)

        return bt,gt
    
    def m_sigma(self,t):
        if self.VarianceMode == 'VP':
            IntBeta = self.Integrate(self.beta,t)

            mt = torch.exp( -0.5*IntBeta )
            sigmat = ( 1 - mt**self.mu )**(1/self.mu)

        elif self.VarianceMode == 'VE':
            IntGamma = self.Integrate(self.gamma,t)

            mt = torch.ones_like(t)
            sigmat = torch.sqrt(IntGamma)

        return mt,sigmat
    


