import torch

class IntegrateSDE():
    def __init__(self,Params):
        self.b_g = Params.b_g
        self.m_sigma = Params.m_sigma
        self.alpha = Params.alpha
        self.T = Params.T
        self.dt = Params.dt
        self.N = Params.N

    def SDECoeffs(self,t):
        bt,gt = self.b_g(t)

        xCoeff = bt*self.dt/2
        scoreCoeff = gt*( (1+self.alpha)*self.dt/2 )
        zCoeff = torch.sqrt(gt*self.alpha*self.dt)

        return xCoeff,scoreCoeff,zCoeff
    
    def EvolveSDEParticles(self,X,ScoreModel):
        ScoreModel.eval()
        with torch.no_grad():
            #We assume Particles.shape = (NumParticles, _)
            if isinstance(X,list):
                Particles = X[0].clone()
                Observations = X[1].clone()
                NumParticles = X[0].shape[0]
            else:
                Particles = X.clone()
                NumParticles = X.shape[0]

            TargetChannelNum = len(Particles.shape)
            AuxShape = list(Particles.shape)
            AuxShape.append(self.N+1)
            HistParticles = torch.zeros(AuxShape)
            HistParticles[...,0] = torch.clone(Particles)

            print(f"Integrating SDE\n-------------------------------")
            for TimeInd in range(0,self.N):
                tLabels = (self.N-TimeInd)*torch.ones((NumParticles,))
                t = (tLabels*self.dt).float().unsqueeze(-1)
                xCoeff,scoreCoeff,zCoeff = self.SDECoeffs(t)

                for _ in range(0,TargetChannelNum-2):
                    xCoeff = xCoeff.unsqueeze(-1)
                    scoreCoeff = scoreCoeff.unsqueeze(-1)
                    zCoeff = zCoeff.unsqueeze(-1)

                Z = torch.randn(Particles.shape)

                NewParticles = Particles + xCoeff*Particles + scoreCoeff*ScoreModel(X,t) + zCoeff*Z

                if isinstance(X,list):
                    X = [NewParticles.detach().clone(),Observations]
                else:
                    X = NewParticles.detach().clone()

                Particles = NewParticles.detach()
                HistParticles[...,TimeInd+1] = Particles

                if TimeInd % 10 == 0:
                    print(f"progress: step {TimeInd:>5d}/{self.N:>5d}")
        print("Done!")
        return HistParticles
    