import torch
from torch import nn
from torch.utils.data import DataLoader

from ScoreNetworkComponents import *


class ScoreMatching():
    def __init__(self,Params,ScoreModelInit,training_data,test_data,Loss,batch_size=256,learning_rate=5e-3,epochs=100):
        self.m_sigma = Params.m_sigma
        self.T = Params.T
        self.dt = Params.dt
        self.N = Params.N
        self.ScoreModel = ScoreModelInit
        self.batch_size = batch_size
        self.train_dataloader = DataLoader(training_data,batch_size=self.batch_size,shuffle=True)
        self.test_dataloader = DataLoader(test_data,batch_size=self.batch_size)
        self.Loss = Loss
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.ScoreModel.parameters(), lr=self.learning_rate, betas=(0.9,0.999), eps = 1e-8)
        self.epochs = epochs
        
    def TrainLoop(self):
        size = len(self.train_dataloader.dataset)
        # Set the model to training mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        self.ScoreModel.train()
         
        for batch, X in enumerate(self.train_dataloader):
            # Compute prediction and loss
            if isinstance(X,list):
                ZShape = X[0].shape
            else:
                ZShape = X.shape

            tSamples = ZShape[0]
            TargetChannelNum = len(ZShape)
            
            tLabels = torch.randint(0, self.N, (tSamples,))
            t = self.dt*tLabels.float().unsqueeze(-1)
            mt, sigmat = self.m_sigma(t)

            for _ in range(0,TargetChannelNum-2):
                mt = mt.unsqueeze(-1)
                sigmat = sigmat.unsqueeze(-1)
            
            Z = torch.randn(ZShape)

            if isinstance(X,list):
                X[0] = mt*X[0] + sigmat*Z
            else:
                X = mt*X + sigmat*Z

            pred = self.ScoreModel(X,tLabels)
            loss = self.Loss.ComputeLoss(pred,sigmat,Z)

            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ScoreModel.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            #torch.save(self.ScoreModel, 'model.pth')

            if batch % 2 == 0:
                loss, current = loss.item(), batch * self.batch_size + tSamples
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                torch.save(self.ScoreModel, 'model.pth')

    def TestLoop(self):
        # Set the model to evaluation mode - important for batch normalization and dropout layers
        self.ScoreModel.eval()
        num_batches = len(self.test_dataloader)
        test_loss = 0 

        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
        with torch.no_grad():
            for X in self.test_dataloader:
                # Compute prediction and loss
                if isinstance(X,list):
                    ZShape = X[0].shape
                else:
                    ZShape = X.shape

                tSamples = ZShape[0]
                TargetChannelNum = len(ZShape)

                tLabels = torch.randint(0, self.N, (tSamples,))
                t = self.dt*tLabels.float().unsqueeze(-1)
                mt, sigmat = self.m_sigma(t)

                for _ in range(0,TargetChannelNum-2):
                    mt = mt.unsqueeze(-1)
                    sigmat = sigmat.unsqueeze(-1)

                Z = torch.randn(ZShape)
                
                if isinstance(X,list):
                    X[0] = mt*X[0] + sigmat*Z
                else:
                    X = mt*X + sigmat*Z

                pred = self.ScoreModel(X,tLabels)
                test_loss += self.Loss.ComputeLoss(pred,sigmat,Z).item()

        test_loss /= num_batches
        print(f"Avg test loss: {test_loss:>8f} \n")

    def Train(self):
        for t in range(self.epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self.TrainLoop()
            self.TestLoop()
        print("Done!\n")


class DUMMYConditionalScoreNetwork1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=8,out_features=8)

    def forward(self,X,t):
        x = X[0].detach().clone() #Particles, dim = (Batch, Particles_dim)
        y = X[1].detach().clone() #Observations, dim = (Batch, Observations_dim)
        x = self.linear(x)
        return x


class ConditionalScoreNetwork1D(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self,X,t):
        pass


class ConditionalScoreNetwork2D(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self,X,t):
        pass


class UnconditionalScoreNetwork1D(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self,X,t):
        pass


class UnconditionalScoreNetwork2D(nn.Module):
    def __init__(self,C,W,H):
        super().__init__()
    

    def forward(self,X,t):
        pass


class ScoreLoss():
    def __init__(self):
        pass

    def ComputeLoss(self,pred,sigmat,Z):
        TargetChannelNum = len(pred.shape)
        for _ in range(0,TargetChannelNum-2):
            sigmat = sigmat.unsqueeze(-1)

        loss = torch.mean( torch.square(sigmat*pred + Z) )
        return loss




