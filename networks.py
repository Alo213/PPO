import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FeedForwardNN(nn.module):
    def __init__(self, in_dim, out_dim):
        super(FeedForwardNN, self).__init__()

        #TODO Cambiar dimensiones
        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)
    
    def forward(self, obs):
        #Convertimos obs a tensores para ser manejados por PyTorch
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype = torch.float)
        
        #TODO Cambiar relu por otra funcion de activacion
        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(obs))
        output = self.layer3(activation2)

        return output

