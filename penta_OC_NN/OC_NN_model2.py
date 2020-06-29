import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Model, self).__init__()
        
        self.hidden_size = hidden_size
        self.fc1 = nn.Sequential(nn.Linear(input_size, hidden_size),
                                 nn.Sigmoid())
        self.fc2 = nn.Linear(hidden_size, 1)
        
    def forward(self, data, r, nu):
        hidden = self.fc1(data)
        output = self.fc2(hidden)
        obj_term1 = 0.5*torch.norm(self.fc1[0].weight, p=2)
        obj_term2 = 0.5*torch.norm(self.fc2.weight, p=2)
        obj_term3 = (1/nu)*torch.mean(F.relu(r-output))
        obj_term4 = -1*r
        return output, obj_term1 + obj_term2 + obj_term3 + obj_term4