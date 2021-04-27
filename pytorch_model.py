########################################################################
# import python-library
########################################################################
# import 
import torch.nn as nn


########################################################################
# pytorch model
########################################################################
class AutoEncoder(nn.Module):
    
    def __init__(self, input_dim):
        super().__init__()
        
        # Encoder Layers
        self.encoder = nn.Sequential(
            
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, 8),
            nn.BatchNorm1d(8),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            
            nn.Linear(8, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, input_dim)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        output = self.decoder(x)
        return output