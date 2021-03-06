import torch
import torch.nn as nn
import torch.nn.functional as F

dueling = True

if dueling:
    class QNetwork(nn.Module):
        """Actor (Policy) Model."""

        def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
            """Initialize parameters and build model.
            Params
            ======
                state_size (int): Dimension of each state
                action_size (int): Dimension of each action
                seed (int): Random seed
                fc1_units (int): Number of nodes in first hidden layer
                fc2_units (int): Number of nodes in second hidden layer
            """
            super(QNetwork, self).__init__()
            self.seed = torch.manual_seed(seed)
 
            self.fc1 = nn.Linear(state_size, fc1_units)
            
            self.fc2_adv = nn.Linear(fc1_units, fc2_units)
            self.fc2_val = nn.Linear(fc1_units, fc2_units)
            
            self.fc3_adv = nn.Linear(fc2_units, action_size)
            self.fc3_val = nn.Linear(fc2_units, 1)

        def forward(self, state):
            """Build a network that maps state -> action values."""
            y = F.relu(self.fc1(state))

            adv = F.relu(self.fc2_adv(y))
            val = F.relu(self.fc2_val(y))

            adv = self.fc3_adv(adv)
            val = self.fc3_val(val)

            x = val + adv - torch.mean(adv, dim=1, keepdim=True)

            return x

else:
    class QNetwork(nn.Module):
        """Actor (Policy) Model."""

        def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
            """Initialize parameters and build model.
            Params
            ======
                state_size (int): Dimension of each state
                action_size (int): Dimension of each action
                seed (int): Random seed
                fc1_units (int): Number of nodes in first hidden layer
                fc2_units (int): Number of nodes in second hidden layer
            """
            super(QNetwork, self).__init__()
            self.seed = torch.manual_seed(seed)
            self.fc1 = nn.Linear(state_size, fc1_units)
            self.fc2 = nn.Linear(fc1_units, fc2_units)
            self.fc3 = nn.Linear(fc2_units, action_size)

        def forward(self, state):
            """Build a network that maps state -> action values."""
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            return self.fc3(x)