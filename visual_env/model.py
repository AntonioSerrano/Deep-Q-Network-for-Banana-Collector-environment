import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        # Same CNN architcture as in Mnih et al. (2015) in Nature's paper: https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
        # i.e. 3 conv layers (with no pooling or dropout layers) with 
        # 2 fully-connected layers at the end.
        # input size = torch.Size([1, 64, 84, 84, 3]) # 64 images per batch, 4 stacked images, 84 pixels wide, 84 pixels high
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4)
        # output width, height = (W + 2*P - F)/S + 1 = (84 + 2*0 - 8)/4 + 1 = 20 (where L = length of the input (in this case the image), P = padding, F = filter or kernel size (one dimension, and S = stride))
        # conv1 output size = torch.Size([64, 32, 20, 20])

        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        # output width, height = (20 + 2*0 - 4)/2 + 1 = 9
        # conv2 output size = torch.Size([64, 64, 9, 9])

        self.fc1 = nn.Linear(64*9*9, 512)
        # fc1 output size = torch.Size([64, 512])

        self.fc2 = nn.Linear(512, action_size)
        # fc2 output size = torch.Size([64, action_size=4])

    def forward(self, state):
        """Build a network that maps state -> action values."""
        #x = F.relu(self.fc1(state))
        #x = F.relu(self.fc2(x))
        #return self.fc3(x)

        # CNN forward pass:
        #print(state.shape) # check state shape torch.Size([64, 3, 84, 84])
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # flatten previous output for the fully-connected layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x
        