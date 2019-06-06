import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import weights_init, norm_col_init
from torch.autograd import Variable
EPS = 0.003

class CNN_encoder(nn.Module):
    def __init__(self, obs_shape, stack_frames):
        super(CNN_encoder, self).__init__()
        self.conv1 = nn.Conv2d(obs_shape[0], 32, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)

        dummy_state = Variable(torch.rand(stack_frames, obs_shape[0], obs_shape[1], obs_shape[2]))
        out = self.forward(dummy_state)
        self.outdim = out.size(-1)
        self.apply(weights_init)
        self.train()

    def forward(self, inputs):
        x = F.relu(self.maxp1(self.conv1(inputs)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        return x


class FC_encoder(nn.Module):
    def __init__(self,  obs_space, out_dim, stack_frames):
        super(FC_encoder, self).__init__()
        self.fc1 = nn.Linear(obs_space[0], out_dim)
        self.fc1.weight.data = norm_col_init(self.fc1.weight.data, 1)
        self.outdim = out_dim * stack_frames

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        return x


class ValueNet(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(input_dim+action_dim, 512)
        self.fc1.weight.data = norm_col_init(self.fc1.weight.data, 1)

        self.fc2 = nn.Linear(512, 512)
        self.fc2.weight.data = norm_col_init(self.fc2.weight.data, 1)

        self.fc3 = nn.Linear(512, 256)
        self.fc3.weight.data = norm_col_init(self.fc3.weight.data, 1)

        self.fc4 = nn.Linear(256, 1)
        self.fc4.weight.data.uniform_(-EPS, EPS)

    def forward(self, inputs, action):
        x = torch.cat((inputs, action), dim=-1)
        x = F.relu(self.fc1(x))
        # x = torch.cat((s1,a1),dim=1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.fc4(x)

        return value


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, head):
        super(Critic, self).__init__()
        self.head = head
        self.state_dim = state_dim

        if self.head is not None:
            input_dim = head.outdim
        else:
            input_dim = self.state_dim

        self.valueNet = ValueNet(input_dim, action_dim)

    def forward(self, state, action):
        if self.head is not None:
            x = self.head(state)
        else:
            x = state
        value = self.valueNet.forward(x, action)

        return value


class PolicyNet(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(PolicyNet, self).__init__()

        self.fc1 = nn.Linear(input_dim, 512)
        self.fc1.weight.data = norm_col_init(self.fc1.weight.data, 1)

        self.fc2 = nn.Linear(512, 256)
        self.fc2.weight.data = norm_col_init(self.fc2.weight.data, 1)

        self.fc3 = nn.Linear(256, action_dim)
        self.fc3.weight.data = norm_col_init(self.fc3.weight.data, 1)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))

        return action


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, head=None):
        super(Actor, self).__init__()
        self.head = head
        if head is not None:
            self.input_dim = head.outdim
        else:
            self.input_dim = state_dim

        self.policyNet = PolicyNet(self.input_dim, action_dim)

    def forward(self, state):
        if self.head is not None:
            x = self.head(state)
        else:
            x = state
        action = self.policyNet(x)
        return action


class DDPG(nn.Module):
    def __init__(self, obs_shape, action_dim, args):
        super(DDPG, self).__init__()
        self.head = None
        if 'fc' in args.model:
            self.head = FC_encoder(obs_shape, 128, args.stack_frames)
            print('Use FC Head')
        elif 'cnn' in args.model:
            self.head = CNN_encoder(obs_shape, args.stack_frames)
            print('Use CNN Head')

        self.actor = Actor(obs_shape[0]*args.stack_frames, action_dim, self.head)
        self.critic = Critic(obs_shape[0]*args.stack_frames, action_dim, self.head)

    def forward(self, state):
        action = self.actor(state)
        value = self.critic(state, action)
        return action, value
