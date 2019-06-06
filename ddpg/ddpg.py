from __future__ import division
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import utils
from model import DDPG


class DDPGAgent:
    def __init__(self, obs_space, action_space, ram, writer, device, args):
        """
        :param obs_space: Dimensions of state (int)
        :param action_space: Dimension of action (int)
        :param ram: replay memory buffer object
        :return:
        """
        self.state_dim = obs_space.shape[0]
        self.action_dim = action_space.shape[0]
        self.action_high = action_space.high
        self.action_low = action_space.low
        self.ram = ram
        self.iter = 1
        self.steps = 0
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.decay_rate = args.decay_rate
        self.eps_start = args.eps_start
        self.eps_end = args.eps_end
        self.eps_decay = args.eps_decay
        self.start_step = args.start_learning
        self.device = device
        self.noise = utils.OrnsteinUhlenbeckActionNoise(self.action_dim)
        self.writer = writer
        self.args = args

        # init network
        target_net = DDPG(obs_space.shape, self.action_dim, args).to(device)
        learn_net = DDPG(obs_space.shape, self.action_dim, args).to(device)
        utils.hard_update(target_net, learn_net)
        self.AC = learn_net
        self.AC_T = target_net
        self.actor_optimizer = torch.optim.Adam(self.AC.actor.policyNet.parameters(), args.lr_a)
        self.critic_optimizer = torch.optim.Adam(self.AC.critic.parameters(), args.lr_c)
        self.actor = self.AC.actor
        self.target_actor = self.AC_T.actor
        self.critic = self.AC.critic
        self.target_critic = self.AC_T.critic

    def save_models(self, name):
        """
        saves the target actor and critic models
        """
        torch.save(self.AC_T.state_dict(), name + '.pt')

    def load_models(self, name, test=False):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        """
        saved_state = torch.load(
            name,
            map_location=lambda storage, loc: storage)
        if test:
            saved_state = {name: param for name, param in saved_state.items() if
                           'actor' in name}
            strict = False
        else:
            strict = True
        self.AC_T.load_state_dict(saved_state, strict=strict)
        utils.hard_update(self.AC, self.AC_T)
        print('Models loaded succesfully')

    def get_exploitation_action(self, state):
        """
        gets the action from target actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        state = Variable(torch.from_numpy(state)).to(self.device)
        action = self.target_actor.forward(state).detach().cpu()
        action = action.data.numpy()
        action = np.squeeze(action)
        action_rescale = self.rescale_action(action, self.action_low, self.action_high)
        return action, action_rescale

    def get_exploration_action(self, state):
        """
        gets the action from actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        self.steps += 1
        if self.decay_rate > self.eps_end and self.steps > self.start_step:
            self.decay_rate -= (self.eps_start - self.eps_end) / self.eps_decay

        state = Variable(torch.from_numpy(state)).to(self.device)
        action = self.actor.forward(state).detach().cpu()
        action = np.squeeze(action)
        noise = self.noise.sample()
        action_noise = (1 - self.decay_rate) * action.data.numpy() + self.decay_rate * noise
        action_noise = np.clip(action_noise, -1, 1)
        action_rescale = self.rescale_action(action_noise, self.action_low, self.action_high)
        self.writer.add_scalar('actions/decay_rate', self.decay_rate, self.steps)
        return action_noise, action_rescale

    def rescale_action(self, action, low, high):
        action_new = action*(high - low)/2.0 + (high + low)/2.0
        return action_new

    def optimize(self):
        """
        Samples a random batch from replay memory and performs optimization
        :return:
        """
        if self.args.pri:
            s1, a1, r1, s2, tree_idx, weights = self.ram.sample(self.batch_size)
            weights = torch.from_numpy(weights).to(self.device)
        else:
            s1, a1, r1, s2 = self.ram.sample(self.batch_size)
        s1 = Variable(torch.from_numpy(s1)).to(self.device)
        a1 = Variable(torch.from_numpy(a1)).to(self.device)
        r1 = Variable(torch.from_numpy(r1)).to(self.device)
        s2 = Variable(torch.from_numpy(s2)).to(self.device)

        # ---------------------- optimize critic ----------------------
        #  Use target actor exploitation policy here for loss evaluation
        a2 = self.target_actor.forward(s2).detach()
        next_val = torch.squeeze(self.target_critic.forward(s2, a2).detach())
        y_expected = r1 + self.gamma * next_val
        y_predicted = torch.squeeze(self.critic.forward(s1, a1))
        if self.args.pri:
            td_error = torch.abs(y_predicted - y_expected)
            loss_critic = torch.sum(weights * td_error ** 2)
            self.ram.update_tree(tree_idx, td_error.detach().cpu().numpy())
        else:
            loss_critic = F.mse_loss(y_predicted, y_expected)

        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # ---------------------- optimize actor ----------------------
        pred_a1 = self.actor.forward(s1)
        loss_actor = -1 * torch.mean(self.critic.forward(s1, pred_a1))
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        # logging
        self.writer.add_scalar('loss/critic', loss_critic, self.iter)
        self.writer.add_scalar('loss/actor', loss_actor, self.iter)
        utils.soft_update(self.AC_T, self.AC, self.tau)
        self.iter += 1