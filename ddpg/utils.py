import numpy as np
import torch
import shutil


def soft_update(target, source, tau):
	"""
	Copies the parameters from source network (x) to target network (y) using the below update
	y = TAU*x + (1 - TAU)*y
	:param target: Target network (PyTorch)
	:param source: Source network (PyTorch)
	:return:
	"""
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(
			target_param.data * (1.0 - tau) + param.data * tau
		)


def hard_update(target, source):
	"""
	Copies the parameters from source network to target network
	:param target: Target network (PyTorch)
	:param source: Source network (PyTorch)
	:return:
	"""
	for target_param, param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_(param.data)


def save_training_checkpoint(state, is_best, episode_count):
	"""
	Saves the models, with all training parameters intact
	:param state:
	:param is_best:
	:param filename:
	:return:
	"""
	filename = str(episode_count) + 'checkpoint.path.rar'
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, 'model_best.pth.tar')




# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
	def __init__(self, action_dim, mu = 0, theta = 0.1, sigma = 0.2):
		self.action_dim = action_dim
		self.mu = mu
		self.theta = theta
		self.sigma = sigma
		self.X = np.ones(self.action_dim) * self.mu

	def reset(self):
		self.X = np.ones(self.action_dim) * self.mu

	def sample(self):
		dx = self.theta * (self.mu - self.X)
		dx = dx + self.sigma * np.random.randn(len(self.X))
		self.X = self.X + dx
		return self.X


def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		weight_shape = list(m.weight.data.size())
		fan_in = np.prod(weight_shape[1:4])
		fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
		w_bound = np.sqrt(6. / (fan_in + fan_out))
		m.weight.data.uniform_(-w_bound, w_bound)
		m.bias.data.fill_(0)
	elif classname.find('Linear') != -1:
		weight_shape = list(m.weight.data.size())
		fan_in = weight_shape[1]
		fan_out = weight_shape[0]
		w_bound = np.sqrt(6. / (fan_in + fan_out))
		m.weight.data.uniform_(-w_bound, w_bound)
		m.bias.data.fill_(0)


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
    return x