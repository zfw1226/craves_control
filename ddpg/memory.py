import numpy as np
import random
from collections import deque


class MemoryBuffer:
	def __init__(self, size):
		self.buffer = deque(maxlen=size)
		self.maxSize = size
		self.len = 0

	def sample(self, count):
		"""
		samples a random batch from the replay memory buffer
		:param count: batch size
		:return: batch (numpy array)
		"""
		batch = []
		count = min(count, self.len)
		batch = random.sample(self.buffer, count)

		s_arr = np.float32([arr[0] for arr in batch])
		a_arr = np.float32([arr[1] for arr in batch])
		r_arr = np.float32([arr[2] for arr in batch])
		s1_arr = np.float32([arr[3] for arr in batch])

		return s_arr, a_arr, r_arr, s1_arr

	def len(self):
		return self.len

	def add(self, s, a, r, s1):
		"""
		adds a particular transaction in the memory buffer
		:param s: current state
		:param a: action taken
		:param r: reward received
		:param s1: next state
		:return:
		"""
		transition = (s,a,r,s1)
		self.len += 1
		if self.len > self.maxSize:
			self.len = self.maxSize
		self.buffer.append(transition)


class PrioMemoryBuffer(object):  # stored as ( s, a, r, s_ ) in SumTree
	epsilon = 0.01  # small amount to avoid zero priority
	alpha = 0.3  # [0~1] convert the importance of TD error to priority
	beta = 0.3  # importance-sampling, from initial value increasing to 1
	beta_increment_per_sampling = 0.001
	abs_err_upper = 10.  # clipped abs error
	memorySize = 0
	def __init__(self, capacity):
		self.tree = SumTree(capacity)

	def len(self):
		return min(self.memorySize, self.tree.capacity)

	def add(self, state, action, reward, newState):
		transition = [state, action, reward, newState]
		max_p = np.max(self.tree.tree[-self.tree.capacity:])
		if max_p == 0:
			max_p = self.abs_err_upper
		self.tree.add(max_p, transition)   # set the max p for new p
		self.memorySize += 1

	def sample(self, size):
		b_idx, b_memory, ISWeights = np.empty((size,), dtype=np.int32), np.zeros(size, dtype=object), np.empty((size, 1))
		pri_seg = self.tree.total_p / size  # priority segment
		self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

		state_batch = []
		action_batch = []
		reward_batch = []
		newState_batch = []

		if self.memorySize < self.tree.capacity:
			min_prob = np.min(self.tree.tree[-self.tree.capacity:-(self.tree.capacity-self.memorySize)]) / self.tree.total_p  # for later calculate ISweight
		else:
			min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p

		for i in range(size):
			a, b = pri_seg * i, pri_seg * (i + 1)
			v = np.random.uniform(a, b)
			idx, p, data = self.tree.get_leaf(v)
			prob = p / self.tree.total_p
			ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
			b_idx[i], b_memory[i] = idx, data

			state_batch.append(data[0])
			action_batch.append(data[1])
			reward_batch.append(data[2])
			newState_batch.append(data[3])
		return np.float32(state_batch), np.float32(action_batch), np.float32(reward_batch), np.float32(
			newState_batch), b_idx, np.float32(ISWeights)

	def update_tree(self, tree_idx, abs_errors):
		abs_errors += self.epsilon  # convert to abs and avoid 0
		clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
		ps = np.power(clipped_errors, self.alpha)
		for ti, p in zip(tree_idx, ps):
			self.tree.update(ti, p)


class SumTree(object):
	data_pointer = 0
	def __init__(self, capacity):
		self.capacity = capacity  # for all priority values
		self.tree = np.zeros(2 * capacity - 1)
		# [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
		self.data = np.zeros(capacity, dtype=object)  # for all transitions
		# [--------------data frame-------------]
        #             size: capacity

	def add(self, p, data):
		tree_idx = self.data_pointer + self.capacity - 1
		self.data[self.data_pointer] = data  # update data_frame
		self.update(tree_idx, p)  # update tree_frame

		self.data_pointer += 1
		if self.data_pointer >= self.capacity:  # replace when exceed the capacity
			self.data_pointer = 0

	def update(self, tree_idx, p):
		change = p - self.tree[tree_idx]
		self.tree[tree_idx] = p
		# then propagate the change through tree
		while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
			tree_idx = (tree_idx - 1) // 2
			self.tree[tree_idx] += change

	def get_leaf(self, v):
		"""
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
		parent_idx = 0
		while True:     # the while loop is faster than the method in the reference code
			cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
			cr_idx = cl_idx + 1
			if cl_idx >= len(self.tree):        # reach bottom, end search
				leaf_idx = parent_idx
				break
			else:       # downward search, always search for a higher priority node
				if v <= self.tree[cl_idx]:
					parent_idx = cl_idx
				else:
					v -= self.tree[cl_idx]
					parent_idx = cr_idx

		data_idx = leaf_idx - self.capacity + 1
		return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

	@property
	def total_p(self):
		return self.tree[0]