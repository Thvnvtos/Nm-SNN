import torch
import torch.nn as nn
import torch.nn.functional as F

import spikingjelly.clock_driven.neuron as sjNeuron
from spikingjelly.clock_driven import surrogate


# Adaptive LIF neuron based on SpikingJelly
class ALIFNode(sjNeuron.BaseNode):
	def __init__(self, n, tau: float = 2, v_threshold_base=1.0, v_reset=0.0, surrogate_function=surrogate.Sigmoid(), detach_reset=False, monitor_state=False):
		
		# n is the number of unique neurons in the layer 
		self.n = n

		# after every T timesteps on one batch of data, every neuron returns to the base threshold value
		self.v_threshold_base = v_threshold_base
		self.tau = tau

		super().__init__(v_threshold_base, v_reset, surrogate_function, detach_reset)

	def reset_Athreshold(self, batch_size, device):
		'''
			Resets or inits the Adaptive threshold in the beginning of every pass,
			uses the batch_size to creat n*batch_size unique neurons, and device to adapt to the network's device 
			(can't do it automatically with .to(device) since the threshold is not a torch module parameter)

			another solution would be to override reset() of SpikingJelly, but in that case the batch_size and device should be given
			in the constructor, because batch size can vary when the examples remaining are less than the size, this solution is better.
			

			==> ( batch_size, n ) Torch tensor on the same device as the network, containing v_threshold_base value
		'''

		self.v_threshold = (torch.ones((batch_size,)+self.n) * self.v_threshold_base).to(device) 

	# From SpikingJelly
	def neuronal_charge(self, dv: torch.Tensor):
		if self.v_reset is None:
			self.v = self.v + (dv - self.v) / self.tau
		else:
			if isinstance(self.v_reset, float) and self.v_reset == 0.:
				self.v = self.v + (dv - self.v) / self.tau
			else:
				self.v = self.v + (dv - (self.v - self.v_reset)) / self.tau

	def extra_repr(self):
		return f"n={self.n},tau={self.tau}, v_threshold_base={self.v_threshold_base}, v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}"



# Adaptive IF neuron based on SpikingJelly
class AIFNode(sjNeuron.BaseNode):
	def __init__(self, n, v_threshold_base=1.0, v_reset=0.0, surrogate_function=surrogate.Sigmoid(), detach_reset=False, monitor_state=False):
		
		# n is the number of unique neurons in the layer 
		self.n = n

		# after every T timesteps on one batch of data, every neuron returns to the base threshold value
		self.v_threshold_base = v_threshold_base

		super().__init__(v_threshold_base, v_reset, surrogate_function, detach_reset)

	def reset_Athreshold(self, batch_size, device):
		'''
			Resets or inits the Adaptive threshold in the beginning of every pass,
			uses the batch_size to creat n*batch_size unique neurons, and device to adapt to the network's device 
			(can't do it automatically with .to(device) since the threshold is not a torch module parameter)

			another solution would be to override reset() of SpikingJelly, but in that case the batch_size and device should be given
			in the constructor, because batch size can vary when the examples remaining are less than the size, this solution is better.
			

			==> ( batch_size, n ) Torch tensor on the same device as the network, containing v_threshold_base value
		'''

		self.v_threshold = (torch.ones((batch_size,)+self.n) * self.v_threshold_base).to(device) 

	# From SpikingJelly
	def neuronal_charge(self, dv: torch.Tensor):
		self.v += dv

	
	def extra_repr(self):
		return f"n={self.n}, v_threshold_base={self.v_threshold_base}, v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}"