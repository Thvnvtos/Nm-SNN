import torch.nn as nn

import spikingjelly.clock_driven.neuron as sjNeuron
from spikingjelly.clock_driven import surrogate

class FC_Nm(nn.Module):
	def __init__(self, n_in = 1024, n_fc = [256, 256]):
		super().__init__()

		self.fc1 = nn.Sequential(
			nn.Linear(n_in, n_fc[0]),
			sjNeuron.LIFNode(tau=2.0, v_threshold=0.1, surrogate_function=surrogate.ATan(), detach_reset=True))	

		self.fc2 = nn.Sequential(
			nn.Linear(n_fc[0], n_fc[1]),
			sjNeuron.LIFNode(tau=2.0, v_threshold=0.1, surrogate_function=surrogate.ATan(), detach_reset=True))

		self.output = nn.Sequential(
			nn.Linear(n_fc[1], 256),
			nn.Tanh())

	def forward(self, x):
		return self.output(self.fc2(self.fc1(x)))