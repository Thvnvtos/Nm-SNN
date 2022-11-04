import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import spikingjelly.clock_driven.neuron as sjNeuron
from spikingjelly.clock_driven import surrogate

import neuron


class SNN(nn.Module):

	def __init__(self, channels: int = 64, v_threshold = 1.0):
		super().__init__()
		self.channels = channels
		conv = []
		conv.extend(SNN.conv3x3(2, channels))
		conv.append(nn.MaxPool2d(2, 2))
		self.threshold_data = []
		self.activity_data = []
		self.dthreshold_data = []
		self.feature_maps = []
		
		for i in range(4):
			conv.extend(SNN.conv3x3(channels, channels))
			conv.append(nn.MaxPool2d(2, 2))
		
		self.conv = nn.Sequential(*conv)
			
		self.fc = nn.Sequential(
				nn.Flatten(),
				nn.Linear(channels * 4 * 4, channels * 2 * 2, bias=False),
			)

		self.fc_sn = neuron.ALIFNode(n = (channels * 2 * 2,),tau=2.0,v_threshold_base=v_threshold, surrogate_function=surrogate.ATan(), detach_reset=True)

		self.output = nn.Sequential(
				nn.Linear(channels * 2 * 2, 11, bias=False),
				sjNeuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True)
				)


	@staticmethod
	def conv3x3(in_channels: int, out_channels):
		return [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
				sjNeuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True)]


	def forward(self, x, use_nm = True):

		x = x.permute(1, 0, 2, 3, 4)  # [N, T, 2, H, W] -> [T, N, 2, H, W]

		batch_size = x.size()[1]
		device = x.device

		self.reset_ANodes(batch_size, device)

		out_spikes_counter = 0
		for t in range(x.shape[0]):

			fm = self.conv(x[t])

			# FC 
			fc_pots = self.fc(fm)
			#self.feature_maps.append(fm)
			#self.activity_data.append(fc_pots)
			if use_nm: self.nm_fc(fm, batch_size, device)
			fc_spikes = self.fc_sn(fc_pots)

			out = self.output(fc_spikes)
			
			out_spikes_counter += out

		return out_spikes_counter / x.shape[0]


	def nm_fc(self, fm, batch_size, device):


		nm_in = fm.reshape((batch_size, self.channels * 4 * 4))

		dThreshold = self.nm(nm_in)#F.tanh(1e9*self.nm(nm_in))
		#self.dthreshold_data.append(dThreshold)
		self.fc_sn.v_threshold = F.relu(self.fc_sn.v_threshold + dThreshold) + 1e-9
		#self.threshold_data.append(self.fc_sn.v_threshold)

	def reset_ANodes(self, batch_size, device):
		
		for m in self.modules():
			if hasattr(m, 'reset_Athreshold'):
				m.reset_Athreshold(batch_size, device)