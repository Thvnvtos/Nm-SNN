import torch, random
import numpy as np

import nm, network

def set_seed(seed):
	torch.manual_seed(seed)
	random.seed(seed)
	np.random.seed(seed)
	torch.cuda.manual_seed(0)
	torch.cuda.manual_seed_all(0)
	torch.backends.cudnn.enabled = False
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


def vect2net(x):
	net = nm.FC_Nm()
	d = net.state_dict()
	i = 0
	for name, W in net.named_parameters():
		if 'weight' in name:
			w_size = list(W.flatten().size())[0]
			d[name] = torch.Tensor(x[i:i+w_size].reshape(W.shape))
			i+=w_size
		elif 'bias' in name:
			w_size = list(W.flatten().size())[0]
			d[name] = torch.Tensor(x[i:i+w_size].reshape(W.shape))
			i+=w_size
	net.load_state_dict(d)
	return net

def vect2net_snn(x):
	net = network.SNN()
	d = net.state_dict()
	i = 0
	for name, W in net.named_parameters():
		if 'weight' in name:
			w_size = list(W.flatten().size())[0]
			d[name] = torch.Tensor(x[i:i+w_size].reshape(W.shape))
			i+=w_size
		elif 'bias' in name:
			w_size = list(W.flatten().size())[0]
			d[name] = torch.Tensor(x[i:i+w_size].reshape(W.shape))
			i+=w_size
	net.load_state_dict(d)
	return net


def net2vect(net):
	x = []
	for name, W in net.named_parameters():
		if 'weight' in name:
			x.append(W.cpu().detach().numpy().flatten())
		elif 'bias' in name:
			x.append(W.cpu().detach().numpy().flatten())
	return np.concatenate(x)