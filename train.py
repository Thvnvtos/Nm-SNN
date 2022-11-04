import torch, copy
import torch.nn.functional as F
from spikingjelly.clock_driven import functional


def train(net, loader, optimizer, device, use_nm = True):

	net.train()
	correct_pred = 0
	for i, inp in enumerate(loader):
		x, label = inp[0].float().to(device), inp[1].to(device)

		optimizer.zero_grad()
		
		y = net(x, use_nm = use_nm)
		label = F.one_hot(label, 11).float()
		loss = F.mse_loss(y, label)
		correct_pred += (y.argmax(dim=1) == label.argmax(dim=1)).sum().item()
		
		loss.backward()
		optimizer.step()
		
		functional.reset_net(net)
		if use_nm: functional.reset_net(net.nm)

	acc = 100. * correct_pred / len(loader.dataset)
	return acc


def get_acc(net, loader, device, use_nm = True):
	
	net.eval()
	correct_pred, loss = 0, 0
	with torch.no_grad():
		for x, label in loader:
			x, label = x.float().to(device), label.to(device)
			
			y = net(x, use_nm = use_nm)
			label = F.one_hot(label, 11).float()
			loss += F.mse_loss(y, label)
			correct_pred += (y.argmax(dim=1) == label.argmax(dim=1)).sum().item()
			
			functional.reset_net(net)
			if use_nm: functional.reset_net(net.nm)

	acc = 100. * correct_pred / len(loader.dataset)
	return acc
