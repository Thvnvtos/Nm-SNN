import utils
utils.set_seed(11)
import network, dataset, train
import json, copy, time, math

import numpy as np
import torch
import torch.optim as optim

from spikingjelly.clock_driven import functional


config_file_path = "config.json"
with open(config_file_path) as f:
	config = json.load(f)

batch_size = config["batch_size"]

n = config["n"]
nf = config["nf"]
ks = config["ks"]
n_fc = config["n_fc"]
T = config["T"]
lr = config["lr"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


pre_classes = [0, 1, 2, 6, 8]

dataset_train = dataset.dataset_prepare(pre_classes, train=True)
dataset_test = dataset.dataset_prepare(pre_classes, train=False)

train_loader = torch.utils.data.DataLoader(dataset_train, 32, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_test, 32)

net = network.SNN().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

best = 0

for i in range(100): 
  acc = train.train(net, train_loader, optimizer, device, False)
  acc_train = train.get_acc(net, train_loader, device, False)
  acc_test = train.get_acc(net, test_loader, device, False)

  torch.save(net.state_dict(), "net_pretrain_last.pth")

  print(f"training acc = {acc_train} | validation acc = {acc_test}")
  if acc_test >= best:
    best = acc_test
    torch.save(net.state_dict(), "net_pretrain_temp.pth")

print("Best ACC = ", best)

net_init = network.SNN().to(device)
net_init.load_state_dict(torch.load("net_pretrain_temp.pth"))

torch.nn.init.normal_(net_init.output[0].weight, 0, 1e-3)


torch.save(net_init.state_dict(), "net_pretrain.pth")