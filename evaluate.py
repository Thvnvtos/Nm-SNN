import torch, json, copy, time, math
import torch.optim as optim
import network, dataset, train, utils
import numpy as np

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

#pre_classes = [0, 1, 2, 6, 8]
#Test = [5, 7, 9]

tasks = [3, 4, 10]
n_tasks = len(tasks)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

d = [dataset.dataset_prepare_fewshot([x], n, train=True) for x in tasks]
loader = [torch.utils.data.DataLoader(d[i], batch_size, shuffle=False) for i in range(n_tasks)]

d_test = [dataset.dataset_prepare_fewshot([x], n, train=False) for x in tasks]
loader_test = [torch.utils.data.DataLoader(d_test[i], batch_size, shuffle=False) for i in range(n_tasks)]


net_init = network.SNN().to(device)
net_init.load_state_dict(torch.load("net_pretrain.pth"))

for param in net_init.conv.parameters():
	param.requires_grad = False
for param in net_init.fc.parameters():
  param.requires_grad = False
  


def evaluate(nm, perm_i, use_nm = True, device = device, loader = loader, loader_test = loader_test, net_init = net_init, tasks=tasks):
  
  net = copy.deepcopy(net_init)
  if use_nm:
    net.nm = nm
    for param in net.nm.parameters():
      param.requires_grad = False
      
  n_tasks = len(tasks)
  accs = []

  for i in range(perm_i, perm_i+n_tasks):

    with torch.no_grad():
      net.output[0].weight[tasks[i%n_tasks]] = copy.deepcopy(net_init.output[0].weight[tasks[i%n_tasks]])

    opt = optim.SGD(net.parameters(), lr = lr)

    _ = train.train(net, loader[i%n_tasks], opt, device, use_nm)
    
    for j in range(perm_i, i+1):
      accs.append((1 + 0.05*(i-perm_i))* (1 + 2*((i-j)/n_tasks)) * train.get_acc(net, loader_test[j%n_tasks], device, use_nm))
      #accs.append(train.get_acc(net, loader_test[j%n_tasks], device, use_nm))

  return np.array(accs)