import utils
utils.set_seed(11)
import evaluate
import numpy as np
from io import StringIO

device = evaluate.device

import torch, json, copy, time, math, sys
import torch.optim as optim
import torch.nn.functional as F
import network, train, utils, dataset
import numpy as np
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

loader_0 = evaluate.loader
loader_test_0 = evaluate.loader_test

net_init = network.SNN().to(device)
net_init.load_state_dict(torch.load("net_pretrain.pth"))

for param in net_init.conv.parameters():
  param.requires_grad = False
for param in net_init.fc.parameters():
  param.requires_grad = False

nm = utils.vect2net(np.load("x_opt.npy")).to(device)


def ev(seed, nm, perm_i, tasks, n=n, batch_size=batch_size, lr=lr, use_nm = True, device = device, loader = loader_0, loader_test = loader_test_0, net_init = net_init):

    # We set all random seeds inside this method, so that we are guaranteed to have the same data generation etc (for every scenario)
    utils.set_seed(seed)
 
    n_tasks = len(tasks) 

    standard_stdout = sys.stdout
    outBuffer = StringIO()
    sys.stdout = outBuffer

    d = [dataset.dataset_prepare_fewshot([x], n, train=True) for x in tasks]
    loader = [torch.utils.data.DataLoader(d[i], batch_size, shuffle=False) for i in range(n_tasks)]

    d_test = [dataset.dataset_prepare_fewshot([x], n, train=False) for x in tasks]
    loader_test = [torch.utils.data.DataLoader(d_test[i], batch_size, shuffle=False) for i in range(n_tasks)]

    sys.stdout = standard_stdout

    net = copy.deepcopy(net_init)
    if use_nm: 
      net.nm = copy.deepcopy(nm)
      for param in net.nm.parameters():
        param.requires_grad = False

    accs = []
    accs_batches = [[] for i in range(n_tasks)]

    for i in range(perm_i, perm_i+n_tasks):

        with torch.no_grad():
            net.output[0].weight[tasks[i%n_tasks]] = copy.deepcopy(net_init.output[0].weight[tasks[i%n_tasks]])
        

        opt = optim.SGD(net.parameters(), lr = lr)
        
        net.train()
        correct_pred = 0
        for ii, inp in enumerate(loader[i%n_tasks]):
          x, label = inp[0].float().to(device), inp[1].to(device)
          opt.zero_grad()
          
          y = net(x, use_nm = use_nm)
          label = F.one_hot(label, 11).float()
          loss = F.mse_loss(y, label)
          correct_pred += (y.argmax(dim=1) == label.argmax(dim=1)).sum().item()
          
          loss.backward()
          opt.step()
          
          functional.reset_net(net)
          if use_nm: functional.reset_net(net.nm)


          #for j in range(perm_i, i+1):
          #  accs_batches[j - perm_i].append(train.get_acc(net, loader_test[j%n_tasks], device, use_nm))   

        for j in range(perm_i, i+1):
            accs.append(train.get_acc(net, loader_test[j%n_tasks], device, use_nm))

    return np.array(accs), accs_batches





def ev_ewc(ewc_lambda, seed, nm, perm_i, tasks, n=n, batch_size=batch_size, lr=lr, device = device, loader = loader_0, loader_test = loader_test_0, net_init = net_init):

    utils.set_seed(0)
 
    n_tasks = len(tasks) 

    standard_stdout = sys.stdout
    outBuffer = StringIO()
    sys.stdout = outBuffer

    d = [dataset.dataset_prepare_fewshot([x], n, train=True) for x in tasks]
    loader = [torch.utils.data.DataLoader(d[i], batch_size, shuffle=False) for i in range(n_tasks)]

    d_test = [dataset.dataset_prepare_fewshot([x], n, train=False) for x in tasks]
    loader_test = [torch.utils.data.DataLoader(d_test[i], batch_size, shuffle=False) for i in range(n_tasks)]

    sys.stdout = standard_stdout


    net = copy.deepcopy(net_init)

    accs = []
    accs_batches = [[] for i in range(n_tasks)]


    fisher_dict = {}
    optpar_dict = {}


    def on_task_update(task_id,opt):
      net.train()
      opt.zero_grad()
  
      # accumulating gradients
      for ii, inp in enumerate(loader[task_id%n_tasks]):
          x, label = inp[0].float().to(device), inp[1].to(device)

          y = net(x, use_nm = False)
          label = F.one_hot(label, 11).float()
          loss = F.mse_loss(y, label)
          loss.backward()

          functional.reset_net(net)

      fisher_dict[task_id] = {}
      optpar_dict[task_id] = {}

      # gradients accumulated can be used to calculate fisher
      for name, param in net.named_parameters():
        if param.requires_grad:
          optpar_dict[task_id][name] = param.data.clone()
          fisher_dict[task_id][name] = param.grad.data.clone().pow(2)


    def train_ewc(task_id, opt):
      net.train()

      for ii, inp in enumerate(loader[task_id%n_tasks]):
        x, label = inp[0].float().to(device), inp[1].to(device)

        opt.zero_grad()

        y = net(x, use_nm = False)
        label = F.one_hot(label, 11).float()
        loss = F.mse_loss(y, label)

        for task in range(perm_i, task_id):
          for name, param in net.named_parameters():
            if param.requires_grad:
              fisher = fisher_dict[task][name]
              optpar = optpar_dict[task][name]
              loss += (fisher * (optpar - param).pow(2)).sum() * ewc_lambda
        
        loss.backward()
        opt.step()

        functional.reset_net(net)


    for i in range(perm_i, perm_i+n_tasks):

        with torch.no_grad():
            net.output[0].weight[tasks[i%n_tasks]] = copy.deepcopy(net_init.output[0].weight[tasks[i%n_tasks]])
        
        opt = optim.SGD(net.parameters(), lr = lr)

        train_ewc(i, opt)
        on_task_update(i, opt)
        

        for j in range(perm_i, i+1):
          accs.append(train.get_acc(net, loader_test[j%n_tasks], device, False))

        '''
        net.eval()
        for j in range(perm_i, i+1):
          correct_pred = 0
          for ii, inp in enumerate(loader_test[j%n_tasks]):
            x, label = inp[0].float().to(device), inp[1].to(device)
            
            y = net(x, use_nm = use_nm)
            label = F.one_hot(label, 11).float()
            correct_pred += (y.argmax(dim=1) == label.argmax(dim=1)).sum().item()

            functional.reset_net(net)
            if use_nm: functional.reset_net(net.nm)
          acc = 100. * correct_pred / len(loader_test[j%n_tasks].dataset)
          accs.append(acc)
        '''

    return np.array(accs), accs_batches






    
def test_nmn(seeds, tasks, n, batch_size, use_nm=True, ewc = 0, nm=nm, lr=7e-2, show_perm_details = True, return_acc=False):
  acc = []
  accs_mean_overall = 0
  all_accs_nonm = []
  n_tasks = len(tasks)
  acc_batches_mean_nonm = [0 for i in range(n_tasks)]

  cnt_accs_nonm = {x:0 for x in range(0, 100, 5)}

  for perm_id in range(0,n_tasks):
    accs_mean = 0
    for seed in seeds:
      
      if ewc == 0:
        acc_curr, acc_batches = ev(seed, nm, perm_id, tasks=tasks, n=n, batch_size=batch_size, lr = lr, use_nm=use_nm, net_init=net_init)
      else:
        acc_curr, acc_batches = ev_ewc(ewc, seed, nm, perm_id, tasks=tasks, n=n, batch_size=batch_size, lr = lr, net_init=net_init)

      for th in range(0, 100, 5):
        for accc in acc_curr:
          if accc <= th:
            cnt_accs_nonm[th] += 1
      accs_mean += acc_curr
      accs_mean_overall += acc_curr
      all_accs_nonm.append(acc_curr)
      for i in range(n_tasks):
        acc_batches_mean_nonm[i] += np.array(acc_batches[i])
    
    if show_perm_details:
      cnt = 0
      for i in range(0, n_tasks):
        for j in range(0, i+1):
          print(f"{accs_mean[cnt]/len(seeds):6.2f} | ", end='')
          cnt += 1
        print()
      acc.append((accs_mean/len(seeds)).mean())
      print(f"\nPermutation {perm_id} mean ACC = {(accs_mean/len(seeds)).mean()}")
      print(f"=====================================================\n")

  for i in range(n_tasks):
    acc_batches_mean_nonm[i] /= (len(seeds)*n_tasks)

  cnt = 0
  print("\n===============================================================================\n")
  for i in range(0, n_tasks):
    for j in range(0, i+1):
      print(f"{accs_mean_overall[cnt]/(len(seeds)*n_tasks):6.2f} | ", end='')
      cnt += 1
    print()
  acc.append((accs_mean_overall/(len(seeds)*n_tasks)).mean())

  print("\n==> Total ACC mean = ", (accs_mean_overall/(len(seeds)*n_tasks)).mean())
  if return_acc: return (accs_mean_overall/(len(seeds)*n_tasks)).mean()