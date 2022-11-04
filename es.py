import utils
utils.set_seed(11)
import evaluate, network, torch, time, nm
import numpy as np


#import wandb
#wandb.login(key="25f19d79982fd7c29f092981a100f187f2c706b4") 

#wandb.init(project="NmEvo_DVS", resume = True)

net_dim = (1024, 512, 256, 256)
N = sum([net_dim[i]*net_dim[i+1] + net_dim[i+1] for i in range(len(net_dim)-1)])
print("=================== Starting ES ========================")
print("=> Nm number of parameters : ", N)

device = evaluate.device

curr_task_ind = [0, 2, 5, 9, 14, 20, 27, 35, 44, 54]
pre_task_ind = list(set([i for i in range(0, 55)]) - set(curr_task_ind))

tasks = [3, 4, 10] 

def F(x, n_tasks):
	nm = utils.vect2net(x).to(device)
	accs = 0
	for i in range(n_tasks):
		accs += evaluate.evaluate(nm, i, tasks=tasks[:n_tasks])
	return accs / n_tasks

total_iter = 5000
lim_cnt_best = 1000

sigma = 0.1
popsize = 20#int(4 + 3*np.log10(N))
mu = 5#popsize//3

w = np.array([np.log(mu + 0.5) - np.log(i) for i in range(1, mu + 1)])
w /= np.sum(w)

x =  2 * sigma * np.random.randn(N)
#x = np.load("x_curr.npy")

start_n_tasks = 3
ind_cl = ((start_n_tasks)*(start_n_tasks-1))//2

for n_tasks in range(start_n_tasks, 4):
	
	best = [0, 0, 0]
	best_pre_mean = np.array(best).mean()
	best_cnt = 0
	iter_cnt = 0

	while best_cnt < lim_cnt_best and iter_cnt < total_iter:
		
		best_cnt += 1
		iter_cnt += 1

		eps = []
		t_iter = 0
		
		Fs = []
		accs = []
		for i in range(popsize):
			t1 = time.time()
			
			eps.append(np.random.randn(N))
			accs.append(F(x + sigma * eps[i], n_tasks))
			Fs.append(accs[i].mean())

			t_iter += time.time() - t1

		idx_best = np.argsort(np.array(Fs))[-mu:][::-1]
		step = np.zeros(N)

		ind_best = idx_best[0]

		log = {}

		all_accs = [0 for i in range(n_tasks)]
		accs_CL = [0 for i in range(n_tasks)]
		accs_FS = [0 for i in range(n_tasks)]


		nm = utils.vect2net(x + sigma * eps[ind_best]).to(device)
		for i in range(n_tasks):
			temp_accs = evaluate.evaluate(nm, i, n_tasks)
			cnt = 0
			for t in range(0, n_tasks):
				for j in range(0, t+1):
					log[f"acc_{j}_{t}_{i}"] = temp_accs[cnt]/((1+0.05*t)*(1 + 2*((t-j)/n_tasks)))
					cnt += 1

			all_accs[i] = temp_accs.mean()
			accs_CL[i] = temp_accs[pre_task_ind[:ind_cl]].mean()
			accs_FS[i] = temp_accs[curr_task_ind[:n_tasks]].mean()
		
		if np.array(all_accs).mean() >= np.array(best).mean():
			best = all_accs
			xopt = x + sigma * eps[ind_best]
			np.save("x_opt.npy", xopt)

		best_mean = np.array(best).mean()
		log[f"Fitness Overall n_tasks={n_tasks}"] = best_mean
		log[f"Fitness curr n_tasks={n_tasks}"] = np.array(all_accs).mean()
		log[f"FS n_tasks={n_tasks}"] = np.array(accs_FS).mean()
		log[f"CL n_tasks={n_tasks}"] = np.array(accs_CL).mean()

		if best_mean > best_pre_mean:
			best_pre_mean = best_mean
			best_cnt = 0

		print(f"\n===========> n_tasks={n_tasks} | iteration={iter_cnt} | best_cnt={best_cnt} : Best Overall = {best_mean:.2f}% | Best curr = {np.array(all_accs).mean():.2f}% \nBest FS = {np.array(accs_FS).mean():.2f}% | Best CL = {np.array(accs_CL).mean():.2f}%")
		print(f"\nperms best overall = {best}")
		print(f"\nperms best curr = {all_accs}")
		print(f"perms CL   = {accs_CL}")
		print(f"perms FS   = {accs_FS}\n")
		#print(f"current perm#{perm_id} acc = {Fs[ind_best]:.2f}%  | best = {best[perm_id]:.2f} ")

		#print(f"Mean iter time = {t_iter/popsize:.3f} sec")
		print(f"Total Generation time = {t_iter/60:.3f} min \n")

		cnt = 0
		for i in range(0, n_tasks):
			for j in range(0, i+1):
				print(f"{accs[ind_best][cnt]/((1+0.05*i)*(1 + 2*((i-j)/n_tasks))):6.2f} | ", end='')
				cnt += 1
			print()

		print(f"===============================================================================================\n")

		#wandb.log(log)

		for i in range(mu):
			ind = idx_best[i]
			step += w[i] * eps[ind]
		x += sigma * step
		if iter_cnt%100 == 0:
			np.save(f"x_curr_{n_tasks}_{iter_cnt}.npy", x)
		np.save("x_curr.npy", x)

	ind_cl += n_tasks

#wandb.finish()