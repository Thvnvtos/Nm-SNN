import json, utils, torch, random
import numpy as np
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture


config_file_path = "config.json"
with open(config_file_path) as f:
	config = json.load(f)


data_path = config["data_path"]
T = config["T"]

dataset_train = DVS128Gesture(data_path, train=True, data_type='frame', split_by='number', frames_number=T)
dataset_test = DVS128Gesture(data_path, train=False, data_type='frame', split_by='number', frames_number=T)




def dataset_prepare(targets, train):
	
	dataset = DVS128Gesture(data_path, train=train, data_type='frame', split_by='number', frames_number=T)
	idx = [(x in targets) for x in dataset.targets]
	valid_idxs = []
	for i in range(len(idx)):
		if idx[i]: valid_idxs.append(i)
	return torch.utils.data.Subset(dataset, valid_idxs)


def dataset_prepare_fewshot(targets, k, train):
		
	dataset = DVS128Gesture(data_path, train=train, data_type='frame', split_by='number', frames_number=T)
	
	valid_idxs = []
	for t in targets:
		idx = [(x == t) for x in dataset.targets]
		valid_idxs_t = []
		for i in range(len(idx)):
			if idx[i]: valid_idxs_t.append(i)

		k_idx = torch.randperm(len(valid_idxs_t))[:k].numpy().tolist()
		valid_idxs += [valid_idxs_t[i] for i in k_idx]
	return torch.utils.data.Subset(dataset, valid_idxs)