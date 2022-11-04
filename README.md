# Mitigating Catastrophic Forgetting in Spiking Neural Networks through Threshold Modulation

This repository contains the code used to produce the results from our paper [Mitigating Catastrophic Forgetting in Spiking Neural Networks through Threshold Modulation](https://openreview.net/forum?id=15SoThZmtU)

## Abstract

Artificial Neural Networks (ANNs) trained with Backpropagation and Stochastic Gradient Descent (SGD) suffer from the problem of Catastrophic Forgetting; when learning tasks sequentially, the ANN tends to abruptly forget previous knowledge upon being trained on a new task. On the other hand, biological neural networks do not suffer from this problem. Spiking Neural Networks (SNNs) are a class of Neural Networks that are closer to biological networks than ANNs and their intrinsic properties inspired from biology could alleviate the problem of Catastrophic Forgetting. In this paper, we investigate if the firing threshold mechanism of SNNs can be used to gate the activity of the network in order to reduce catastrophic forgetting. To this end, we evolve a Neuromodulatory Network that adapts the thresholds of an SNN depending on the spiking activity of the previous layer. Our experiments on different datasets show that the neurmodulated SNN can mitigate forgetting significantly with respect to a fixed threshold SNN. We also show that the evolved Neuromodulatory Network can generalize to multiple new scenarios and analyze its behavior.

## Requirements
- PyTorch
- SpikingJelly: which can be installed with:
```
pip install spikingjelly
```

## Usage Instructions
- DVS128 Gesture dataset can be downloaded from https://research.ibm.com/interactive/dvsgesture/
- The DVS128Gesture.zip should be unzipped to the path specified in config.json file as {data_directory_path}/DVS128Gesture
- Currently changing the settings of the experiment requires manually changing the evaluate_nmn.py.
- Running 
```
python evaluate_nmn.py 
```
outputs the results of the chosen Continual Learning scenario setting for the vanilla SNN, the Neuromodulated SNN and the EWC SNN.
- The best EWC lambda values for each scenario can be found on evaluate_nmn.py.
