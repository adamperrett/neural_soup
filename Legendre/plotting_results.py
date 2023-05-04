from matplotlib import pyplot as plt
# from torch_cluster import grid_cluster, knn_graph
from fast_pytorch_kmeans import KMeans
from torchvision import transforms, datasets
import torch
from tqdm import tqdm
import numpy as np

conv_results = torch.load("data/results cluster_all True 10x[10, 25, 50, 100, 200, 400, 1000, 2500, 5000, 10000, 20000, 40000, 60000] max1000 mnist0.5 sigmoid cnnTrue hidden_size[200] test_acc[99.04].pt")
ff_results = torch.load("data/results cluster_all True 10x[10, 25, 50, 100, 200, 400, 1000, 2500, 5000, 10000, 20000, 40000, 60000] max1000 mnist0.5 sigmoid hidden_size[200] test_acc[98.1].pt")

# conv_ave = [{setting: np.mean(conv_results[k][setting]) for setting in conv_results[k]} for k in conv_results]
# ff_ave = [{setting: np.mean(ff_results[k][setting]) for setting in conv_results[k]} for k in ff_results]
# conv_std = [{setting: np.std(conv_results[k][setting]) for setting in conv_results[k]} for k in conv_results]
# ff_std = [{setting: np.std(ff_results[k][setting]) for setting in conv_results[k]} for k in ff_results]
setting = 'vex_cave'
conv_ave = [{setting: np.mean(conv_results[k][setting])} for k in conv_results]
ff_ave = [{setting: np.mean(ff_results[k][setting])} for k in ff_results]
conv_std = [{setting: np.std(conv_results[k][setting])} for k in conv_results]
ff_std = [{setting: np.std(ff_results[k][setting])} for k in ff_results]

k_settings = [k for k in ff_results]

plt.figure()
for setting in conv_ave[k_settings[0]]:
    mean = [k[setting] for k in conv_ave]
    std = [k[setting] for k in conv_std]
    plt.plot(k_settings, mean, label='c{}'.format(setting))
    stdev1 = np.array(mean) + np.array(std)
    stdev2 = np.array(mean) - np.array(std)
    plt.fill_between(k_settings, stdev1, stdev2, alpha=0.5)
for setting in ff_ave[k_settings[0]]:
    mean = [k[setting] for k in ff_ave]
    std = [k[setting] for k in ff_std]
    plt.plot(k_settings, mean, label='f{}'.format(setting))
    stdev1 = np.array(mean) + np.array(std)
    stdev2 = np.array(mean) - np.array(std)
    plt.fill_between(k_settings, stdev1, stdev2, alpha=0.5)
plt.legend(loc="lower right")
# plt.ylim([-20, 120])
plt.ylabel('Test accuracy')
plt.xlabel('Number of clusters')
plt.show()

print("done")
