from matplotlib import pyplot as plt
# from torch_cluster import grid_cluster, knn_graph
from fast_pytorch_kmeans import KMeans
from torchvision import transforms, datasets
import torch
from tqdm import tqdm
import numpy as np
# from Legendre.train_mnist import NeuralNet

gpu = True
if gpu:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = 'cuda'
else:
    torch.set_default_tensor_type(torch.FloatTensor)
    device = 'cpu'

'''
removal techniques:
    iteratively remove most similar
    batch remove similar?
    different distance metrics
    remove separately
'''


sizes_of_k = [10, 25, 50, 100, 200, 400, 1000, 2500, 5000, 10000, 20000, 40000]
repeats = 100
max_iter = 10
remove_all = True

test_label = "remove_all {} {}x{} max{}".format(remove_all, repeats, sizes_of_k, max_iter)

def normalise_and_remove(list_of_values, final_size, p=2, random=True):
    if list_of_values.shape[0] < final_size:
        return list_of_values
    if random:
        indexes = np.random.random_integers(0, len(list_of_values), final_size)
        return full_all_to_net_cavex(list_of_values[indexes], final_size)
    mins = torch.min(list_of_values, dim=0)[0]
    ranges = torch.max(list_of_values - mins, dim=0)[0]
    list_of_values -= mins
    list_of_values /= ranges + (ranges == 0)
    dist = torch.cdist(list_of_values, list_of_values, p=2)
    length = dist.shape[0]
    dist = dist.reshape(length*length)
    largest_distances = torch.topk(dist, k=final_size)
    dist += (torch.eye(length).to(device) * torch.max(dist)).reshape(length*length)
    shortest_distances = torch.topk(dist, k=length-final_size, largest=False)
    return shortest_distances

def full_all_to_net_cavex(fa, k):
    nm = torch.reshape(fa[:, :(280*28)], [k, 10, 784]).transpose(0, 1)
    nc = torch.reshape(fa[:, (280*28):(280*28)+10], [k, 10]).transpose(0, 1)
    cm = torch.reshape(fa[:, (280*28)+10:(280*28)+(280*28)+10], [k, 10, 784]).transpose(0, 1)
    cc = torch.reshape(fa[:, (280*28)+(280*28)+10:], [k, 10]).transpose(0, 1)
    return nm, nc, cm, cc

def full_cavex_to_mc(fc, k):
    cvm = torch.reshape(fc[:, :(280*28)], [k, 10, 784]).transpose(0, 1)
    cvc = torch.reshape(fc[:, (280*28):(280*28)+10], [k, 10]).transpose(0, 1)
    return cvm, cvc

def piecewise_cavex(x, cave_m, cave_c, vex_m, vex_c, max_mindex=False):
    cy = []
    vy = []
    for cm, cc, vm, vc in zip(cave_m, cave_c, vex_m, vex_c):
        cx = torch.matmul(x, torch.transpose(cm, 0, 1))
        vx = torch.matmul(x, torch.transpose(vm, 0, 1))
        cy.append(cx + cc)
        vy.append(vx + vc)
    cy = torch.stack(cy)
    vy = torch.stack(vy)
    y_max = torch.max(vy, dim=2)[0]
    y_min = torch.min(cy, dim=2)[0]
    vex_cave = torch.transpose((y_min + y_max) / 2, 0, 1)
    return vex_cave

def piecewise_value(x, net_m, net_c, cavex_m, cavex_c, max_mindex=False):
    # if max_mindex:
    #     sy = []
    #     cavexy = []
    #     for sm, sc, cvm, cvc in zip(net_m, net_c, cavex_m, cavex_c):
    #         smx = torch.matmul(x, sm)
    #         cavexx = torch.matmul(x, cvm)
    #         sy.append(smx + sc)
    #         cavexy.append(cavexx + cvc)
    #     sy = torch.stack(sy)
    #     cavexy = torch.stack(cavexy)
    #     max_vex_max3 = torch.max(cavexy, dim=0)[1]
    #     # min_cave_min3 = torch.min(-cavexy, dim=0)[1]
    #     vex_sy = torch.stack([torch.stack(
    #         [sy[output_i][j][i] for i, output_i in enumerate(idx)]) for j, idx in enumerate(max_vex_max3)])
    #     # cave_sy = torch.stack([torch.stack(
    #     #     [sy[output_i][j][i] for i, output_i in enumerate(idx)]) for j, idx in enumerate(min_cave_min3)])
    #     # y = (vex_sy + cave_sy) / 2
    #     y = vex_sy
    # else:
    #     vexy = []
    #     cavey = []
    #     for sm, sc, cvm, cvc in zip(net_m, net_c, cavex_m, cavex_c):
    #         vexx = torch.matmul(x, sm + cvm)
    #         cavex = torch.matmul(x, sm - cvm)
    #         vexy.append(vexx + cvc + sc)
    #         cavey.append(cavex - cvc + sc)
    #     vexy = torch.stack(vexy)
    #     cavey = torch.stack(cavey)
    #     y_max = torch.max(vexy, dim=0)[0]
    #     y_min = torch.min(cavey, dim=0)[0]
    #     y = (y_min + y_max) / 2
    # return y
    sy = []
    vexy = []
    cavey = []
    cavexy = []
    for sm, sc, cvm, cvc in zip(net_m, net_c, cavex_m, cavex_c):
        smx = torch.matmul(x, torch.transpose(sm, 0, 1))
        vexx = torch.matmul(x, torch.transpose(sm + cvm, 0, 1))
        cavex = torch.matmul(x, torch.transpose(sm - cvm, 0, 1))
        cavexx = torch.matmul(x, torch.transpose(cvm, 0, 1))
        sy.append((smx + sc))
        vexy.append((vexx + cvc + sc))
        cavey.append((cavex - cvc + sc))
        cavexy.append(cavexx + cvc)
    sy = torch.stack(sy)
    vexy = torch.stack(vexy)
    cavey = torch.stack(cavey)
    cavexy = torch.stack(cavexy)
    y_max = torch.max(vexy, dim=2)[0]
    y_min = torch.min(cavey, dim=2)[0]
    # max_vex_max = torch.max(vexy, dim=2)[1]
    # min_cave_min = torch.min(cavey, dim=2)[1]
    # max_vex_max2 = torch.max(sy + cavexy, dim=2)[1]
    # min_cave_min2 = torch.min(sy - cavexy, dim=2)[1]
    max_vex_max3 = torch.max(cavexy, dim=2)
    min_cave_min3 = torch.min(-cavexy, dim=2)
    vex_sy = torch.stack([torch.stack(
        [sy[j][i][output_i] for i, output_i in enumerate(idx)]) for j, idx in enumerate(max_vex_max3[1])])
    cave_sy = torch.stack([torch.stack(
        [sy[j][i][output_i] for i, output_i in enumerate(idx)]) for j, idx in enumerate(min_cave_min3[1])])
    # if max_mindex:
    #     y = torch.transpose((vex_sy + cave_sy) / 2, 0, 1)
    # else:
    #     y = torch.transpose((y_min + y_max) / 2, 0, 1)

    vex_cave = torch.transpose((y_min + y_max) / 2, 0, 1)
    indexing = torch.transpose((vex_sy + cave_sy) / 2, 0, 1)
    vex_sub_vex = torch.transpose((y_max - max_vex_max3[0]), 0, 1)
    vex_add_cave = torch.transpose((y_max + min_cave_min3[0]), 0, 1)
    cave_add_vex = torch.transpose((y_min + max_vex_max3[0]), 0, 1)
    cave_sub_cave = torch.transpose((y_min - min_cave_min3[0]), 0, 1)
    y_max_cavexy = torch.stack([torch.stack(
        [cavexy[j][i][output_i] for i, output_i in enumerate(idx)]) for j, idx in enumerate(torch.max(vexy, dim=2)[1])])
    sub_indexed = torch.transpose((y_max - y_max_cavexy), 0, 1)
    cavexy_y_max = torch.stack([torch.stack(
        [vexy[j][i][output_i] for i, output_i in enumerate(idx)]) for j, idx in enumerate(max_vex_max3[1])])
    indexed_y_sub_vex = torch.transpose((cavexy_y_max - max_vex_max3[0]), 0, 1)

    return vex_cave, indexing, vex_sub_vex, vex_add_cave, cave_add_vex, cave_sub_cave, sub_indexed, indexed_y_sub_vex


net_file = 'mnist0.5 sigmoid hidden_size[200] test_acc[98.1]'

print("loading files")
# net_m = torch.load('data/net_m {}.pt'.format(net_file))
# net_c = torch.load('data/net_c {}.pt'.format(net_file))
# cavex_m = torch.load('data/cavex_m {}.pt'.format(net_file))
# cavex_c = torch.load('data/cavex_c {}.pt'.format(net_file))

# cave_m = torch.load('data/cave_m {}.pt'.format(net_file))
# cave_c = torch.load('data/cave_c {}.pt'.format(net_file))
# full_cave = torch.hstack([cave_m.transpose(0, 1).reshape([cave_m.shape[1], cave_m.shape[0] * cave_m.shape[2]]),
#                           cave_c.transpose(0, 1)])
# torch.save(full_cave, 'data/full_cave {}.pt'.format(net_file))

# vex_m = torch.load('data/vex_m {}.pt'.format(net_file))
# vex_c = torch.load('data/vex_c {}.pt'.format(net_file))
# full_vex = torch.hstack([vex_m.transpose(0, 1).reshape([vex_m.shape[1], vex_m.shape[0] * vex_m.shape[2]]),
#                          vex_c.transpose(0, 1)])
# torch.save(full_vex, 'data/full_vex {}.pt'.format(net_file))

# cave_m = net_m - cavex_m
# cave_c = net_c - cavex_c
# vex_m = net_m + cavex_m
# vex_c = net_c + cavex_c
#
# torch.save(cave_m, 'data/cave_m {}.pt'.format(net_file))
# torch.save(cave_c, 'data/cave_c {}.pt'.format(net_file))
# torch.save(vex_m, 'data/vex_m {}.pt'.format(net_file))
# torch.save(vex_c, 'data/vex_c {}.pt'.format(net_file))

# full_cavex = torch.hstack([cavex_m.transpose(0, 1).reshape([cavex_m.shape[1], cavex_m.shape[0] * cavex_m.shape[2]]),
#                            cavex_c.transpose(0, 1)])
# torch.save(full_cavex, 'data/full_cavex {}.pt'.format(net_file))

# full_net = torch.hstack([net_m.transpose(0, 1).reshape([net_m.shape[1], net_m.shape[0] * net_m.shape[2]]),
#                          net_c.transpose(0, 1)])
# torch.save(full_net, 'data/full_net {}.pt'.format(net_file))

# full_net = torch.load('data/full_net {}.pt'.format(net_file))
# full_cavex = torch.load('data/full_cavex {}.pt'.format(net_file))

# print("Stacking")
# full_all = torch.hstack([net_m.transpose(0, 1).reshape([net_m.shape[1], net_m.shape[0] * net_m.shape[2]]),
#                          net_c.transpose(0, 1),
#                          cavex_m.transpose(0, 1).reshape([net_m.shape[1], net_m.shape[0] * net_m.shape[2]]),
#                          cavex_c.transpose(0, 1)
#                          ])
# full_all = torch.hstack([full_net, full_cavex])
# torch.save(full_all, 'data/full_all {}.pt'.format(net_file))
# full_all = torch.load('data/full_all {}.pt'.format(net_file))

# full_cave = torch.load('data/full_cave {}.pt'.format(net_file))
# full_vex = torch.load('data/full_vex {}.pt'.format(net_file))

print("generating data")
batch_size = 128
c_size = 128
trainset = datasets.MNIST('', download=True, train=True, transform=transforms.ToTensor())
testset = datasets.MNIST('', download=True, train=False, transform=transforms.ToTensor())
c_loader = torch.utils.data.DataLoader(trainset, batch_size=c_size,
                                           shuffle=True, generator=torch.Generator(device=device))
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                           shuffle=True, generator=torch.Generator(device=device))
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=True, generator=torch.Generator(device=device))


print("K it means what?")
if remove_all:
    full_all = torch.load('data/full_all {}.pt'.format(net_file))
    results = {}
    metric_name = ['vex_cave', 'indexing', 'vex_sub_vex', 'vex_add_cave', 'cave_add_vex', 'cave_sub_cave',
                   'sub_indexed', 'indexed_y_sub_vex']
    for K in sizes_of_k:
        results[K] = {'vex_cave': [], 'indexing': [], 'vex_sub_vex': [], 'vex_add_cave': [], 'cave_add_vex': [],
                      'cave_sub_cave': [], 'sub_indexed': [], 'indexed_y_sub_vex': []}
        for r in range(repeats):
            print("Starting repeat {}/{} of size {}".format(r+1, repeats, K))
            net_m, net_c, cavex_m, cavex_c = normalise_and_remove(full_all[:100, :], 10)
            print("calculating testing accuracy")
            metrics = len(metric_name)
            with torch.no_grad():
                correct_out = [0 for i in range(metrics)]
                total = 0
                all_out = [[] for i in range(metrics)]
                all_mindex = []
                for images, labels in tqdm(test_loader):
                    images = images.reshape(-1, 784).to(torch.device(device)) - 0.5

                    various_out = piecewise_value(images, net_m, net_c, cavex_m, cavex_c)

                    for out, out_v in enumerate(various_out):
                        _, pred = torch.max(out_v, 1)
                        correct_out[out] += (pred == labels).sum().item()
                        all_out[out].append(out_v)

                    total += labels.size(0)

            for correct, name in zip(correct_out, metric_name):
                print('{} testing accuracy: {} %'.format(name, 100 * correct / total))
                results[K][name].append(100 * correct / total)
else:
    full_cave = torch.load('data/full_cave {}.pt'.format(net_file))
    full_vex = torch.load('data/full_vex {}.pt'.format(net_file))
    results = {}
    for K in sizes_of_k:
        results[K] = []
        for r in range(repeats):
            print("Starting repeat {}/{} of size {}".format(r+1, repeats, K))
            cave_means = KMeans(n_clusters=K, mode='euclidean', verbose=1, max_iter=max_iter)
            vex_means = KMeans(n_clusters=K, mode='euclidean', verbose=1, max_iter=max_iter)
            _ = cave_means.fit_predict(full_cave)
            _ = vex_means.fit_predict(full_vex)
            cave_centroids = cave_means.centroids
            vex_centroids = vex_means.centroids
            cave_m, cave_c = full_cavex_to_mc(cave_centroids, K)
            vex_m, vex_c = full_cavex_to_mc(vex_centroids, K)
            print("calculating testing accuracy")
            with torch.no_grad():
                correct_v = 0
                total = 0
                for images, labels in tqdm(test_loader):
                    images = images.reshape(-1, 784).to(torch.device(device)) - 0.5
                    out_v = piecewise_cavex(images, cave_m, cave_c, vex_m, vex_c)
                    _, pred = torch.max(out_v, 1)
                    correct_v += (pred == labels).sum().item()

                    total += labels.size(0)

            print('Model testing accuracy: {} %'.format(100 * correct_v / total))
            results[K].append(100 * correct_v / total)


torch.save(results, 'data/results {}.pt'.format(test_label))

# N, D = 10000, 2
# x = 0.7 * torch.randn(N, D, dtype=dtype, device=device_id) + 0.3

# plt.figure(figsize=(8, 8))
# plt.scatter(x[:, 0].cpu(), x[:, 1].cpu(), c=labels.cpu(), s=30000 / len(x), cmap="tab10")
# plt.scatter(kmeans.centroids[:, 0].cpu(), kmeans.centroids[:, 1].cpu(), c="black", s=50, alpha=0.8)
# # plt.axis([-2, 2, -2, 2])
# plt.tight_layout()
# plt.show()

print("Done")
