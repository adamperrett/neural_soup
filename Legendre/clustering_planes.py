import time
import torch
from matplotlib import pyplot as plt

use_cuda = torch.cuda.is_available()
dtype = torch.float32 if use_cuda else torch.float64
device_id = "cuda:0" if use_cuda else "cpu"

def KMeans(x, K=10, Niter=10, verbose=True):
    """Implements Lloyd's algorithm for the Euclidean metric."""

    start = time.time()
    N, D = x.shape  # Number of samples, dimension of the ambient space

    c = x[:K, :].clone()  # Simplistic initialization for the centroids

    x_i = torch.tensor(x.view(N, 1, D))  # (N, 1, D) samples
    c_j = torch.tensor(c.view(1, K, D))  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):

        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c /= Ncl  # in-place division to compute the average

    if verbose:  # Fancy display -----------------------------------------------
        if use_cuda:
            torch.cuda.synchronize()
        end = time.time()
        print(
            f"K-means for the Euclidean metric with {N:,} points in dimension {D:,}, K = {K:,}:"
        )
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                Niter, end - start, Niter, (end - start) / Niter
            )
        )

    return cl, c

# N, D, K = 10000, 2, 50
#
# x = 0.7 * torch.randn(N, D, dtype=dtype, device=device_id) + 0.3
#
# cl, c = KMeans(x, K)
#
# plt.figure(figsize=(8, 8))
# plt.scatter(x[:, 0].cpu(), x[:, 1].cpu(), c=cl.cpu(), s=30000 / len(x), cmap="tab10")
# plt.scatter(c[:, 0].cpu(), c[:, 1].cpu(), c="black", s=50, alpha=0.8)
# plt.axis([-2, 2, -2, 2])
# plt.tight_layout()
# plt.show()

net_file = 'mnist0.5 sigmoid hidden_size[200] test_acc[98.1]'

print("loading files")
net_m = torch.load('data/net_m {}.pt'.format(net_file))
net_c = torch.load('data/net_c {}.pt'.format(net_file))
cavex_m = torch.load('data/cavex_m {}.pt'.format(net_file))
cavex_c = torch.load('data/cavex_c {}.pt'.format(net_file))

# full_cavex = torch.hstack([cavex_m.reshape([cavex_m.shape[0]*cavex_m.shape[1], cavex_m.shape[2]]),
#                            cavex_c.reshape([cavex_c.shape[0]*cavex_c.shape[1], 1])])
# full_net = torch.hstack([net_m.reshape([net_m.shape[0]*net_m.shape[1], net_m.shape[2]]),
#                          net_c.reshape([net_c.shape[0]*net_c.shape[1], 1])])
print("Stacking")
full_all = torch.hstack([net_m.reshape([net_m.shape[0]*net_m.shape[1], net_m.shape[2]]),
                         net_c.reshape([net_c.shape[0]*net_c.shape[1], 1]),
                         cavex_m.reshape([cavex_m.shape[0] * cavex_m.shape[1], cavex_m.shape[2]]),
                         cavex_c.reshape([cavex_c.shape[0] * cavex_c.shape[1], 1])
                         ])
# full_all = torch.hstack([full_cavex, full_net])
torch.save(full_all, 'data/full_all {}.pt'.format(net_file))
# full_all = torch.load('data/full_all {}.pt'.format(net_file))

print("K it means what?")
cl, c = KMeans(full_all, 10)

# torch.save(net_m, 'data/net_m {}.pt'.format(net_file))
# torch.save(net_c, 'data/net_c {}.pt'.format(net_file))
# torch.save(cavex_m, 'data/cavex_m {}.pt'.format(net_file))
# torch.save(cavex_c, 'data/cavex_c {}.pt'.format(net_file))



print("Done")
