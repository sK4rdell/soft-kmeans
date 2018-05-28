import numpy as np
import matplotlib.pyplot as plt


def plot_k_means(x, r, k):

    random_colors = np.random.random((k, 3))
    colors = r * random_colors
    plt.scatter(x[:,0], x[:,1], c=colors)
    plt.show()


def initialize_centers(x, num_k):
    N, D = x.shape
    centers = np.zeros((num_k, D))
    used_idx = []
    for k in range(num_k):
        idx = np.random.choice(N)
        while idx in used_idx:
            idx = np.random.choice(N)
        used_idx.append(idx)
        centers[k] = x[idx]
    return centers

def update_centers(r, x, K):
    N, D = x.shape
    centers = np.zeros((num_k, D))
    for k in range(K):
        centers[k] = ( (x * r[:, k] ) / r[:, k].sum()).sum()
    return centers

def cost():
    pass

def cluster_responsibilities(centers, x, beta):
    N, _ = x.shape
    K, D = centers.shape
    R = np.zeros((N, K))

    for n in range(N):
        R[n] = np.exp(-beta * np.linalg.norm(centers - x[n], 2))

    R /= R.sum(axis=1, keepdims=True)

    return R

def soft_k_means(x, num_k, max_iters=20, beta=1.):
    centers = initialize_centers(x, num_k)
    prev_cost = 0
    for _ in range(max_iters):

        r = cluster_responsibilities(centers, x, beta)

        centers = update_centers(r, x, num_k)
        cost = cost(x, centers)
        if cost == prev_cost:
            break
        prev_cost = cost
        
    plot_k_means(x, r, num_k)




def main():
    dim = 2
    dist = 4
    std = 1
    mu0 = np.array([0,0])
    mu1 = np.array([dist, dist])
    mu2 = np.array([0, dist])
    # num samps per class
    Nc = 300
    x0 = np.random.randn(Nc, dim) * std + mu0
    x1 = np.random.randn(Nc, dim) * std + mu1
    x2 = np.random.randn(Nc, dim) * std + mu2
    x = np.concatenate((x0, x1, x2), axis=0)
    
    soft_k_means(x, num_k=3)
    

if __name__ == "__main__":
    main()