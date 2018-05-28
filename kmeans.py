import numpy as np
import matplotlib.pyplot as plt


def plot_k_means(x, r, k):

    random_colors = np.random.random((k, 3))
    colors = r.dot(random_colors)
    print(r[:20])
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

def update_centers(x, r, K):
    N, D = x.shape
    centers = np.zeros((K, D))
    for k in range(K):
        centers[k] = r[:, k].dot(x) / r[:, k].sum()
    return centers

def square_dist(a, b):
    return (a - b) ** 2

def cost_func(x, r, centers, K):
    
    cost = 0
    for k in range(K):
        norm = np.linalg.norm(x - centers[k], 2)
        cost += (norm * np.expand_dims(r[:, k], axis=1) ).sum()
    return cost


def cluster_responsibilities(centers, x, beta):
    N, _ = x.shape
    K, D = centers.shape
    R = np.zeros((N, K))

    for n in range(N):        
        R[n] = np.exp(-beta * np.linalg.norm(centers - x[n], 2, axis=1)) 
    R /= R.sum(axis=1, keepdims=True)

    return R

def soft_k_means(x, K, max_iters=20, beta=1.):
    centers = initialize_centers(x, K)
    prev_cost = 0
    for _ in range(max_iters):
        r = cluster_responsibilities(centers, x, beta)
        centers = update_centers(x, r, K)
        cost = cost_func(x, r, centers, K)
        if np.abs(cost - prev_cost) < 1e-5:
            break
        prev_cost = cost
        
    plot_k_means(x, r, K)


def generate_samples(std=1, dim=2, dist=4):
    mu0 = np.array([0,0])
    mu1 = np.array([dist, dist])
    mu2 = np.array([0, dist])
    # num samps per class
    Nc = 300
    x0 = np.random.randn(Nc, dim) * std + mu0
    x1 = np.random.randn(Nc, dim) * std + mu1
    x2 = np.random.randn(Nc, dim) * std + mu2
    x = np.concatenate((x0, x1, x2), axis=0)
    return x
    

def main():
    x = generate_samples()
    soft_k_means(x, K=3)
    

if __name__ == "__main__":
    main()