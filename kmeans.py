import numpy as np
import matplotlib.pyplot as plt


def main():
    dim = 2
    dist = 4
    std = 1
    mu0 = np.array([0,0])
    mu1 = np.array([dist, dist])
    mu2 = np.array([0, dist])
    # num samps per class
    Nc = 300

    s0 = np.random.randn(Nc, dim) * std + mu0
    s1 = np.random.randn(Nc, dim) * std + mu1
    s2 = np.random.randn(Nc, dim) * std + mu2

    s = np.concatenate((s0,s1,s2), axis=0)
    print(s.shape)
    

if __name__ == "__main__":
    main()