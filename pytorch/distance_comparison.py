#!/usr/bin/env python

import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import subprocess
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def AnchorFeature(x):
    # from position metrix to anchor feature
    batch = True
    if len(x.shape) == 2:
        x = x.reshape(1, *x.shape)
        batch = False
    b, n, d = x.shape
    d = torch.sqrt(torch.sum((x.reshape(b, 1, n, d) - x.reshape(b, n, 1, d)) ** 2, dim=3) + 1e-9)
    d = torch.sort(d, dim=2)[0]
    if not batch:
        d = d[0]
    return d

def AnchorFeatureDistance(x):
    # from distance matrix to anchor feature
    batch = True
    if len(x.shape) == 2:
        x = x.reshape(1, *x.shape)
        batch = False
    b, n, _ = x.shape
    d = torch.sort(x, dim=2)[0]
    if not batch:
        d = d[0]
    return d


def log_sinkhorn(lK):
    b, n, _ = lK.shape
    lv1 = torch.zeros(b, n, 1)
    lv2 = torch.zeros(b, n, 1)
    prv_gamma = torch.zeros(lK.shape)
    while True:
        tmp = lK + lv2.reshape(b, 1, n) - np.log(n)
        lv1 = - torch.logsumexp(tmp, 2).reshape(b, n, 1)
        tmp = lK.transpose(1, 2) + lv1.reshape(b, 1, n) - np.log(n)
        lv2 = - torch.logsumexp(tmp, 2).reshape(b, n, 1)
        gamma = torch.exp(lv1 + lK + lv2.reshape(b, 1, n))
        diff = torch.sqrt(((gamma - prv_gamma) ** 2).sum())
        norm = torch.sqrt((gamma ** 2).sum())
        prv_gamma = gamma.clone()
        if diff / norm < 1e-3:
            return gamma

def load(filename):
    a = []
    with open(filename) as f:
        f.readline()
        for r in f:
            a.append(list(map(float, r.split())))
    return np.array(a)
        
def main():
    torch.manual_seed(0)
    t = sys.argv[1]
    if t == 'AW':
        eps = float(sys.argv[2])
    x = torch.FloatTensor(load(sys.argv[-2]))[None, ...] # (batch, n, n)
    y = torch.FloatTensor(load(sys.argv[-1]))[None, ...]
    n = x.shape[1]
    xf = AnchorFeatureDistance(x)
    yf = AnchorFeatureDistance(y)
    if t == 'AW':
        C = torch.sum(torch.abs(xf.reshape(1, n, 1, n) - yf.reshape(1, 1, n, n)), dim=3) / (n * n * n)
        lK = -C / eps
        P = log_sinkhorn(lK)
        AW = torch.sum(P * C + eps * P * (torch.log(torch.max(P, torch.ones(P.shape) * 1e-18)) - 1), dim=[1, 2])
        loss = torch.mean(AW)
    elif t == 'AE':
        lossm = torch.mean(torch.abs(xf.reshape(1, n, 1, n) - yf.reshape(1, 1, n, n)))
        lossx = torch.mean(torch.abs(xf.reshape(1, n, 1, n) - xf.reshape(1, 1, n, n)))
        lossy = torch.mean(torch.abs(yf.reshape(1, n, 1, n) - yf.reshape(1, 1, n, n)))
        loss = lossm * 2 - lossx - lossy
    print(loss)


if __name__ == '__main__':
    main()
