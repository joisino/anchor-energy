import anchor
import numpy as np

def load(filename):
    a = []
    with open(filename) as f:
        f.readline()
        for r in f:
            a.append(list(map(float, r.split())))
    return np.array(a)

d1 = load('./samplein/1.txt')
d2 = load('./samplein/2.txt')

a1 = np.ones(len(d1)) / len(d1)
a2 = np.ones(len(d2)) / len(d2)

print('AE', '{:.6f}'.format(anchor.anchor_energy(d1, a1, d2, a2)))
print('RAE', '{:.6f}'.format(anchor.robust_anchor_energy(d1, a1, d2, a2)))
print('AW', '{:.6f}'.format(anchor.anchor_wasserstein(d1, a1, d2, a2, 1e-3)))
print('RAW', '{:.6f}'.format(anchor.robust_anchor_wasserstein(d1, a1, d2, a2, 1e-8)))
print('GW', '{:.6f}'.format(anchor.gromov_wasserstein(d1, a1, d2, a2, 100, 1)))

print('AEM', anchor.anchor_energy_matching(d1, a1, d2, a2))
print('AWM', anchor.anchor_wasserstein_matching(d1, a1, d2, a2, 1e-3))
print('GWM', anchor.gromov_wasserstein_matching(d1, a1, d2, a2, 100, 1))
