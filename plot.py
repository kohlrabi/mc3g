#!/usr/bin/env python3

import matplotlib.pyplot as plt

from mc3g import plot


vr = 0.8       # vaccine rate
ve = 0.8       # US CDC data
p = 1e-2        # prevalence
N = 100       # number of participants
sens = 0.8      # test sensitivity
spec = 0.97     # test specifity
runs = 10_000   # number of Monte Carlo runs

f = plot(vr, ve, p, N, sens, spec, runs)
f.show()
plt.show()
