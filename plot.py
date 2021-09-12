#!/usr/bin/env python3

import matplotlib.pyplot as plt

from mc3g import plot


vr = 0.66       # vaccine rate
ve = 0.66       # US CDC data
p = 1e-2        # prevalence
N = 200         # number of participants
sens = 0.8      # test sensitivity
spec = 0.97     # test specifity
runs = 10_000   # number of Monte Carlo runs

f = plot(vr, ve, p, N, sens, spec, runs)
f.suptitle(f'{N:d} Personen, Prävalenz {p*100:.2f}%, \nImpfquote von {vr*100:.0f}%, Impfwirksamkeit von {ve*100:.0f}%')
f.subplots_adjust(top=0.88)
f.show()
plt.show()
