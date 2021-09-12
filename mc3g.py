#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import matplotlib
import numba
import numpy as np


plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['font.size'] = 15
rand = np.random.random

@numba.njit
def event(vax_rate : float = 0.8,
          vax_eff : float = 0.66,
          pos_frac : float = 0.01,
          test_sensitivity : float = 0.8,
          test_specifity : float = 0.97,
          test_vax : bool = False,
          test_unvax : bool = True,
          N : int = 1_000,
          bootstrap : int = 1_000
         ):
    """
    Hold an event with 3G rules, and determine the number of infected people in attendance, 
    the number of infected people rejected by the rules, 
    and the number of not-infected people rejected by the rules
    
    The event is simulated by Monte Carlo, and the number of runs can be chosen by the `bootstrap` argument.
    
    To hold a 2G event, set `vax_rate` to 1.
    
    To hold a noG event, set `test_vax` and `test_unvax` to False
    
    vax_rate: vaccination rate of the event (default=0.8)
    vax_eff: Efficacy of the vaccine against infections (default=0.66)
    pos_frac: Fraction of positive people in the visitors / prevalance (default=0.01)
    test_sensitivity: Test sensitivity of the rapid antigen test (default=0.8)
    test_specifity: Test specifity of the rapid antigen test (default=0.97)
    test_vax: Boolean flag whether to test vaccinated visitors
    test_unvax: Boolean flag whether to test unvaccinated visitors
    N: Number of people for the event
    bootstrap: Number of Monte Carlo runs
    
    Returns:
    res_pos: Array of length `bootstrap` with the number of infected people in attendance
    res_rej_pos: Array of length `bootstrap` with the number of infected people rejected by the rules
    res_rej_neg: Array of length `bootstrap` with the number of not infected people rejected by the rules
    """
    
    res_pos = np.empty(bootstrap, dtype=np.uint32) # number of positives inside
    res_rej_neg = np.empty(bootstrap, dtype=np.uint32) # number of positives rejected
    res_rej_pos = np.empty(bootstrap, dtype=np.uint32) # number of negatives rejected
    
    for i in numba.prange(bootstrap):
        # counters
        num = 0 # number of people inside
        pos = 0 # number of positives inside
        rej_pos = 0 # number of positives rejected
        rej_neg = 0 # number of negatives rejected

        while num < N:
            # roll an vaxxed or unvaxxed person as long as event not full
            if rand() < vax_rate:
                # vaxxed person
                is_positive = rand() < pos_frac * (1 - vax_eff)
                tested = test_vax
            else:
                # unvaxxed person
                is_positive = rand() < pos_frac
                tested = test_unvax
                
            # if person should be tested, do it
            test_positive = False
            if tested:
                if is_positive:
                    # true positive
                    if rand() < test_sensitivity:
                        test_positive = True
                        rej_pos += 1
                else:
                    # false positive
                    if rand() < 1 - test_specifity:
                        test_positive = True
                        rej_neg += 1

            if not test_positive:
                # allow to enter when not tested positive
                pos += is_positive
                num += 1
        
        # tally counts
        res_pos[i] = np.uint32(pos)
        res_rej_pos[i] = np.uint32(rej_pos)
        res_rej_neg[i] = np.uint32(rej_neg)
        
    return res_pos, res_rej_pos, res_rej_neg



def plot(vq=0.81, ve=0.66, p=1e-2, N=200):

    ev = event(vq, ve, p, 0.8, 0.97, False, True, N, 10_000)
    ev2 = event(vq, ve, p, 0.8, 0.97, True, True, N, 10_000)
    ev3 = event(1., ve, p, 0.8, 0.97, False, True, N, 10_000)
    ev4 = event(vq, ve, p, 0.8, 0.97, False, False, N, 10_000)

    f, ax = plt.subplots(3, 1, figsize=(10,10))

    means = ev[0].mean(), ev2[0].mean(), ev3[0].mean(), ev4[0].mean()
    means_rej = ev[1].mean(), ev2[1].mean(), ev[2].mean(), ev2[2].mean()

    for i in range(3):
        if i == 0:
            ax[i].hist(ev[i], bins=np.arange(ev[i].min()-4, ev[i].max()+4, 1, dtype=int), density=True, align='left', histtype='step', label=f'3G: {means[0]:.2f}');
            ax[i].hist(ev2[i], bins=np.arange(ev2[i].min()-4, ev2[i].max()+4, 1, dtype=int ), density=True, align='left', histtype='step', label=f'Alle testen: {means[1]:.2f}');
            ax[i].hist(ev3[i], bins=np.arange(ev3[i].min()-4, ev3[i].max()+4, 1, dtype=int), density=True, align='left', histtype='step', label=f'2G: {means[2]:.2f}');
            ax[i].hist(ev4[i], bins=np.arange(ev3[i].min()-4, ev3[i].max()+4, 1, dtype=int), density=True, align='left', histtype='step', label=f'NoG: {means[3]:.2f}');
        else:
            ax[i].hist(ev[i], bins=np.arange(ev[i].min()-4, ev[i].max()+4, 1, dtype=int), density=True, align='left', histtype='step', label=f'3G: {means_rej[(i-1)*2]:.2f}');
            ax[i].hist(ev2[i], bins=np.arange(ev2[i].min()-4, ev2[i].max()+4, 1, dtype=int), density=True, align='left', histtype='step', label=f'Alle testen: {means_rej[(i-1)*2+1]:.2f}');
            #ax[i].hist(ev2[i], bins=np.arange(ev2[i].min()-4, ev2[i].max()+4, 1, dtype=int ), density=True, align='left', histtype='step', label=f'2G: {means[1]:.2f}');
            #ax[i].hist(ev3[i], bins=np.arange(ev3[i].min()-4, ev3[i].max()+4, 1, dtype=int), density=True, align='left', histtype='step', label=f'NoG: {means[2]:.2f}');
        ax[i].set_xlabel('Anzahl Infizierter im Hörsaal')
        ax[i].set_ylabel('rel. Häufigkeit')
        #ax[i].set_title(f'{N} Studenten im Hörsaal bei Prävalenz von {p*100}%')
        #ax[i].set_title(f'{N} Studenten im Hörsaal bei Prävalenz von {p*100}%')
        ax[i]
        ax[i].legend()


    ax[0].set_title('Anzahl Infizierter im Hörsaal')
    ax[1].set_title('Anzahl abgelehnter Infizierter')
    ax[2].set_title('Anzahl abgelehnter nicht-Infizierter')

    ax[1].set_xlabel('Anzahl abgelehnter Infizierter')
    ax[2].set_xlabel('Anzahl abgelehnter nicht-Infizierter')


    plt.tight_layout()

    return f


