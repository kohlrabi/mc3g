import numba
import numpy as np
import typing as tp


rand = np.random.random

@numba.jit(nopython=True, parallel=True, nogil=True)
def event(
        vax_rate : float = 0.8,
        vax_eff : float = 0.66,
        pos_frac : float = 0.01,
        test_sensitivity : float = 0.8,
        test_specifity : float = 0.97,
        test_vax : bool = False,
        test_unvax : bool = True,
        N : int = 1_000,
        runs : int = 1_000
        ) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Hold an event for `N` people with 3G rules, and determine:

        1. the number of infected people in attendance, 
        2. the number of infected people rejected by the rules, 
        3. the number of not-infected people rejected by the rules
    
    The event is simulated by Monte Carlo, and the number of runs can be chosen by the `runs` argument.
    
    To hold a 2G event, set `vax_rate` to 1.
    
    To hold a noG event, set `test_vax` and `test_unvax` to False

    The vaccine efficacy can take negative values to account for waning vaccine efficacy / ADE
    
    @param vax_rate         Vaccination rate of the event (default=0.8)
    @param vax_eff          Efficacy of the vaccine against infections (default=0.66)
    @param pos_frac         Fraction of positive people in the visitors / prevalance (default=0.01)
    @param test_sensitivity Test sensitivity of the rapid antigen test (default=0.8)
    @param test_specifity   Test specifity of the rapid antigen test (default=0.97)
    @param test_vax         Boolean flag whether to test vaccinated visitors
    @param test_unvax       Boolean flag whether to test unvaccinated visitors
    @param N                Number of people attending the event
    @param runs             Number of Monte Carlo runs
   
    @return                 Tuple of np.ndarrays containing
                                res_pos: Array of length `runs` with the number of infected people in attendance
                                res_rej_pos: Array of length `runs` with the number of infected people rejected by the rules
                                res_rej_neg: Array of length `runs` with the number of not infected people rejected by the rules
    """
    
    res_pos = np.empty(runs, dtype=np.uint32) # number of positives inside
    res_pos_v = np.empty(runs, dtype=np.uint32) # number of vaxed positives inside
    res_pos_u = np.empty(runs, dtype=np.uint32) # number of unvaxed positives inside
    res_num_v = np.empty(runs, dtype=np.uint32) # number of vaxed inside
    res_num_u = np.empty(runs, dtype=np.uint32) # number of unvaxed inside
    res_rej_neg = np.empty(runs, dtype=np.uint32) # number of positives rejected
    res_rej_pos = np.empty(runs, dtype=np.uint32) # number of negatives rejected
    
    for i in numba.prange(runs):
        # counters
        num = 0 # number of people inside
        pos = 0 # number of positives inside
        pos_v = 0
        pos_u = 0
        num_v = 0
        num_u = 0
        rej_pos = 0 # number of positives rejected
        rej_neg = 0 # number of negatives rejected
        vaxed = False

        while num < N:
            # roll an vaxxed or unvaxxed person as long as event not full
            if rand() < vax_rate:
                # vaxxed person
                vaxed = True
                is_positive = rand() < pos_frac * (1 - vax_eff)
                tested = test_vax
            else:
                # unvaxxed person
                vaxed = False
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
                if vaxed == True:
                    pos_v += is_positive
                    num_v += 1
                else:
                    pos_u += is_positive
                    num_u += 1
                num += 1
        
        # tally counts
        res_pos_v[i] = np.uint32(pos_v)
        res_pos_u[i] = np.uint32(pos_u)
        res_num_v[i] = np.uint32(num_v)
        res_num_u[i] = np.uint32(num_u)
        res_pos[i] = res_pos_v[i] + res_pos_u[i] 
        res_rej_pos[i] = np.uint32(rej_pos)
        res_rej_neg[i] = np.uint32(rej_neg)
        
    return res_pos, res_rej_pos, res_rej_neg, res_pos_v, res_pos_u, res_num_v, res_num_u

