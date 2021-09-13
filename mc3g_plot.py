import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from mc3g import event


plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['font.size'] = 15


def plot(
        vr : float = 0.8,
        ve : float = 0.8,
        p : float = 1e-2,
        N : int = 100,
        sens : float = 0.8,
        spec : float = 0.97,
        runs : int = 10_000
        ) -> matplotlib.figure:

    ev = event(vr, ve, p, sens, spec, False, True, N, runs) # 3G
    ev2 = event(vr, ve, p, sens, spec, True, True, N, runs) # all tested
    ev3 = event(1., ve, p, sens, spec, False, True, N, runs) # 2G
    # calculate noG from 3G by adding the rejected infected
    ev4 = ev[0] + ev[1], np.zeros_like(ev[1]), np.zeros_like(ev[2]) # noG

    f, ax = plt.subplots(3, 1, figsize=(10,10))

    means = ev[0].mean(), ev2[0].mean(), ev3[0].mean(), ev4[0].mean()
    means_rej = ev[1].mean(), ev2[1].mean(), ev[2].mean(), ev2[2].mean()

    def do_hist(ax, ev, label):
        """
        common operations for all plots
        """
        ax.hist(ev, bins=np.arange(ev.min() - 0.5, ev.max() + 0.5, 1), density=True, histtype='step', label=label, lw=1.5)
        ax.set_ylabel('rel. Häufigkeit')
        ax.legend()


    for i in range(3):
        if i == 0:
            do_hist(ax[i], ev[i], f'3G: {means[0]:.2f}')
            do_hist(ax[i], ev2[i], f'Alle testen: {means[1]:.2f}')
            do_hist(ax[i], ev3[i], f'2G: {means[2]:.2f}')
            do_hist(ax[i], ev4[i], f'noG: {means[3]:.2f}')
            ax[i].set_xlabel('Anzahl Infizierter im Innenraum')
            ax[i].set_ylabel('rel. Häufigkeit')
        else:
            do_hist(ax[i], ev[i], f'3G: {means_rej[(i-1)*2]:.2f}')
            do_hist(ax[i], ev2[i], f'Alle testen: {means_rej[(i-1)*2]:.2f}')


    ax[0].set_title('Anzahl Infizierter im Innenraum')
    ax[1].set_title('Anzahl abgelehnter Infizierter')
    ax[2].set_title('Anzahl abgelehnter nicht-Infizierter')

    ax[1].set_xlabel('Anzahl abgelehnter Infizierter')
    ax[2].set_xlabel('Anzahl abgelehnter nicht-Infizierter')


    plt.tight_layout()
    f.suptitle(f'Personen: {N:d}, Prävalenz: {p*100:.2f}%, Test-Sensitivität: {sens*100:.2f}%,\nTest-Spezifität: {spec*100:.2f}%, Impfquote: {vr*100:.0f}%, Impfstoffwirksamkeit: {ve*100:.0f}%')
    f.subplots_adjust(top=0.88)

    return f

