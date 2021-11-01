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
            ax[i].set_xlabel('Anzahl eingelassener Infizierter')
            ax[i].set_ylabel('rel. Häufigkeit')
        else:
            do_hist(ax[i], ev[i], f'3G: {means_rej[(i-1)*2]:.2f}')
            do_hist(ax[i], ev2[i], f'Alle testen: {means_rej[(i-1)*2+1]:.2f}')


    ax[0].set_title('Anzahl eingelassener Infizierter')

    ax[1].set_title('Anzahl abgewiesener Infizierter')
    ax[1].set_xlabel('Anzahl abgewiesener Infizierter')
    
    ax[2].set_title('Anzahl abgewiesener nicht-Infizierter')
    ax[2].set_xlabel('Anzahl abgewiesener nicht-Infizierter')


    plt.tight_layout()
    f.suptitle(f'Personen: {N:d}, Prävalenz: {p*100:.2f}%, Test-Sensitivität: {sens*100:.2f}%,\nTest-Spezifität: {spec*100:.2f}%, Impfquote: {vr*100:.0f}%, Impfstoffwirksamkeit: {ve*100:.0f}%')
    f.subplots_adjust(top=0.88)

    return f



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
from PIL import Image

def plot_figures(
    vr : float = 0.8,
    ve : float = 0.8,
    p : float = 1e-2,
    N : int = 100,
    sens : float = 0.8,
    spec : float = 0.97,
    runs : int = 10_000
    ) -> matplotlib.figure:

    ev = event(vr, ve, p, sens, spec, False, True, N, runs) # 3G

    #x = np.linspace(0, 10, 20)
    #y = np.cos(x)
    vaxed_neg = Image.open('images/person_blue.png')
    unvaxed_neg = Image.open('images/person_green.png')
    vaxed_pos = Image.open('images/person_pink.png')
    unvaxed_pos = Image.open('images/person_red.png')

    images = vaxed_neg, vaxed_pos, unvaxed_neg, unvaxed_pos

    cols = int(np.ceil(np.sqrt(N)))

    rej_pos, rej_neg, pv, pu, v, u  = np.mean(ev[1:], axis=1).astype(int)
    nv, nu = v - pv, u - pu

    #print(nv, pv, nu, pu)

    f, ax = plt.subplots(2, 1, figsize=(10,10))

    r = np.zeros(N, dtype=int)
    r[nv:nv+pv] = 1
    r[nv+pv:nv+pv+nu] = 2
    r[nv+pv+nu:] = 3

    if r.size != 0:
        while True:
            try:
                r = r.reshape(-1, cols)
                break
            except ValueError:
                cols -= 1

        for i in range(4):
            x, y = np.where(r == i)
            imscatter(x, -y, images[i], zoom=0.15, ax=ax[0])
        
        for i in range(4):
            imscatter(0.15 + 0.2 * i, 0.95, images[i], zoom=0.15, ax=ax[0], boxcoords=ax[0].transAxes)
            n = (r == i).sum()
            ax[0].text(0.17 + 0.2 * i, 0.93, f': {n}', transform=ax[0].transAxes)
        yl = ax[0].get_ylim()
        ax[0].set_ylim(yl[0], yl[1] - (yl[1]-y[0]) * 0.2)
    ax[0].set_xticklabels([])
    ax[0].set_yticklabels([])
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    r = np.zeros(rej_pos + rej_neg, dtype=int)
    r[:rej_neg] = 2
    r[rej_neg:] = 3
    cols = int(np.ceil(np.sqrt(rej_pos + rej_neg)))
    
    if r.size != 0:
        while True:
            try:
                r = r.reshape(-1, cols)
                break
            except ValueError:
                cols -= 1

        for i in range(4):
            x, y = np.where(r == i)
            imscatter(x, -y, images[i], zoom=0.15, ax=ax[1])

        for i in range(4):
            imscatter(0.15 + 0.2 * i, 0.95, images[i], zoom=0.15, ax=ax[1], boxcoords=ax[1].transAxes)
            n = (r == i).sum()
            ax[1].text(0.17 + 0.2 * i, 0.93, f': {n}', transform=ax[1].transAxes)
    ax[1].set_xlim(ax[0].get_xlim())
    ax[1].set_ylim(ax[0].get_ylim())
    ax[1].set_xticklabels([])
    ax[1].set_yticklabels([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    return f

def imscatter(x, y, image, ax=None, zoom=1, *args, **kwargs):
    if ax is None:
        ax = plt.gca()
    #try:
    #    image = plt.imread(image)
    #except TypeError:
    #    # Likely already an array...
    #    pass
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False, *args, **kwargs)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists
