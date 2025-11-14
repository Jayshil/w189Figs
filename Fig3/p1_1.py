import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
import pickle
import os
import plotstyles

# This file is to compute equitorial temperature map from Cowan & Agol (2008)
nphi = 100
pin = os.getcwd() + '/Data'

# ---------------------------------------------------
#
# ---------------------------------------------------

# Now, loading the posteriors
fname = glob(pin + '/*.pkl')[0]
post = pickle.load(open(fname, 'rb'))
post1 = post['posterior_samples']

rprs = np.nanmedian(post1['p_p1'])

## Cowan & Agol (2008) phase curve parameters
E, C1, D1 = post1['E'], post1['C1'], post1['D1']

try:
    C2 = post1['C2']
    D2 = post1['D2']
except:
    C2, D2 = np.zeros(len(E)), np.zeros(len(E))

## Compute the temperature map parameters
A0 = (E - C1 - C2) / 2
A1 = 2 * C1 / np.pi
B1 = -2 * D1 / np.pi
A2 = 3 * C2 / 2
B2 = -3 * D2 / 2

# Computing the map
phi_ang = np.linspace(-np.pi, np.pi, nphi)

# Computing the Fp/F* map
fpfs_map_pos_samples = np.zeros((len(E), nphi))
for integration in tqdm(range(len(E))):
    fpfs_ph = A0[integration] + (A1[integration] * np.cos(phi_ang)) + (B1[integration] * np.sin(phi_ang))+\
                                (A2[integration] * np.cos(2*phi_ang)) + (B2[integration] * np.sin(2*phi_ang))
    fpfs_map_pos_samples[integration, :] = fpfs_ph * 0.75

fig, axs = plt.subplots(figsize=(16/2, 9/2))
axs.plot(np.rad2deg(phi_ang), np.nanmedian(fpfs_map_pos_samples*1e6, axis=0), color='navy', lw=2., zorder=100)
for i in range(100):
    axs.plot(np.rad2deg(phi_ang), fpfs_map_pos_samples[np.random.randint(0,fpfs_map_pos_samples.shape[0]), :]*1e6, alpha=0.5, color='orangered', lw=1., zorder=10)

axs.set_xlim([-180., 180.])

axs.set_xlabel('Longitude [deg]')
axs.set_ylabel(r'I$_\text{eq}$ [ppm]')

plt.tight_layout()

plt.show()
#plt.savefig(out_path + '/fpfs_vs_long.pdf')#, dpi=500)