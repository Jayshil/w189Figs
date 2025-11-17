import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import get_quantiles
import os
import plotstyles

# This file is to plot equator temperature distribution comparison from TESS and CHEOPS

# Longitude angles
phi_ang = np.rad2deg( np.linspace(-np.pi, np.pi, 100) )

# Loading the equator temperature map
eq_temp_tess = np.load(os.getcwd() + '/Data/Equatorial_temp_Map.npy')
eq_temp_cheops = np.load(os.getcwd() + '/Data/Equatorial_temp_Map_CHEOPS.npy')

# Quantile models
quantiles_tess, quantiles_cheops = np.zeros((3, len(phi_ang))), np.zeros((3, len(phi_ang)))
for i in tqdm(range(100)):
    # For TESS
    qua_tess = get_quantiles(eq_temp_tess[:,i])
    quantiles_tess[0,i], quantiles_tess[1,i], quantiles_tess[2,i] = qua_tess[0], qua_tess[1], qua_tess[2]

    # For CHEOPS
    qua_cheops = get_quantiles(eq_temp_cheops[:,i])
    quantiles_cheops[0,i], quantiles_cheops[1,i], quantiles_cheops[2,i] = qua_cheops[0], qua_cheops[1], qua_cheops[2]


fig, axs = plt.subplots(figsize=(16/2, 9/2))

# For TESS
axs.plot(phi_ang, quantiles_tess[0,:], color='orangered', lw=2., zorder=50, label='TESS')
### Errorbars
axs.fill_between(phi_ang, y1=quantiles_tess[2], y2=quantiles_tess[1], color='orangered', zorder=10, alpha=0.5)
axs.plot(phi_ang, quantiles_tess[2], color='orangered', lw=0.7, zorder=10)
axs.plot(phi_ang, quantiles_tess[1], color='orangered', lw=0.7, zorder=10)

# For CHEOPS
axs.plot(phi_ang, quantiles_cheops[0,:], color='royalblue', lw=2., zorder=50, label='CHEOPS')
### Errorbars
axs.fill_between(phi_ang, y1=quantiles_cheops[2], y2=quantiles_cheops[1], color='cornflowerblue', zorder=10, alpha=0.5)
axs.plot(phi_ang, quantiles_cheops[2], color='cornflowerblue', lw=0.7, zorder=10)
axs.plot(phi_ang, quantiles_cheops[1], color='cornflowerblue', lw=0.7, zorder=10)

#axs.axvline(0., ls='--', lw=0.9, color='k', zorder=0)
#axs.axvline(90., ls='-.', lw=0.9, color='k', zorder=0)
#axs.axvline(-90., ls='-.', lw=0.9, color='k', zorder=0)

axs.set_xlim([-130., 130.])
axs.set_ylim([2000, 3350])

axs.set_xlabel('Longitude [deg]')
axs.set_ylabel('Equatorial Temperature [K]')

#axs.legend(loc='lower center')
axs.text(-60, 2200, 'Temperature in TESS bandpass', color='orangered')
axs.text(-67, 2100, 'Temperature in CHEOPS bandpass', color='royalblue')

#axs.text(3., 2400., 'Local noon', color='crimson', rotation=90)

#axs.text(-10., 3275., 'Day', color='deeppink')
#axs.text(-120., 3275., 'Night', color='navy')
#axs.text(100., 3275., 'Night', color='navy')

plt.tight_layout()

plt.show()
#plt.savefig(os.getcwd() + '/PC/Analysis/CowanPC_wGP_wPhOff/sin34/temp_comp.pdf')