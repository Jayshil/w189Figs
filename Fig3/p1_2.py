import numpy as np
import matplotlib.pyplot as plt
import os
import plotstyles

# This file is to compute equitorial temperature map from Cowan & Agol (2008)

nphi = 100
pin = os.getcwd() + '/Data'

# ---------------------------------------------------
#
# ---------------------------------------------------

# Computing the map
phi_ang = np.linspace(-np.pi, np.pi, nphi)

# Loading the temperature map
tmap = np.load(pin + '/Temperature_map.npy')
eq_tmap_samples = tmap[:, :, 24]

fig, axs = plt.subplots(figsize=(16/2, 9/2))
axs.plot(np.rad2deg(phi_ang), np.nanmedian(eq_tmap_samples, axis=0), color='navy', lw=2., zorder=100)
for i in range(100):
    axs.plot(np.rad2deg(phi_ang), eq_tmap_samples[np.random.randint(0,eq_tmap_samples.shape[0]), :], alpha=0.5, color='orangered', lw=1., zorder=10)

axs.set_xlim([-180., 180.])

axs.set_xlabel('Longitude [deg]')
axs.set_ylabel(r'Temperature [K]')

plt.tight_layout()

plt.show()
#plt.savefig(out_path + '/Equitorial_temp_variation.pdf')#, dpi=500)"""