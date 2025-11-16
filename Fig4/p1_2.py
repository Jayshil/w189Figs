import numpy as np
import matplotlib.pyplot as plt
import plotstyles
import os

# Loading the temperature map
temp_map = np.load(os.getcwd() + '/Data/Med_temperature_Map.npy')

# Generating the nesg
ntheta, nphi = 49, 100

# Computing the map
phi_ang = np.linspace(-np.pi, np.pi, nphi)
theta_ang = np.linspace(-np.pi/2, np.pi/2, ntheta)

theta2d, phi2d = np.meshgrid(theta_ang, phi_ang)

# Plot the temperature map
fig = plt.figure()
ax = fig.add_subplot(111, projection='mollweide')
cax = ax.pcolormesh(phi2d, theta2d, temp_map)
cax.set_clim([1500,3000])
plt.colorbar(cax, label='T [K]')
plt.tight_layout()
plt.show()
#plt.savefig(out_path + '/Med_tmap.png', dpi=500)