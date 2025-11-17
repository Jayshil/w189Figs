import numpy as np
import matplotlib.pyplot as plt
import plotstyles
import os

tim_pdc, fl_pdc, fle_pdc = np.loadtxt(os.getcwd() + '/Data/TESS51.dat', usecols=(0,1,2), unpack=True)
tim_scal, fl_scal, fle_scal = np.loadtxt(os.getcwd() + '/Data/simple_aperture_photometry.dat', usecols=(0,1,2), unpack=True)


tref = int(np.min(tim_scal))

fig, axs = plt.subplots()
axs.errorbar(tim_pdc-tref, fl_pdc, yerr=fle_pdc, fmt='.', color='orangered', label='PDC-SAP')
axs.errorbar(tim_scal-tref, fl_scal-0.009, yerr=fle_scal, fmt='.', color='cornflowerblue', label='SCALPLES')

axs.set_xlabel(r'Time [BJD $-$ {:d} d]'.format(tref))
axs.set_ylabel('Relative flux + offset')

#axs.legend(loc='lower left')

axs.set_yticks(ticks=np.array([0.986, 0.988, 0.990, 0.992, 0.994, 0.996, 0.998, 1.000, 1.002]),\
               labels=np.array(['0.986', '', '0.990', '', '0.994', '', '0.998', '', '1.002']))

axs.text(0.5, 0.985, 'PDC-SAP light curve', color='orangered')
axs.text(0.5, 0.984, 'Simple aperture photometry', color='cornflowerblue')

axs.set_ylim([0.983, 1.002])

plt.show()
#plt.savefig(os.getcwd() + '/Data/data_comp.png', dpi=500)