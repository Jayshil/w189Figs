import numpy as np
import matplotlib.pyplot as plt
from pytransit.contamination import TabulatedFilter
from utils import evaluate_pytransit_CowanPC_model, get_phases, tdur
import plotstyles
from astropy.timeseries import LombScargle
import astropy.units as u
from astropy.stats import SigmaClip, mad_std
import pickle
import os

# This file is to plot a PSD of the residuals

# First Select the model (It can be sine or kelpHomo or kelpInhomo or kelpThm or CowanPC)
mask_traocc = True
pout = os.getcwd() + '/Data'

# Loading the data first
tim7, fl7, fle7 = np.loadtxt(pout + '/lc.dat', usecols=(0,1,2), unpack=True)

# Filter
## TESS bandpass
wav_t, band_t = np.loadtxt(os.getcwd() + '/Data/tess_response_fun.txt', usecols=(0,1), unpack=True)
wav_t, band_t = wav_t*1e9, band_t/np.max(band_t)
### Saving bandpass so that `pytransit` can understand it!
fltr_t = TabulatedFilter('TESS', wav_t, band_t)

# Creating the median physical model and random models

## Loading the parameters
post = pickle.load(open(pout + '/_dynesty_DNS_posteriors.pkl', 'rb'))
samples = post['posterior_samples']


## For GP and phase-off param
try:
    ph_off = samples['phaseoff']
except:
    try:
        ph_off = samples['hotspot_off']
    except:
        ph_off = np.zeros(100)

## For second order phase curve parameters in case of Cowan & Agol 2008 phase curve
try:
    C2, D2 = samples['C2'], samples['D2']
except:
    C2, D2 = np.zeros(100), np.zeros(100)

# For GP parameters
try:
    GP_S0, GP_Q, GP_rho = samples['GP_S0'], samples['GP_Q'], samples['GP_rho']
    GP = True
except:
    GP_S0, GP_Q, GP_rho = np.zeros(100), np.zeros(100), np.zeros(100)
    GP = False

# Mflux
mflx = np.nanmedian(samples['mflux_TESS51'])
sigw = np.nanmedian(samples['sigma_w_TESS51'])


# --------------------------------------------------------------
#
#                     Computing models
#
# --------------------------------------------------------------
# Median model (To detrend the data)
_, _, lin_model, gp_model, total_model, _ = \
    evaluate_pytransit_CowanPC_model(times=tim7, fluxes=fl7, errors=fle7, fltr=fltr_t,\
                                     per=np.nanmedian(samples['P_p1']), bjd0=np.nanmedian(samples['t0_p1']),\
                                     rprs=np.nanmedian(samples['p_p1']), ar=np.nanmedian(samples['a_p1']),\
                                     mst=np.nanmedian(samples['mst']), rst=np.nanmedian(samples['rst']), vsini=np.nanmedian(samples['vsini']),\
                                     q1=np.nanmedian(samples['q1']), q2=np.nanmedian(samples['q2']), tpole=np.nanmedian(samples['tpole']),\
                                     phi=np.nanmedian(samples['phi']), lamp=np.nanmedian(samples['lamp']), bb1=np.nanmedian(samples['b_p1']),\
                                     E=np.nanmedian(samples['E']), C1=np.nanmedian(samples['C1']), D1=np.nanmedian(samples['D1']),\
                                     C2=np.nanmedian(C2), D2=np.nanmedian(D2), mflx=mflx, sigw=sigw, GP=GP, GP_S0=np.nanmedian(GP_S0),\
                                     GP_Q=np.nanmedian(GP_Q), GP_rho=np.nanmedian(GP_rho))


# Residuals
resids = (fl7 - lin_model - gp_model)# * 1e6
resid_err = np.sqrt( fle7**2 + (sigw * 1e-6)**2 )

# ---------------------------------------------
#             Computing the PSD
# ---------------------------------------------
t14 = tdur(per=np.nanmedian(samples['P_p1']), ar=np.nanmedian(samples['a_p1']), rprs=np.nanmedian(samples['p_p1']), bb=np.nanmedian(samples['b_p1']))
if mask_traocc:
    phs_t = get_phases(tim7, np.nanmedian(samples['P_p1']), np.nanmedian(samples['t0_p1']))
    phs_e = get_phases(tim7, np.nanmedian(samples['P_p1']), np.nanmedian(samples['t0_p1']) + np.nanmedian(samples['P_p1'])/2)

    mask = np.where((np.abs(phs_e*np.nanmedian(samples['P_p1'])) >= t14/2)&(np.abs(phs_t*np.nanmedian(samples['P_p1'])) >= t14/2))[0]
    #mask = np.where(np.abs(phs_t*per) >= t14/2)[0]
    tim7, resids, resid_err = tim7[mask], resids[mask], resid_err[mask]

# Sigma clipping
sc = SigmaClip(sigma_upper=3.5, sigma_lower=3.5, stdfunc=mad_std, maxiters=None)
msk1 = sc(resids).mask

tim, resids, resid_err = tim7[~msk1], resids[~msk1], resid_err[~msk1]


# -------------------------------
#      PSD: un-binned data
# -------------------------------

tim1 = tim*24*60*60*u.second
fl1 = 1e6*resids*u.dimensionless_unscaled
min_freq, max_freq = 1 / np.ptp(tim1), 1 / np.nanmedian(np.diff(tim1)) * 0.5

freq_grid = np.linspace(min_freq, max_freq, 100000)
psd1 = LombScargle(tim1, fl1, normalization='psd').power(frequency=freq_grid)

idx_max_pow = np.argsort(psd1.value)

freq1 = freq_grid[idx_max_pow[-1]]
per2 = 1/freq1
per3 = per2.to(u.day)


# Making the plot:
fig, axs = plt.subplots(figsize=(16/2, 9/2))

## Un-binned data
axs.plot(freq_grid, psd1, alpha=0.7, color='orangered', label='For un-binned data', zorder=10)
axs.axvline(freq1.value, ls='--', c='maroon', lw=1., zorder=5)
axs.text(3.29e-6, 0.9, 'Maximum power: ', rotation=90)#, fontweight='bold')
axs.text(4.6e-6, 0.9, str(np.around(per3.value, 4)) + ' d', rotation=90)#, fontweight='bold')


# Define properties to define upper axis as well:
def freq2tim(x):
    x = x * u.Hz
    return (1/x).to(u.d).value
def tim2freq(x):
    x = x * u.s
    return (1/x).to(u.Hz).value

#axs.axvline(tim2freq(0.24 * 24 * 60 * 60), ls=':', lw=1., c='k', zorder=5)
#axs.axvline(tim2freq(1.2 * 24 * 60 * 60), ls=':', lw=1., c='r', zorder=5)

ax2 = axs.secondary_xaxis("top", functions=(freq2tim, tim2freq))
ax2.tick_params(axis='both', which='major')
ax2.set_xlabel("Time [d]")

axs.set_xscale('log')
axs.set_yscale('log')
axs.set_xlabel(r'Frequency [Hz]')
axs.set_ylabel(r'Power [ppm$^2$ Hz$^{-1}$]')
axs.set_xlim([np.min(freq_grid.value), np.max(freq_grid.value)])

plt.tight_layout()
plt.show()
#plt.savefig(os.getcwd() + '/StelVar/Plots/psd.pdf')#, dpi=500)