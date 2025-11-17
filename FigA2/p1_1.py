import numpy as np
import matplotlib.pyplot as plt
from pytransit.contamination import TabulatedFilter
from utils import evaluate_pytransit_CowanPC_model, GP_model_pred
import plotstyles
from utils import lcbin
import pickle
import os

# This file is to plot GP model

# First Select the model (It can be sine or kelpHomo or kelpInhomo or kelpThm or CowanPC)
pout = os.getcwd() + '/Data'

# Loading the data first
tim7, fl7, fle7 = np.loadtxt(pout + '/lc.dat', usecols=(0,1,2), unpack=True)
## Loading any extra linear parameters
lin_pars = np.genfromtxt(pout + '/lc.dat')
lin_pars = lin_pars[:,3:]

# Filter
## TESS bandpass
wav_t, band_t = np.loadtxt(os.getcwd() + '/Data/tess_response_fun.txt', usecols=(0,1), unpack=True)
wav_t, band_t = wav_t*1e9, band_t/np.max(band_t)
### Saving bandpass so that `pytransit` can understand it!
fltr_t = TabulatedFilter('TESS', wav_t, band_t)

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

# For linear parameters
if lin_pars.shape[1] != 0:
    thetas = np.array([ np.nanmedian( samples['theta' + str(th) + '_TESS51'] ) for th in range(lin_pars.shape[1]) ])
    LIN = True
else:
    LIN, thetas, lin_pars = False, None, None

# Mflux
mflx = np.nanmedian(samples['mflux_TESS51'])
sigw = np.nanmedian(samples['sigma_w_TESS51'])


# --------------------------------------------------------------
#
#                     Computing models
#
# --------------------------------------------------------------
# Median model (to detrend the data)
_, flux_norm, lin_model, gp_model, total_model, _ = \
    evaluate_pytransit_CowanPC_model(times=tim7, fluxes=fl7, errors=fle7, fltr=fltr_t,\
                                     per=np.nanmedian(samples['P_p1']), bjd0=np.nanmedian(samples['t0_p1']), rprs=np.nanmedian(samples['p_p1']), ar=np.nanmedian(samples['a_p1']),\
                                     mst=np.nanmedian(samples['mst']), rst=np.nanmedian(samples['rst']), vsini=np.nanmedian(samples['vsini']), q1=np.nanmedian(samples['q1']),\
                                     q2=np.nanmedian(samples['q2']), tpole=np.nanmedian(samples['tpole']), phi=np.nanmedian(samples['phi']), lamp=np.nanmedian(samples['lamp']),\
                                     bb1=np.nanmedian(samples['b_p1']), E=np.nanmedian(samples['E']), C1=np.nanmedian(samples['C1']), D1=np.nanmedian(samples['D1']),\
                                     C2=np.nanmedian(C2), D2=np.nanmedian(D2), mflx=mflx, sigw=sigw, LIN=LIN, thetas=thetas, lin_vecs=lin_pars, GP=GP,\
                                     GP_S0=np.nanmedian(GP_S0), GP_Q=np.nanmedian(GP_Q), GP_rho=np.nanmedian(GP_rho))
    
# Detreding the data
detrended_fl = fl7 - flux_norm - lin_model
detrended_vars = fle7**2 + (sigw*1e-6)**2
dummy_times = np.linspace(np.min(tim7), np.max(tim7), 10000)

# Binned data
bin_tim, bin_fl, bin_fle, _ = lcbin(time=tim7, flux=detrended_fl, binwidth=0.3)

gp_median_model = GP_model_pred(times=tim7, pred_times=dummy_times, resids=detrended_fl, detrended_vars=detrended_vars,\
                                GP_S0=np.nanmedian(samples['GP_S0']), GP_Q=np.nanmedian(samples['GP_Q']),\
                                GP_rho=np.nanmedian(samples['GP_rho']))

gp_random_models = np.zeros((50, len(dummy_times)))
for i in range(50):
    gp_random_models[i,:] =\
        GP_model_pred(times=tim7, pred_times=dummy_times, resids=detrended_fl, detrended_vars=detrended_vars,\
                      GP_S0=np.random.choice(samples['GP_S0']), GP_Q=np.random.choice(samples['GP_Q']),\
                      GP_rho=np.random.choice(samples['GP_rho']))

tref = int(np.min(tim7))

# Figure codes starts from here

fig, axs = plt.subplots(figsize=(15/1.5, 6/1.5))

# Phase curve
xlim3, xlim4 = np.min(tim7)-tref, np.max(tim7)-tref

axs.errorbar(tim7-tref, detrended_fl*1e6, fmt='.', alpha=0.1, c='cornflowerblue', zorder=1)
axs.errorbar(bin_tim-tref, bin_fl*1e6, yerr=bin_fle*1e6, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=100)
axs.plot(dummy_times-tref, gp_median_model*1e6, c='navy', lw=2.5, zorder=50)
for i in range(50):
    axs.plot(dummy_times-tref, gp_random_models[i,:]*1e6, c='orangered', alpha=0.5, lw=0.7, zorder=10)

axs.set_ylabel('Relative flux [ppm]', fontfamily='serif')#, fontsize=14, labelpad=25)
axs.set_xlabel(r'Time [BJD $-$ ' + str(tref) + ' d]', fontfamily='serif')#, fontsize=14, labelpad=25)
axs.set_xlim([xlim3, xlim4])
axs.set_ylim([-250., 250.])

plt.tight_layout()
plt.show()
#plt.savefig(pout + '/GP_model.pdf')