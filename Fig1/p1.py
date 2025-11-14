import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gd
import juliet
from pytransit.contamination import TabulatedFilter
from utils import evaluate_pytransit_CowanPC_model
import plotstyles
from utils import lcbin
from kelp import Filter
import pickle
import os

# First Select the model (It can be sine or kelpHomo or kelpInhomo or kelpThm or CowanPC)
pout = os.getcwd() + '/Data'

# Loading the data first
tim7, fl7, fle7 = np.loadtxt(pout + '/lc.dat', usecols=(0,1,2), unpack=True)
## Loading any extra linear parameters
lin_pars = np.genfromtxt(pout + '/lc.dat')
lin_pars = lin_pars[:,3:]

# Filter
## TESS bandpass
wav_t, band_t = np.loadtxt(pout + '/tess_response_fun.txt', usecols=(0,1), unpack=True)
wav_t, band_t = wav_t*1e9, band_t/np.max(band_t)
### Saving bandpass so that `pytransit` can understand it!
fltr_t = TabulatedFilter('TESS', wav_t, band_t)

## TESS bandpass, in Kelp way!
filt_kelp = Filter.from_name("TESS")

# Creating the median physical model and random models

## Loading the parameters
post = pickle.load(open(pout + '/_dynesty_DNS_posteriors.pkl', 'rb'))
samples = post['posterior_samples']

## Dummy times
dummy_tim = np.linspace( np.nanmedian(samples['t0_p1'] - 0.499*samples['P_p1']), np.nanmedian(samples['t0_p1'] + 0.499*samples['P_p1']), 10000)
dummy_fl, dummy_fle = np.ones(len(dummy_tim)), 0.1 * np.ones(len(dummy_tim))

phases_tra = juliet.utils.get_phases(t=tim7, P=np.nanmedian(samples['P_p1']), t0=np.nanmedian(samples['t0_p1']), phmin=0.5)
phases_pc = juliet.utils.get_phases(t=tim7, P=np.nanmedian(samples['P_p1']), t0=np.nanmedian(samples['t0_p1']), phmin=0.8)

dummy_phs_tra = juliet.utils.get_phases(t=dummy_tim, P=np.nanmedian(samples['P_p1']), t0=np.nanmedian(samples['t0_p1']), phmin=0.5)
dummy_phs_pc = juliet.utils.get_phases(t=dummy_tim, P=np.nanmedian(samples['P_p1']), t0=np.nanmedian(samples['t0_p1']), phmin=0.8)

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
# Median model (To detrend the data)
_, _, lin_model, gp_model, total_model, _ = \
    evaluate_pytransit_CowanPC_model(times=tim7, fluxes=fl7, errors=fle7, fltr=fltr_t,\
                                      per=np.nanmedian(samples['P_p1']), bjd0=np.nanmedian(samples['t0_p1']),\
                                      rprs=np.nanmedian(samples['p_p1']), ar=np.nanmedian(samples['a_p1']),\
                                      mst=np.nanmedian(samples['mst']), rst=np.nanmedian(samples['rst']), vsini=np.nanmedian(samples['vsini']),\
                                      q1=np.nanmedian(samples['q1']), q2=np.nanmedian(samples['q2']), tpole=np.nanmedian(samples['tpole']),\
                                      phi=np.nanmedian(samples['phi']), lamp=np.nanmedian(samples['lamp']), bb1=np.nanmedian(samples['b_p1']),\
                                      E=np.nanmedian(samples['E']), C1=np.nanmedian(samples['C1']), D1=np.nanmedian(samples['D1']),\
                                      C2=np.nanmedian(C2), D2=np.nanmedian(D2), mflx=mflx, sigw=sigw,\
                                      LIN=LIN, thetas=thetas, lin_vecs=lin_pars,\
                                      GP=GP, GP_S0=np.nanmedian(GP_S0), GP_Q=np.nanmedian(GP_Q), GP_rho=np.nanmedian(GP_rho))

# Median physical model (smooth)
physical_model, _, _, _, _, _ = \
    evaluate_pytransit_CowanPC_model(times=dummy_tim, fluxes=dummy_fl, errors=dummy_fle, fltr=fltr_t,\
                                     per=np.nanmedian(samples['P_p1']), bjd0=np.nanmedian(samples['t0_p1']),\
                                     rprs=np.nanmedian(samples['p_p1']), ar=np.nanmedian(samples['a_p1']),\
                                     mst=np.nanmedian(samples['mst']), rst=np.nanmedian(samples['rst']), vsini=np.nanmedian(samples['vsini']),\
                                     q1=np.nanmedian(samples['q1']), q2=np.nanmedian(samples['q2']), tpole=np.nanmedian(samples['tpole']),\
                                     phi=np.nanmedian(samples['phi']), lamp=np.nanmedian(samples['lamp']), bb1=np.nanmedian(samples['b_p1']),\
                                     E=np.nanmedian(samples['E']), C1=np.nanmedian(samples['C1']), D1=np.nanmedian(samples['D1']),\
                                     C2=np.nanmedian(C2), D2=np.nanmedian(D2), mflx=mflx, sigw=sigw)

# Random models (smooth)
random_models = np.zeros((50, len(dummy_tim)))
for i in range(50):
    random_models[i,:], _, _, _, _, _ = \
        evaluate_pytransit_CowanPC_model(times=dummy_tim, fluxes=dummy_fl, errors=dummy_fle, fltr=fltr_t,\
                                         per=np.random.choice(samples['P_p1']), bjd0=np.random.choice(samples['t0_p1']),\
                                         rprs=np.random.choice(samples['p_p1']), ar=np.random.choice(samples['a_p1']),\
                                         mst=np.random.choice(samples['mst']), rst=np.random.choice(samples['rst']), vsini=np.random.choice(samples['vsini']),\
                                         q1=np.random.choice(samples['q1']), q2=np.random.choice(samples['q2']), tpole=np.random.choice(samples['tpole']),\
                                         phi=np.random.choice(samples['phi']), lamp=np.random.choice(samples['lamp']), bb1=np.random.choice(samples['b_p1']),\
                                         E=np.random.choice(samples['E']), C1=np.random.choice(samples['C1']), D1=np.random.choice(samples['D1']),\
                                         C2=np.random.choice(C2), D2=np.random.choice(D2), mflx=np.random.choice(samples['mflux_TESS51']),\
                                         sigw=np.random.choice(samples['sigma_w_TESS51']))


# Detreding the data
detrended_fl = (fl7 - lin_model - gp_model) * (1 + mflx)
detrended_fle = np.sqrt( fle7**2 + (np.nanmedian(sigw*1e-6))**2 )

resids = (fl7 - total_model) * 1e6

# Binned data (transit)
bin_tim_tra, bin_fl_tra, bin_fle_tra, _ = lcbin(phases_tra, detrended_fl, 0.005)
_, bin_resid_tra, bin_resid_err_tra, _ = lcbin(phases_tra, resids, 0.005)

# Binned data (phase curve)
bin_tim_pc, bin_fl_pc, bin_fle_pc, _ = lcbin(phases_pc, detrended_fl, 0.02)
_, bin_resid_pc, bin_resid_err_pc, _ = lcbin(phases_pc, resids, 0.02)

idx_dummy_phs = np.argsort(dummy_phs_pc)
# Figure codes starts from here

fig = plt.figure(figsize=(16/1.5, 14/1.5))
gs = gd.GridSpec(3, 2, height_ratios=[6,6,3], width_ratios=[1,3], hspace=0.1, wspace=0.1)

ax0 = plt.subplot(gs[0,:])
ax0.errorbar(phases_pc, detrended_fl, fmt='.', alpha=0.1, c='cornflowerblue', zorder=1)
ax0.errorbar(bin_tim_pc, bin_fl_pc, yerr=bin_fle_pc, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=100)
ax0.plot(dummy_phs_pc[idx_dummy_phs], physical_model[idx_dummy_phs], c='navy', lw=2.5, zorder=50)

ax0.set_ylabel('Normalised flux', fontsize=14, fontfamily='serif')
ax0.set_xlabel('Orbital phase', fontsize=14, fontfamily='serif', labelpad=10)

ax0.xaxis.tick_top()
ax0.xaxis.set_label_position('top')

ax0.set_xlim([-0.2, 0.8])
ax0.set_ylim([0.9935, 1+1000e-6])
#ax0.xaxis.set_major_formatter(plt.NullFormatter())

# Transit
xlim1, xlim2 = -0.05, 0.05
## -- Upper panel
ax1 = plt.subplot(gs[1,0])
ax1.errorbar(phases_tra, (detrended_fl-1.)*1e6, fmt='.', alpha=0.25, c='cornflowerblue', zorder=1)
ax1.errorbar(bin_tim_tra, (bin_fl_tra-1.)*1e6, yerr=bin_fle_tra*1e6, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=100)
ax1.plot(dummy_phs_tra, (physical_model-1.)*1e6, color='navy', lw=2.5, zorder=50)
for i in range(50):
    ax1.plot(dummy_phs_tra, (random_models[i,:]-1.)*1e6, color='orangered', alpha=0.5, lw=0.7, zorder=10)

ax1.set_ylabel('Relative flux [ppm]', fontsize=14, fontfamily='serif')
ax1.set_xlim([xlim1, xlim2])
ax1.set_ylim([(0.9935-1.)*1e6, 1000])

ax1.tick_params(labelfontfamily='serif')
ax1.set_xticks(ticks=np.array([-0.04, 0.0, 0.04]))
plt.setp(ax1.get_yticklabels(), fontsize=12)
ax1.xaxis.set_major_formatter(plt.NullFormatter())

## -- Bottom panel
ax2 = plt.subplot(gs[2,0])
ax2.errorbar(phases_tra, resids, fmt='.', alpha=0.1, c='cornflowerblue')
ax2.errorbar(bin_tim_tra, bin_resid_tra, yerr=bin_resid_err_tra, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=20)
ax2.axhline(0.0, lw=2.5, color='navy', zorder=10)

ax2.set_xlabel('Orbital phase', fontsize=14, fontfamily='serif')
ax2.set_ylabel('Residuals [ppm]', fontsize=14, fontfamily='serif')
ax2.set_xlim([xlim1, xlim2])
ax2.set_ylim([-200., 200.])

ax2.set_xticks(ticks=np.array([-0.04, 0.0, 0.04]), labels=np.array([-0.04, 0.0, 0.04]))
ax2.tick_params(labelfontfamily='serif')
plt.setp(ax2.get_xticklabels(), fontsize=12)
plt.setp(ax2.get_yticklabels(), fontsize=12)


# Phase curve
xlim3, xlim4 = -0.2, 0.8

## Upper panel
ax3 = plt.subplot(gs[1,1])
ax3.errorbar(phases_pc, (detrended_fl-1.)*1e6, fmt='.', alpha=0.1, c='cornflowerblue', zorder=1)
ax3.errorbar(bin_tim_pc, (bin_fl_pc-1.)*1e6, yerr=bin_fle_pc*1e6, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=100)
ax3.plot(dummy_phs_pc[idx_dummy_phs], (physical_model[idx_dummy_phs]-1.)*1e6, c='navy', lw=2.5, zorder=50)
for i in range(50):
    ax3.plot(dummy_phs_pc[idx_dummy_phs], (random_models[i,:][idx_dummy_phs]-1.)*1e6, c='orangered', alpha=0.7, lw=0.7, zorder=10)
ax3.axhline(0., color='k', ls='--', lw=1.5, zorder=25)

ax3.set_ylabel('Relative flux [ppm]', rotation=270, fontsize=14, labelpad=25, fontfamily='serif')
ax3.set_xlim([xlim3, xlim4])
ax3.set_ylim([-150., 300.])

ax3.yaxis.tick_right()
ax3.tick_params(labelright=True, labelfontfamily='serif')
plt.setp(ax3.get_yticklabels(), fontsize=12)
ax3.yaxis.set_label_position('right')
ax3.xaxis.set_major_formatter(plt.NullFormatter())

## Bottom panel
ax4 = plt.subplot(gs[2,1])
ax4.errorbar(phases_pc, resids, fmt='.', color='cornflowerblue', alpha=0.1)
ax4.errorbar(bin_tim_pc, bin_resid_pc, yerr=bin_resid_err_pc, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=100)
ax4.axhline(0., lw=2.5, color='navy')

ax4.set_xlabel('Orbital phase', fontsize=14, fontfamily='serif')
ax4.set_ylabel('Residuals [ppm]', rotation=270, fontsize=14, labelpad=25, fontfamily='serif')
ax4.set_xlim([xlim3, xlim4])
ax4.set_ylim([-75., 75.])

ax4.yaxis.tick_right()
ax4.tick_params(labelright=True, labelfontfamily='serif')
plt.setp(ax4.get_xticklabels(), fontsize=12)
plt.setp(ax4.get_yticklabels(), fontsize=12)
ax4.yaxis.set_label_position('right')

plt.tight_layout()
plt.show()
#plt.savefig(pout + '/combined_fig.pdf')