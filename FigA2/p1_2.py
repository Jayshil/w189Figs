import numpy as np
import matplotlib.pyplot as plt
from pytransit.contamination import TabulatedFilter
from utils import evaluate_pytransit_CowanPC_model
from utils import GP_model_pred
from glob import glob
from utils import tdur, get_phases
import plotstyles
from utils import lcbin
import pickle
import os

# Instruments
instruments = ['CHEOPS701', 'CHEOPS702']
method = 'CowanPC'
pout = os.getcwd() + '/Data'

## Loading the CHEOPS data
cheops_data_all = pickle.load(open(os.getcwd() + '/Data/CHEOPS_data_all_roll.pkl', 'rb'))

# Saving the data in a dictionary (all transits will be saved as different instrument)
tim, fl, fle = {}, {}, {}
lin_pars, filter_pyt, filter_klp = {}, {}, {}
for ins in range(len(instruments)):
    # Transit time and period (from Deline et al. 2022)
    per, T0 = 2.724035, 2459016.434866
    ar, bb1 = 4.587, 0.433
    rprs = 0.06958
    rst, mst = 2.363, 2.073
    vsini, tpole = 92.5, 7967
    phi, lamp = 68.2, 91.7

    t14 = tdur(per=per, ar=ar, rprs=rprs, bb=bb1)

    ins_data = cheops_data_all[instruments[ins][6:]]

    ## Masking transits
    phs_tra = get_phases(t=ins_data['tim'], P=per, t0=T0)
    mask_tra = np.where(np.abs(phs_tra*per) >= 0.5*t14)[0]

    tim[instruments[ins]], fl[instruments[ins]], fle[instruments[ins]] = ins_data['tim'][mask_tra], ins_data['fl'][mask_tra], ins_data['fle'][mask_tra]

    lin_pars[instruments[ins]] = np.transpose( ins_data['lin'][:,mask_tra] )

    ## CHEOPS bandpass
    wav_c, band_c = np.loadtxt(os.getcwd() + '/Data/cheops_response_fun.txt', usecols=(0,1), unpack=True)
    wav_c, band_c = wav_c/10, band_c/np.max(band_c)
    ### Saving bandpass so that `pytransit` can understand it!
    filter_pyt[instruments[ins]] = TabulatedFilter('CHEOPS', wav_c, band_c)

# Creating the median physical model and random models

## Loading the parameters
post = pickle.load(open(pout + '/_dynesty_DNS_posteriors_CHEOPS.pkl', 'rb'))
samples = post['posterior_samples']


## For GP and phase-off param
try:
    ph_off = samples['phaseoff']
except:
    try:
        ph_off = samples['hotspot_off']
    except:
        ph_off = np.zeros(100)

try:
    C2, D2 = samples['C2'], samples['D2']
except:
    C2, D2 = np.zeros(100), np.zeros(100)

# Mflux
mflx, sigw = {}, {}
for i in range(len(instruments)):
    mflx[instruments[i]] = np.nanmedian(samples['mflux_' + instruments[i]])
    sigw[instruments[i]] = np.nanmedian(samples['sigma_w_' + instruments[i]])


# --------------------------------------------------------------
#
#                     Computing models
#
# --------------------------------------------------------------
detrended_fl, detrended_vars = {}, {}
tmin, tmax = 1e100, 0
for ins in range(len(instruments)):
    # For GP and linear parameters
    ## GP params
    GP_S0, GP_Q, GP_rho = samples['GP_S0'], samples['GP_Q'], samples['GP_rho']
    GP = True
    ## For linear parameters
    LIN = True
    th_all = np.array([ np.nanmedian(samples['thetaPC' + str(i)]) for i in range(10) ])
    th_all = np.hstack(( th_all, np.nanmedian(samples['theta10_' + instruments[ins]]) ))
    th_all = np.hstack(( th_all, np.nanmedian(samples['theta11_' + instruments[ins]]) ))
    lin_vecs = lin_pars[instruments[ins]]

    # Median model (to detrend the data)
    _, flux_norm, lin_model, gp_model, total_model, _ = \
        evaluate_pytransit_CowanPC_model(times=tim[instruments[ins]], fluxes=fl[instruments[ins]], errors=fle[instruments[ins]], fltr=filter_pyt[instruments[ins]],\
                                         per=per, bjd0=T0, rprs=rprs, ar=ar, mst=mst, rst=rst, vsini=vsini, q1=0., q2=0., tpole=tpole, phi=phi, lamp=lamp, bb1=bb1,\
                                         E=np.nanmedian(samples['E']), C1=np.nanmedian(samples['C1']), D1=np.nanmedian(samples['D1']), C2=np.nanmedian(C2), D2=np.nanmedian(D2),\
                                         mflx=mflx[instruments[ins]], sigw=sigw[instruments[ins]], LIN=LIN, lin_vecs=lin_vecs, thetas=th_all, GP=GP,\
                                         GP_S0=np.nanmedian(GP_S0), GP_Q=np.nanmedian(GP_Q), GP_rho=np.nanmedian(GP_rho), LTTD=True)
    
    # Detreding the data
    detrended_fl[instruments[ins]] = (fl[instruments[ins]] - lin_model - flux_norm)
    detrended_vars[instruments[ins]] = fle[instruments[ins]]**2 + ( sigw[instruments[ins]] * 1e-6 )**2

    # Determining tmin and tmax over all instruments
    if np.min( tim[instruments[ins]] ) < tmin:
        tmin = np.min( tim[instruments[ins]] )
    if np.max( tim[instruments[ins]] ) > tmax:
        tmax = np.max( tim[instruments[ins]] )

# Collecting all times, detrended fluxes and vars in a big array so that we can compute a GP model for that
all_tim, all_detrended_fl, all_detrended_vars = np.array([]), np.array([]), np.array([])
for ins in range(len(instruments)):
    all_tim = np.hstack(( all_tim, tim[instruments[ins]] ))
    all_detrended_fl = np.hstack(( all_detrended_fl, detrended_fl[instruments[ins]] ))
    all_detrended_vars = np.hstack(( all_detrended_vars, detrended_vars[instruments[ins]] ))

# Dummy times
dummy_tim = np.linspace(tmin, tmax, 10000)

# Median physical model (smooth)
gp_median_model = GP_model_pred(times=all_tim, pred_times=dummy_tim, resids=all_detrended_fl,\
                                detrended_vars=all_detrended_vars, GP_S0=np.nanmedian(samples['GP_S0']),\
                                GP_Q=np.nanmedian(samples['GP_Q']), GP_rho=np.nanmedian(samples['GP_rho']))

# Random models (smooth)
gp_random_models = np.zeros((50, len(dummy_tim)))
for i in range(50):
    gp_random_models[i,:] = GP_model_pred(times=all_tim, pred_times=dummy_tim, resids=all_detrended_fl,\
                                          detrended_vars=all_detrended_vars, GP_S0=np.random.choice(samples['GP_S0']),\
                                          GP_Q=np.random.choice(samples['GP_Q']), GP_rho=np.random.choice(samples['GP_rho']))


# To collect binned data points because if there are more than one instruments,
# binned data points will be common for all instruments

## Binned for phase curves
bin_tim, bin_fl, bin_fle, _ = lcbin(time=np.hstack(all_tim), flux=np.hstack(all_detrended_fl), binwidth=0.3)
tref = int(np.min(all_tim))

# Figure codes starts from here
fig, axs = plt.subplots(figsize=(15/1.5, 6/1.5))

# Phase curve
xlim3, xlim4 = np.min(all_tim)-tref, np.max(all_tim)-tref

axs.errorbar(all_tim-tref, all_detrended_fl*1e6, fmt='.', alpha=0.1, c='cornflowerblue', zorder=1)
axs.errorbar(bin_tim-tref, bin_fl*1e6, yerr=bin_fle*1e6, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=100)
axs.plot(dummy_tim-tref, gp_median_model*1e6, c='navy', lw=2.5, zorder=50)
for i in range(50):
    axs.plot(dummy_tim-tref, gp_random_models[i,:]*1e6, c='orangered', alpha=0.5, lw=0.7, zorder=10)

axs.set_ylabel('Relative flux [ppm]', fontfamily='serif')#, fontsize=14, labelpad=25)
axs.set_xlabel(r'Time [BJD $-$ ' + str(tref) + ' d]', fontfamily='serif')#, fontsize=14, labelpad=25)
axs.set_xlim([xlim3, xlim4])

axs.set_ylim([-100., 100.])

plt.tight_layout()
plt.show()
#plt.savefig(pout + '/GP_model.pdf')