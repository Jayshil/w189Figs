import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gd
from pytransit.contamination import TabulatedFilter
from utils import evaluate_pytransit_CowanPC_model
import plotstyles
from utils import lcbin, get_phases, tdur
import pickle
import os

# Instruments
instruments = ['CHEOPS201', 'CHEOPS202', 'CHEOPS203', 'CHEOPS204', 'CHEOPS701', 'CHEOPS702']

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
post = pickle.load(open(os.getcwd() + '/Data/_dynesty_DNS_posteriors_CHEOPS.pkl', 'rb'))
samples = post['posterior_samples']

## Dummy times
dummy_phs_pc = np.linspace(0., 1., 10000)
dummy_tim = T0 + (dummy_phs_pc*per)
#dummy_tim = np.linspace( T0 - 0.49*per, T0 + 0.49*per, 10000)
dummy_fl, dummy_fle = np.ones(len(dummy_tim)), 0.1 * np.ones(len(dummy_tim))
#dummy_phs_pc = juliet.utils.get_phases(t=dummy_tim, P=per, t0=T0, phmin=1.)

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
detrended_fl, residuals = {}, {}
phases_pc = {}
for ins in range(len(instruments)):
    # For GP and linear parameters
    if instruments[ins][0:4] == 'TESS':
        ## GP parameters
        try:
            GP_S0, GP_Q, GP_rho = samples['GP_S0'], samples['GP_Q'], samples['GP_rho']
            GP = True
        except:
            GP_S0, GP_Q, GP_rho = np.zeros(100), np.zeros(100), np.zeros(100)
            GP = False
        ## Linear parameters
        LIN, lin_vecs = False, None
        th_all = np.array([0.])
    elif instruments[ins][0:7] == 'CHEOPS7':
        ## GP params
        GP_S0, GP_Q, GP_rho = samples['GP_S0'], samples['GP_Q'], samples['GP_rho']
        GP = True
        ## For linear parameters
        LIN = True
        th_all = np.array([ np.nanmedian(samples['thetaPC' + str(i)]) for i in range(10) ])
        th_all = np.hstack(( th_all, np.nanmedian(samples['theta10_' + instruments[ins]]) ))
        th_all = np.hstack(( th_all, np.nanmedian(samples['theta11_' + instruments[ins]]) ))
        lin_vecs = lin_pars[instruments[ins]]
    elif instruments[ins][0:7] == 'CHEOPS2':
        ## GP params
        GP_S0, GP_Q, GP_rho = np.zeros(100), np.zeros(100), np.zeros(100)
        GP = False
        ## For linear parameters
        LIN = True
        th_all = np.array([ np.nanmedian(samples['thetaECL' + str(i)]) for i in range(10) ])
        th_all = np.hstack(( th_all, np.nanmedian(samples['theta10_' + instruments[ins]]) ))
        th_all = np.hstack(( th_all, np.nanmedian(samples['theta11_' + instruments[ins]]) ))
        th_all = np.hstack(( th_all, np.nanmedian(samples['theta12_' + instruments[ins]]) ))
        lin_vecs = lin_pars[instruments[ins]]

    # Median model (to detrend the data)
    _, _, lin_model, gp_model, total_model, _ = \
        evaluate_pytransit_CowanPC_model(times=tim[instruments[ins]], fluxes=fl[instruments[ins]], errors=fle[instruments[ins]], fltr=filter_pyt[instruments[ins]],\
                                         per=per, bjd0=T0, rprs=rprs, ar=ar, mst=mst, rst=rst, vsini=vsini, tpole=tpole, phi=phi, lamp=lamp, bb1=bb1,\
                                         E=np.nanmedian(samples['E']), C1=np.nanmedian(samples['C1']), D1=np.nanmedian(samples['D1']), C2=np.nanmedian(C2), D2=np.nanmedian(D2),\
                                         mflx=mflx[instruments[ins]], sigw=sigw[instruments[ins]], LIN=LIN, lin_vecs=lin_vecs, thetas=th_all, GP=GP,\
                                         GP_S0=np.nanmedian(GP_S0), GP_Q=np.nanmedian(GP_Q), GP_rho=np.nanmedian(GP_rho), LTTD=True)
    
    # Detreding the data
    detrended_fl[instruments[ins]] = (fl[instruments[ins]] - lin_model - gp_model) * (1 + mflx[instruments[ins]])
    residuals[instruments[ins]] = (fl[instruments[ins]] - total_model) * 1e6
    phases_pc[instruments[ins]] = get_phases(t=tim[instruments[ins]], P=per, t0=T0, phmin=1.)


physical_model, _, _, _, _, _ = \
    evaluate_pytransit_CowanPC_model(times=dummy_tim, fluxes=dummy_fl, errors=dummy_fle, fltr=filter_pyt[instruments[ins]],\
                                     per=per, bjd0=T0, rprs=rprs, ar=ar, mst=mst, rst=rst, vsini=vsini, tpole=tpole, phi=phi, lamp=lamp, bb1=bb1,\
                                     E=np.nanmedian(samples['E']), C1=np.nanmedian(samples['C1']), D1=np.nanmedian(samples['D1']), C2=np.nanmedian(C2), D2=np.nanmedian(D2),\
                                     mflx=0., sigw=0.)

# Random models (smooth)
random_models = np.zeros((50, len(dummy_tim)))
for i in range(50):
    random_models[i,:], _, _, _, _, _ = \
        evaluate_pytransit_CowanPC_model(times=dummy_tim, fluxes=dummy_fl, errors=dummy_fle, fltr=filter_pyt[instruments[ins]],\
                                         per=per, bjd0=T0, rprs=rprs, ar=ar, mst=mst, rst=rst, vsini=vsini, tpole=tpole, phi=phi, lamp=lamp, bb1=bb1,\
                                         E=np.random.choice(samples['E']), C1=np.random.choice(samples['C1']), D1=np.random.choice(samples['D1']),\
                                         C2=np.random.choice(C2), D2=np.random.choice(D2), mflx=0., sigw=0.)


# Another loop to collect binned data points because if there are more than one instruments,
# binned data points will be common for all instruments
all_phs_pc, all_fl, all_resid = [], [], []
for i in range(len(instruments)):
    ## Saving the data
    ### Phase curve data
    all_phs_pc = all_phs_pc + [ phases_pc[instruments[i]] ]

    ### Flux and residuals
    all_fl = all_fl + [ detrended_fl[instruments[i]] ]
    all_resid = all_resid + [ residuals[instruments[i]] ]

## Binned for phase curves
bin_phs_pc, bin_fl_pc, bin_fle_pc, _ = lcbin(time=np.hstack(all_phs_pc), flux=np.hstack(all_fl), binwidth=0.03)
_, bin_resid_pc, bin_reserr_pc, _ = lcbin(time=np.hstack(all_phs_pc), flux=np.hstack(all_resid), binwidth=0.03)

# Figure codes starts from here

fig = plt.figure(figsize=(16/1.5,9/1.5))
gs = gd.GridSpec(2, 1, height_ratios=[2,1])#, wspace=0.1)

# Phase curve
xlim3, xlim4 = 0.01, 0.99
## Upper panel
ax1 = plt.subplot(gs[0])
for ins in range(len(instruments)):
    ax1.errorbar(phases_pc[instruments[ins]], (detrended_fl[instruments[ins]]-1.)*1e6, fmt='.', alpha=0.1, c='cornflowerblue', zorder=1)
ax1.errorbar(bin_phs_pc, (bin_fl_pc-1.)*1e6, yerr=bin_fle_pc*1e6, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=100)
ax1.plot(dummy_phs_pc, (physical_model-1.)*1e6, c='navy', lw=2.5, zorder=50)
for i in range(50):
    ax1.plot(dummy_phs_pc, (random_models[i,:]-1.)*1e6, c='orangered', alpha=0.5, lw=0.7, zorder=10)

ax1.set_ylabel('Relative flux [ppm]', fontfamily='serif')#, fontsize=14, labelpad=25)
ax1.set_xlim([xlim3, xlim4])
if instruments[0] == 'TESS51':
    ax1.set_ylim([-50., 350.])
ax1.set_ylim([-50., 150.])

ax1.xaxis.set_major_formatter(plt.NullFormatter())

## Bottom panel
ax2 = plt.subplot(gs[1])
for ins in range(len(instruments)):
    ax2.errorbar(phases_pc[instruments[ins]], residuals[instruments[ins]], fmt='.', color='cornflowerblue', alpha=0.1)
ax2.errorbar(bin_phs_pc, bin_resid_pc, yerr=bin_reserr_pc, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=100)
ax2.axhline(0., lw=2.5, color='navy')

ax2.set_xlabel('Orbital Phase')#, fontsize=14, fontfamily='serif')
ax2.set_ylabel('Residuals [ppm]')#, fontsize=14, labelpad=25, fontfamily='serif')
ax2.set_xlim([xlim3, xlim4])

if instruments[0] == 'TESS51':
    ax2.set_ylim([-75., 75.])
else:
    ax2.set_ylim([-20., 20.])

plt.tight_layout()
plt.show()
#plt.savefig(pout + '/phase_folded_cheops.png', dpi=500)