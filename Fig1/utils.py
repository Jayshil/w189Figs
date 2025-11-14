import numpy as np
from celerite2.terms import SHOTerm
from celerite2 import GaussianProcess
from pytransit import OblateStarModel
from scipy.signal import medfilt
import astropy.constants as con
import astropy.units as u
import juliet

def lcbin(time, flux, binwidth=0.06859, nmin=4, time0=None,
        robust=False, tmid=False):
    """
    ------ A function from pycheops -------
    Calculate average flux and error in time bins of equal width.
    The default bin width is equivalent to one CHEOPS orbit in units of days.
    To avoid binning data on either side of the gaps in the light curve due to
    the CHEOPS orbit, the algorithm searches for the largest gap in the data
    shorter than binwidth and places the bin edges so that they fall at the
    centre of this gap. This behaviour can be avoided by setting a value for
    the parameter time0.
    The time values for the output bins can be either the average time value
    of the input points or, if tmid is True, the centre of the time bin.
    If robust is True, the output bin values are the median of the flux values
    of the bin and the standard error is estimated from their mean absolute
    deviation. Otherwise, the mean and standard deviation are used.
    The output values are as follows.
    * t_bin - average time of binned data points or centre of time bin.
    * f_bin - mean or median of the input flux values.
    * e_bin - standard error of flux points in the bin.
    * n_bin - number of flux points in the bin.
    :param time: time
    :param flux: flux (or other quantity to be time-binned)
    :param binwidth:  bin width in the same units as time
    :param nmin: minimum number of points for output bins
    :param time0: time value at the lower edge of one bin
    :param robust: use median and robust estimate of standard deviation
    :param tmid: return centre of time bins instead of mean time value
    :returns: t_bin, f_bin, e_bin, n_bin
    """
    if time0 is None:
        tgap = (time[1:]+time[:-1])/2
        gap = time[1:]-time[:-1]
        j = gap < binwidth
        gap = gap[j]
        tgap = tgap[j]
        time0 = tgap[np.argmax(gap)]
        time0 = time0 - binwidth*np.ceil((time0-min(time))/binwidth)

    n = int(1+np.ceil(np.ptp(time)/binwidth))
    r = (time0,time0+n*binwidth)
    n_in_bin,bin_edges = np.histogram(time,bins=n,range=r)
    bin_indices = np.digitize(time,bin_edges)

    t_bin = np.zeros(n)
    f_bin = np.zeros(n)
    e_bin = np.zeros(n)
    n_bin = np.zeros(n, dtype=int)

    for i,n in enumerate(n_in_bin):
        if n >= nmin:
            j = bin_indices == i+1
            n_bin[i] = n
            if tmid:
                t_bin[i] = (bin_edges[i]+bin_edges[i+1])/2
            else:
                t_bin[i] = np.nanmean(time[j])
            if robust:
                f_bin[i] = np.nanmedian(flux[j])
                e_bin[i] = 1.25*np.nanmean(abs(flux[j] - f_bin[i]))/np.sqrt(n)
            else:
                f_bin[i] = np.nanmean(flux[j])
                e_bin[i] = np.std(flux[j])/np.sqrt(n-1)

    j = (n_bin >= nmin)
    return t_bin[j], f_bin[j], e_bin[j], n_bin[j]
    
def gaussian_log_likelihood(residuals, variances):
    taus = 1. / variances
    return -0.5 * (len(residuals) * np.log(2*np.pi) + np.sum(-np.log(taus.astype(float)) + taus * (residuals**2)))

def CowanPC_model(times, te, per, E, C1, D1, C2, D2):
    omega_t = 2 * np.pi * (times - te) / per
    pc = E + ( C1 * (np.cos( omega_t ) - 1.) ) + ( D1 * np.sin( omega_t ) ) +\
             ( C2 * (np.cos( 2*omega_t ) - 1.) ) + ( D2 * np.sin( 2*omega_t ) )
    return pc

def evaluate_pytransit_CowanPC_model(times, fluxes, errors, fltr, per, bjd0, rprs, ar, mst, rst, vsini,\
                                         q1, q2, tpole, phi, lamp, bb1, E, C1, D1, C2, D2, mflx, sigw,\
                                         LIN=False, lin_vecs=None, thetas=None, GP=False, GP_S0=None, GP_Q=None, GP_rho=None, LTTD=True):
    
    if LTTD:
        ## This mean that we need to correct for the light travel time delay
        ### Computing inclination from bb and ar
        inc1 = np.arccos(bb1 / ar)
        term1 = 1 - np.cos(2 * np.pi * (times - bjd0) / per)
        abyc = ( ( ar * rst * u.R_sun / con.c ).to(u.d) ).value
        times = times - ( abyc * term1 * np.sin(inc1) )
    
    # Defining oblate star model!
    model = OblateStarModel(filters=fltr, rstar=2.365, model='husser2013', sres=100, pres=8, tmin=5000, tmax=10000)
    # Converting angles to radians
    phi, lamp = np.radians(phi), np.radians(lamp)
    
    # Computing rho and rper from the existing data
    ## rper
    rper = 2*np.pi*rst*con.R_sun/(vsini*(u.km/u.s))
    rper = rper * np.sin( (np.pi/2) - phi )
    rper = rper.to(u.day)
    rper = rper.value
    ## rho
    term1 = con.G * mst*con.M_sun * rper*u.day * rper*u.day/(2 * np.pi * np.pi * ((rst * con.R_sun)**3))
    term1 = term1.decompose()
    fstar = 1/(1 + term1)
    rho = 3 * np.pi/(2 * con.G * fstar * rper*u.day * rper*u.day)
    rho = rho.to(u.g/u.cm**3)
    rho = rho.value
   
    ## LDCs
    u1, u2 = juliet.utils.reverse_ld_coeffs('quadratic', q1, q2)
    ldcs = np.array([u1, u2])
    
    ## rprs to numpy.ndarray (as `pytransit` only takes array for k)
    rprs = np.array([rprs])

    ## Converting bb to inc1
    inc1 = np.arccos(bb1 / ar)

    # Now computing models
    ## Oblate star transit model
    model.rstar = rst*con.R_sun.value
    model.set_data(times)
    flux_tra = model.evaluate_ps(k=rprs, rho=rho, rperiod=rper, tpole=tpole, phi=phi, beta=0.22, ldc=ldcs,\
        t0=bjd0, p=per, a=ar, i=inc1, l=lamp, e=0., w=np.pi/2)
    
    ## Oblate star occultation model
    flux_om = model.evaluate_ps(k=rprs, rho=rho, rperiod=rper, tpole=tpole, phi=phi, beta=0., ldc=np.array([0., 0.]),\
        t0=bjd0+(per/2), p=per, a=ar, i=np.pi-inc1, l=-1*lamp, e=0., w=(np.pi/2)+np.pi)
    flx1 = flux_om-np.min(flux_om)
    flux_ecl = flx1/np.max(flx1)

    ## Phase curve (juliet-like; copied from juliet)
    sine_model = CowanPC_model(times=times, te=bjd0+(per/2), per=per, E=E, C1=C1, D1=D1, C2=C2, D2=D2)
    ### phase curve + eclipse model
    sine_ecl_model = 1. + sine_model * flux_ecl

    # Total physical model
    phy_model = sine_ecl_model*flux_tra
    
    # Normalising the flux
    flux_norm = phy_model / ( 1 + mflx )

    # Adding linear parameters if LIN=True
    if LIN:
        if lin_vecs.shape[0] != len(times):
            raise Exception('Please provide linear correlators in shape of ({:d},n)'.format(len(times)))
        else:
            lin_model = np.nansum(thetas * lin_vecs, axis=1)
    else:
        lin_model = np.zeros(len(times))

    # Deterministic model
    fl_deter = flux_norm + lin_model

    # Computing residuals
    resids = fluxes - fl_deter

    # Now, if GP=True, compute the GP model (and also, loglikelihood)
    vars = errors**2 + (sigw*1e-6)**2
    if GP:
        ker = SHOTerm(S0=GP_S0, Q=GP_Q, rho=GP_rho)
        gp = GaussianProcess(ker, mean=0.)
        gp.compute(times, diag=vars, quiet=True)
        
        loglike = gp.log_likelihood(resids)
        gp_model = gp.predict(y=resids, t=times, return_cov=False)
    else:
        gp_model = np.zeros(len(times))
        loglike = gaussian_log_likelihood(resids, vars)

    total_model = fl_deter + gp_model
    return phy_model, flux_norm, lin_model, gp_model, total_model, loglike

def GP_model_pred(times, pred_times, resids, detrended_vars, GP_S0, GP_Q, GP_rho):
    ker = SHOTerm(S0=GP_S0, Q=GP_Q, rho=GP_rho)
    gp = GaussianProcess(ker, mean=0.)
    gp.compute(times, diag=detrended_vars, quiet=True)
    
    gp_model = gp.predict(y=resids, t=pred_times, return_cov=False)

    return gp_model