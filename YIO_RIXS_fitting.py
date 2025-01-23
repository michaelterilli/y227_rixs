# for the fitting procedures
from lmfit import *

# importing the workhorse
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# import all plotting packages
import matplotlib.pyplot as plt
from matplotlib import colors, cm, animation
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

def gaussian(params, x, y):
    '''
    1-d gaussian: gaussian(x, area, cen, fwhm, bg)
    '''
    area = params['area']
    cen = params['cen']
    fwhm = params['fwhm']
    bg = params['bg']
    
    sig = fwhm/(2*np.sqrt(2*np.log(2)))
    model = (area/(np.sqrt(2*np.pi)*sig))*np.exp(-(x-cen)**2 /(2*sig**2))+bg
    return model-y

def fit_elastic(x, y, el_pk, el_range, plot_fit=False, print_fit=False):
    '''
    el_pk: Initial guess of the elastic line
    el_wid: Initial guess of the elastic peak width (FWHM)
    '''
    # create a set of Parameters
    params = Parameters()
    #params.add('offset', value=1.)
    params.add('area', value=3e-5)
    params.add('cen', value=el_pk)
    #params.add('fwhm', value=35, min=10) #in meV. This can be used in Eloss data
    params.add('fwhm', value=0.035, min=1e-3) #in eV. This can be used in incident energy data
    params.add('bg', value=0., min=0.)

    # do fit, here with leastsq model
    subx = x[np.where((x>el_pk-el_range) & (x<el_pk+el_range))]
    suby = y[np.where((x>el_pk-el_range) & (x<el_pk+el_range))]
    
    minner = Minimizer(gaussian, params, fcn_args=(subx, suby))
    result = minner.minimize()

    # calculate final result
    final = suby + result.residual

    # write error report
    if print_fit:
        report_fit(result)

    # try to plot results
    try:
        if plot_fit:
            plt.figure()
            plt.plot(subx, suby, 'ko')
            plt.plot(subx, final, 'r-')
            plt.show()
    except ImportError:
        pass
    
    return final, result


#One Gauss (elastic) + Two anti-Lor (one magnon + one continuum) + bg fit 
def rixsALor_2(params, x, y):
    '''
    Magnon' dispersion for RIXS

    '''
    A0 = params['El_area']
    w0 = params['El_wid']
    x0 = params['El_pk']

    A1 = params['Pk1_area']
    x1 = params['Pk1_pk']
    w1 = params['Pk1_wid']
    
    A2 = params['Pk2_area']
    x2 = params['Pk2_pk']
    w2 = params['Pk2_wid']
    
    C0 = params['bg']
    
    sig0 = w0/(2*np.sqrt(2*np.log(2)))
    elpk = A0*np.exp(-(x-x0)**2/2/sig0**2)/sig0/np.sqrt(2*np.pi)
    
    antiLor1 = A1*(w1/2)/np.pi/((x-x0-x1)**2+(w1/2)**2)-A1*(w1/2)/np.pi/((x-x0+x1)**2+(w1/2)**2)
    aLor1 = np.where(antiLor1<0, 0, antiLor1)
    
    antiLor2 = A2*(w2/2)/np.pi/((x-x0-x2)**2+(w2/2)**2)-A2*(w2/2)/np.pi/((x-x0+x2)**2+(w2/2)**2)
    aLor2 = np.where(antiLor2<0, 0., antiLor2)
    
    bg = C0
    model = elpk+aLor1+aLor2+bg
    
    return model-y


def fit_rixsALor_2(x, y, plot_fit=False):
    # create a set of Parameters
    params = Parameters()
    params.add('El_area', value=2e-3, min=0.)
    params.add('El_wid', value=20)
    params.add('El_pk', value=0, vary=False)
    
    params.add('Pk1_area', value=1e-4, min=0.)
    params.add('Pk1_pk', value=30, min=0)
    params.add('Pk1_wid', value=40, min=35)
    
    params.add('Pk2_area', value=5e-5, min=0.)
    params.add('Pk2_pk', value=73, min=50., max = 200)
    params.add('Pk2_wid', value=100, min=35)
    
    
    params.add('bg', value=1e-7, max=1e-6)
    
    # do fit, here with leastsq model
    
    minner = Minimizer(rixsALor_2, params, fcn_args=(x, y))
    result = minner.minimize()

    # calculate final result
    final = y + result.residual

    # write error report
    report_fit(result)

    # try to plot results
    try:
        if plot_fit:
            plt.figure()
            plt.plot(x, y, 'ko')
            plt.plot(x, final, 'r-')
            plt.plot(x, result.residual, 'g-')
            plt.show()
    except ImportError:
        pass
    
    return final, result

#One Gauss (elastic) + Two anti-Lor (two magnons + one continuum) +bg fit 
def rixsALor_3(params, x, y):
    '''
    Magnon' dispersion for RIXS

    '''
    A0 = params['El_area']
    w0 = params['El_wid']
    x0 = params['El_pk']

    A1 = params['Pk1_area']
    x1 = params['Pk1_pk']
    w1 = params['Pk1_wid']
    
    A2 = params['Pk2_area']
    x2 = params['Pk2_pk']
    w2 = params['Pk2_wid']
    
    A3 = params['Pk3_area']
    x3 = params['Pk3_pk']
    w3 = params['Pk3_wid']
    
    C0 = params['bg']
    
    sig0 = w0/(2*np.sqrt(2*np.log(2)))
    elpk = A0*np.exp(-(x-x0)**2/2/sig0**2)/sig0/np.sqrt(2*np.pi)
    
    antiLor1 = A1*(w1/2)/np.pi/((x-x0-x1)**2+(w1/2)**2)-A1*(w1/2)/np.pi/((x-x0+x1)**2+(w1/2)**2)
    aLor1 = np.where(antiLor1<0, 0, antiLor1)
    
    antiLor2 = A2*(w2/2)/np.pi/((x-x0-x2)**2+(w2/2)**2)-A2*(w2/2)/np.pi/((x-x0+x2)**2+(w2/2)**2)
    aLor2 = np.where(antiLor2<0, 0., antiLor2)
    
    antiLor3 = A3*(w3/2)/np.pi/((x-x0-x3)**2+(w3/2)**2)-A3*(w3/2)/np.pi/((x-x0+x3)**2+(w3/2)**2)
    aLor3 = np.where(antiLor3<0, 0., antiLor3)
    
    bg = C0
    model = elpk+aLor1+aLor2+aLor3+bg
    
    return model-y


def fit_rixsALor_3(x, y, plot_fit=False):
    # create a set of Parameters
    params = Parameters()
    params.add('El_area', value=1.5e-3, min=0.)
    params.add('El_wid', value=30)
    params.add('El_pk', value=0, vary=False)
    
    params.add('Pk1_area', value=1e-4, min=0.)
    params.add('Pk1_pk', value=30, min=20, max=40)
    params.add('Pk1_wid', value=40, min=35, max=50)
    
    params.add('Pk2_area', value=1e-4, min=0.)
    params.add('Pk2_pk', value=50, min=27, max=60)
    params.add('Pk2_wid', value=40, min=35, max=50)
    
    params.add('Pk3_area', value=5e-5, min=0.)
    params.add('Pk3_pk', value=73, min=0., max = 200)
    params.add('Pk3_wid', value=100, min=35)
    
    
    params.add('bg', value=1e-7, max=2e-6)
    
    # do fit, here with leastsq model
    
    minner = Minimizer(rixsALor_3, params, fcn_args=(x, y))
    result = minner.minimize()

    # calculate final result
    final = y + result.residual

    # write error report
    report_fit(result)

    # try to plot results
    try:
        if plot_fit:
            plt.figure()
            plt.plot(x, y, 'ko')
            plt.plot(x, final, 'r-')
            plt.plot(x, result.residual, 'g-')
            plt.show()
    except ImportError:
        pass
    
    return final, result