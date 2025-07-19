import pywt
import numpy as np
from scipy.stats import norm, chi2
from modwt import modwt


def wavelet_variance(x, filters, level):
    wavelet = pywt.Wavelet(filters)
    filter_length = wavelet.rec_len
    w = modwt(x, filters, level)
    scale_length = (2**level-1)*(filter_length-1) + 1
    n_interior_coeffs = len(x) - scale_length + 1
    interior_coeffs = w[level-1][scale_length-1:]
    var = np.sum(interior_coeffs ** 2)/n_interior_coeffs
    return var

def wavelet_variance_CI(x, filters , level, alpha):
    var_estim = wavelet_variance(x, filters, level)
    wavelet = pywt.Wavelet(filters)
    filter_length = wavelet.rec_len
    w = modwt(x, filters, level)
    N = len(x)
    scale_length = (2**level-1)*(filter_length-1) + 1
    n_interior_coeffs = N - scale_length + 1
    total = 0.0
    for tau in range(1, n_interior_coeffs):
        indices = np.arange(scale_length - 1, N - tau)
        products = w[level-1][indices]*w[level-1][indices+tau]
        s_tau = np.sum(products)/(n_interior_coeffs)
        total += s_tau**2

    A = (var_estim**2)/2 + total
    

    
    df = n_interior_coeffs*(var_estim**2)/A
    
    U = var_estim*df / chi2.ppf(alpha/2, df)
    L =  var_estim*df / chi2.ppf(1-alpha/2, df)
    return [L, var_estim, U]




def wavelet_cross_covariance(x, y, filters, level, lag):
    wavelet = pywt.Wavelet(filters)
    filter_length = wavelet.rec_len
    

    w_x = modwt(x, filters, level)
    w_y = modwt(y, filters, level)
    
    # Compute Lj and Ñj
    Lj = (2**level - 1) * (filter_length - 1) + 1
    Nj_tilde = len(x) - Lj + 1 

    wj_x = w_x[level-1]
    wj_y = w_y[level-1]
    
    if 0 <= lag <= Nj_tilde - 1:
        t_start = Lj - 1
        t_end = len(x) - lag
        x_coeffs = wj_x[t_start:t_end]
        y_coeffs = wj_y[t_start + lag:t_end + lag]
    elif -Nj_tilde + 1 <= lag < 0:
        t_start = Lj - 1 - lag
        t_end = len(x)
        x_coeffs = wj_x[t_start:t_end]
        y_coeffs = wj_y[t_start + lag:t_end + lag]
    else:
        return 0.0


    cross_cov = np.sum(x_coeffs * y_coeffs) / Nj_tilde
    return cross_cov

def wavelet_correlation(x,y, filters, level, lag):
    cross_cov = wavelet_cross_covariance(x,y,filters, level, lag)
    sig_x = np.sqrt(wavelet_variance(x,filters,level))
    sig_y = np.sqrt(wavelet_variance(y, filters, level))
    correlaiton = cross_cov/(sig_y*sig_x)
    return correlaiton

def wavelet_correlation_CI(x,y, filters, level, lag, alpha):
    wavelet = pywt.Wavelet(filters)
    filter_length = wavelet.rec_len
    Lj = (2**level - 1) * (filter_length - 1) + 1
    Nj_tilde = Nj_tilde = len(x) - Lj + 1 - lag
    correlation = wavelet_correlation(x,y, filters, level, lag)
    z = norm.ppf(1 - alpha / 2)
    U = np.tanh(np.arctanh(correlation)+z*np.sqrt(1/(Nj_tilde-3)))
    L = np.tanh(np.arctanh(correlation)-z*np.sqrt(1/(Nj_tilde-3)))
    return [float(L),float(correlation),float(U)]
