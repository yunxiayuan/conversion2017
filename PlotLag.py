# -*- coding: utf-8 -*-
"""
Spyder Editor
    
This script is to analyze cumulative conversion numbers per delay(1-30).
"""
def conv_pred(filename, para=[8.4933783, -1.27472965e+2, -3.12001656, 8.15316654e-2]):
        
    import numpy as np
    import datetime
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    data = pd.read_csv(filename)
    
    # Convert date string to date type
    def str2date(x):
        return datetime.datetime.strptime(x,"%Y-%m-%d").date()
    
    #ReportDate = [str2date(d) for d in data['report_date']]
    #ClickDate = [str2date(d) for d in data['click_date']]
    ReportDate = list(map(str2date, data['report_date']))
    ClickDate = list(map(str2date, data['click_date']))
    delay = [a - b for a, b in zip(ReportDate, ClickDate)]
    delay = [a.days for a in delay]
    data['delay'] = delay
    
    # Calculate mean conversion number per delay
    data_group = data.groupby(['delay'])
    data_mean = data_group.mean()
    data_mean.reset_index(inplace = True)
        
    # Calculate incremental conversions
    c = np.array(data_mean['conversions'])
    data_mean['ic'] = np.concatenate((np.array([c[0]]), c[1:]-c[:-1]), axis=0)
    
    # fit function of conversion number on delay    
    def conv_lag(x, a, b, c):
        return a + b * np.exp(c*x)
    
    (par, cov) = curve_fit(conv_lag, data_mean["delay"], data_mean["ic"], p0=[3, 1, -0.2])
    print('Parameters are ')
    print(par)
    # fitted value of incremental conversion numbers
    ic = [conv_lag(x, par[0], par[1], par[2]) for x in data_mean["delay"]]
    # Plot data and fitted values
    plt.plot(data_mean['delay'], ic, '.-', label='fit')
    plt.plot(data_mean['delay'], data_mean['ic'], 'o', label='data')
    plt.legend(loc='best')
    plt.title('Incremental conversion numbers')
    plt.xlabel('delay: days')
    plt.ylabel('incremental conversion number')
    plt.show()
    
    # Evaluate the curve fitting
    res = [a - b for a, b in zip(ic, data_mean['ic'])] # residuals
    res_mean = np.mean(res)
    res_diff = res-res_mean
    res_rmse = np.sqrt(np.mean(res_diff**2))
    print('Residual bias is %f, RMSE is %f' % (res_mean, res_rmse))
    
    # Fit cumulative conversion numbers
    
    def log_fit(x, a, b, c, d):
        return a + b*np.exp(c*x) + d*x
    delay_array = np.array(data_mean['delay'])
    conv_array = np.array(data_mean['conversions']) 
    if len(conv_array) > 3: #p0=[17, -13.209, -0.507, 0.0357]
        (par2, ac) = curve_fit(log_fit, delay_array, conv_array, para)
    else:
        par2 = para[:]
    print('Parameters are ')
    print(par2)
    conv = [log_fit(x, par2[0], par2[1], par2[2], par2[3]) for x in delay_array]
    plt.figure()
    plt.plot(data_mean['delay'], conv, '.-', label='fit')
    plt.plot(data_mean['delay'], data_mean['conversions'], 'o', label='data')
    plt.legend()
    plt.title('Cumulative conversion numbers')
    plt.xlabel('delay: days')
    plt.ylabel('cumulative conversion number')
    plt.show()
    
    # Evaluate the curve fitting
    res2 = [a - b for a, b in zip(conv, data_mean['conversions'])] # residuals
    res_mean2 = np.mean(res2)
    res_diff2 = res2-res_mean2
    res_rmse2 = np.sqrt(np.mean(res_diff2**2))
    print('Residual bias is %f, RMSE is %f' % (res_mean2, res_rmse2))
     
import pandas as pd    
conv_pred("13572.csv.gz")
