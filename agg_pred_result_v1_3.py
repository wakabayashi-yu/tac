# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 17:34:54 2019

@author: wakabayashi
filename: agg_pred_result
予測結果の分析
"""
import sys
import os
import copy
import datetime as dt
from dateutil.relativedelta import relativedelta
import matplotlib.dates as mdates
import math
import pandas as pd
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import warnings
import matplotlib.pyplot as plt

"""
# 自作モジュール追加
myfunc_path = 'D:/Work/TAC/20191107'
sys.path.append(myfunc_path)

%matplotlib qt
%matplotlib inline
"""

plt.rcParams["font.family"] = "IPAexGothic"

warnings.simplefilter('ignore')

### my function ###
from common import calc_error_rate_stats

#%%
def calc_num_in_limit(err_rate, limits):
    """指定した範囲に入る件数の算出
    """
    columns = list(err_rate.columns)
    
    n = len(limits)
    
    cols = ['lower', 'upper']
    cols = cols + columns
    nums = np.concatenate([limits, np.zeros([n,len(columns)],int)], axis=1)
    nums = pd.DataFrame(nums, columns=cols) #記録領域
    
    for col_id in range(len(columns)):
        col_name = columns[col_id]
        
        for i in range(n):
            bool1 = limits[i][0] <= err_rate[col_name]
            bool2 = err_rate[col_name] < limits[i][1]
            nums[col_name].iloc[i] = sum(bool1 * bool2)
    return nums


###############################################################################
#%% initialize
root_dir = 'D:/Work/TAC/20191209'
#base_dir = os.path.join(root_dir, 'output/20191220_変数選択有b★')
base_dir = os.path.join(root_dir, 'Pred3monthAhead/output/20200115b')
output_dir = os.path.join(base_dir, '予測結果集計')
dir_path = copy.deepcopy(output_dir)
if os.path.isdir(dir_path) == False:
    os.makedirs(dir_path)

# データ読込 : 予測誤差率
fname = os.path.join(base_dir, 'err_rate.csv')
err_rate = pd.read_csv(fname, header=0, index_col=0, encoding='cp932')


#%% 誤差率の統計量の算出
valid_key = np.where(err_rate['内示誤差率 [%]'] != 0)[0]

forecast_err_rate = err_rate['内示誤差率 [%]'].iloc[valid_key]
AI_err_rate = err_rate['予測誤差率 [%]'].iloc[valid_key]

num = sum(forecast_err_rate.values >= AI_err_rate.values)
print('予測の方が高精度な品目数 =', num, '/', len(valid_key))


bins = 20
threshold = 100
key = np.where(err_rate['予測誤差率 [%]'] <= threshold)[0]
key = np.intersect1d(key, valid_key)

columns = ['内示誤差率 [%]', '予測誤差率 [%]']
tmp = err_rate[columns].iloc[key,:]
stats = calc_error_rate_stats(tmp, bins, threshold) #myfun

# save
fname = os.path.join(output_dir, 'err_rate_stats.csv')
stats.to_csv(fname, encoding='cp932')


### 誤差率が指定した範囲内に入る品目数の算出 ###
columns = ['内示誤差率 [%]', '予測誤差率 [%]']

limits = [[0,10],
         [10,15],
         [15,20],
         [20,30],
         [30,4000]]

tmp = err_rate[columns].iloc[valid_key,:]
nums = calc_num_in_limit(tmp, limits) #myfun

# save
fname = os.path.join(output_dir, 'num_in_ErrRateLimit.csv')
nums.to_csv(fname, encoding='cp932')


#%%
### 予測の方が高精度な場合の改善率
key = np.where(err_rate['内示誤差率 [%]'] >= err_rate['予測誤差率 [%]'])[0]
key = np.intersect1d(key, valid_key)

buffer = err_rate['内示誤差率 [%]'].iloc[key] - err_rate['予測誤差率 [%]'].iloc[key]
print(buffer.mean())


### 予測の方が低精度な場合の悪化率
key = np.where(err_rate['内示誤差率 [%]'] < err_rate['予測誤差率 [%]'])[0]
key = np.intersect1d(key, valid_key)
index = np.where(err_rate['予測誤差率 [%]'] <= 100)[0]
key = np.intersect1d(key, index)

buffer = - (err_rate['内示誤差率 [%]'].iloc[key] - err_rate['予測誤差率 [%]'].iloc[key])
print(buffer.mean())


#%% グラフ表示
font_size = 15
nbin = 20

columns = ['内示誤差率 [%]', '予測誤差率 [%]']
colors = {columns[0]:'b', columns[1]:'r'}
lists = [[0,1], [1,0]]

for i in range(len(columns)):
    col_name = columns[i]
    
    plt.figure(figsize=(12,7))
    for i in lists[i]:
        col_name = columns[i]
        error = err_rate[col_name].iloc[valid_key]
        #plt.hist(error[valid_key], bins=nbin, rwidth=0.8, range=(0,100), alpha=0.6, label=col_name, color=colors[col_name])
        plt.hist(error[valid_key], bins=nbin, rwidth=0.8, range=(0,100), label=col_name, color=colors[col_name])
    
    plt.rcParams["font.size"] = font_size
    plt.legend()
    plt.grid()
    plt.xlabel('平均誤差率 [%]')
    plt.ylabel('頻度')
    
    # グラフ保存
    fname = 'ErrorHist_' + col_name + '.png'
    fname = os.path.join(output_dir, fname)
    if 1:
        plt.savefig(fname)

#%%
for i in range(len(columns)):
    plt.figure(figsize=(12,7))
    col_name = columns[i]
    error = err_rate[col_name].iloc[valid_key]
    plt.hist(error[valid_key], bins=nbin, rwidth=0.8, range=(0,100), label=col_name, color=colors[col_name])
    
    plt.rcParams["font.size"] = font_size
    #plt.legend()
    plt.grid()
    plt.xlabel('平均誤差率 [%]')
    plt.ylabel('頻度')
    
    # グラフ保存
    fname = 'ErrorHist_v2_' + col_name + '.png'
    fname = os.path.join(output_dir, fname)
    if 1:
        plt.savefig(fname)

