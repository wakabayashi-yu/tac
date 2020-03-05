# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 18:51:53 2019

@author: wakabayashi
filename: pred_tac
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
from sklearn.model_selection import train_test_split
import pywt
#from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler #正規化
from sklearn.metrics import r2_score
import statsmodels.api as sm #トレンド + 季節成分に分解
#from sklearn.linear_model import LinearRegression #線形回帰
from sklearn.linear_model import RidgeCV #ridge回帰
from sklearn.linear_model import LassoLarsCV #lasso回帰
from sklearn.linear_model import LassoLarsIC #lasso回帰
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor as RFR #random forest
from xgboost import XGBRegressor #xgboost


#%% import my function
from common import calc_mean_error_rate
from common import conv_timestamp
#from common import make_subplot
#from common import gen_timestamp
from common import get_overlap_item_idxs
from common import make_no_overlap_data
from common import make_subplot
from common import get_comb_key

"""
# 自作モジュール追加
myfunc_path = 'D:/Work/TAC/20191107'
sys.path.append(myfunc_path)

%matplotlib qt
%matplotlib inline
"""

plt.rcParams["font.family"] = "IPAexGothic"

warnings.simplefilter('ignore')


#%%
def train_test_ridge(input_data, output_data, train_key, test_key, n_cv=3):
    """
    ridge回帰による学習/予測
    """
    # set parameter
    #alphas = 10 ** np.arange(-3, 1, 0.1)
    alphas = 10 ** np.arange(-2, 1, 0.1)
    
    # 例外処理 : 学習データ点数が分割数より少ない場合
    if len(train_key) < n_cv:
        n_cv = len(train_key)
    
    #-------------
    # 学習
    #-------------
    x = input_data[train_key,:]
    y = output_data[train_key]
    
    # インスタンス
    x_scaler = StandardScaler() #正規化
    y_scaler = StandardScaler() #正規化
    
    clf = RidgeCV(alphas=alphas, cv=n_cv)
    
    # モデル構築
    x_scaler.fit(x) #正規化
    y_scaler.fit(y.reshape(-1,1)) #正規化
    
    y_ = y_scaler.transform(y.reshape(-1,1))
    y_ = y_.reshape(-1)
    
    #import pdb; pdb.set_trace()
    
    # モデル構築
    clf.fit(x_scaler.transform(x), y_)
    
    # モデルパラメータ取得
    #alpha = clf.alpha_ #ハイパーパラメータ
    a = clf.coef_ #係数
    b = clf.intercept_ #切片
    p = np.append(a, b)
    
    #-------------
    # 予測
    #-------------
    x = input_data[test_key,:]
    
    # 例外処理 : xのデータ点数 = 1の場合 ⇒配列を整形
    if x.ndim == 1:
        x = x.reshape(1,-1)
    
    # 予測
    tmp = clf.predict(x_scaler.transform(x))
    y_pred = y_scaler.inverse_transform(tmp) #非正規化
    return y_pred, p


#%%
def train_test_lasso(input_data, output_data, train_key, test_key, n_cv=3):
    """
    lasso回帰による学習/予測
    """
    # 例外処理 : 学習データ点数が分割数より少ない場合
    if len(train_key) < n_cv:
        n_cv = len(train_key)
    
    #-------------
    # 学習
    #-------------
    x = input_data[train_key,:]
    y = output_data[train_key]
    
    # インスタンス生成
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    clf = LassoLarsCV(cv=n_cv, positive=True)
    #clf = LassoLarsCV(cv=n_cv, positive=True)
    #clf = LassoLarsIC(criterion='bic', positive=True)
    
    # モデル構築
    x_scaler.fit(x) #正規化
    y_scaler.fit(y.reshape(-1,1)) #正規化
    
    y_ = y_scaler.transform(y.reshape(-1,1))
    y_ = y_.reshape(-1)
    
    #import pdb; pdb.set_trace()
    
    #error_flag = 0 #初期化
    try:
        clf.fit(x_scaler.transform(x), y_)
    except ValueError:
        clf = LassoLarsIC(criterion='bic', positive=True)
        clf.fit(x_scaler.transform(x), y_)
    
    
    # モデルパラメータ取得
    #alpha = clf.alpha_ #ハイパーパラメータ
    a = clf.coef_ #係数
    b = clf.intercept_ #切片
    p = np.append(a, b)
    
    #-------------
    # 予測
    #-------------
    x = input_data[test_key,:]
    
    # 例外処理 : xのデータ点数 = 1の場合 ⇒配列を整形
    if x.ndim == 1:
        x = x.reshape(1,-1)
    
    # 予測
    tmp = clf.predict(x_scaler.transform(x))
    y_pred = y_scaler.inverse_transform(tmp) #非正規化
    return y_pred, p

#%%
def train_test_en(input_data, output_data, train_key, test_key, n_cv=3):
    """
    elastic net回帰による学習/予測
    """
    # set parameter
    #alphas = 10 ** np.arange(-2, 1, 0.1)
    
    # 例外処理 : 学習データ点数が分割数より少ない場合
    if len(train_key) < n_cv:
        n_cv = len(train_key)
    
    #-------------
    # 学習
    #-------------
    x = input_data[train_key,:]
    y = output_data[train_key]
    
    # インスタンス
    x_scaler = StandardScaler() #正規化
    y_scaler = StandardScaler() #正規化
    
    clf = ElasticNetCV(l1_ratio=[.05, .15, .5, .7, .9, .95, .99, 1], n_jobs=8, n_alphas=20, cv=n_cv)
    
    # モデル構築
    x_scaler.fit(x) #正規化
    y_scaler.fit(y.reshape(-1,1)) #正規化
    
    y_ = y_scaler.transform(y.reshape(-1,1))
    y_ = y_.reshape(-1)
    
    #import pdb; pdb.set_trace()
    
    # モデル構築
    with warnings.catch_warnings(): #警告無視
        warnings.simplefilter("ignore")
        clf.fit(x_scaler.transform(x), y_)
    
    # モデルパラメータ取得
    #alpha = clf.alpha_ #ハイパーパラメータ
    a = clf.coef_ #係数
    b = clf.intercept_ #切片
    p = np.append(a, b)
    
    #-------------
    # 予測
    #-------------
    x = input_data[test_key,:]
    
    # 例外処理 : xのデータ点数 = 1の場合 ⇒配列を整形
    if x.ndim == 1:
        x = x.reshape(1,-1)
    
    # 予測
    tmp = clf.predict(x_scaler.transform(x))
    y_pred = y_scaler.inverse_transform(tmp) #非正規化
    return y_pred, p

#%%
def train_test_randomForest(input_data, output_data, train_key, test_key, n_cv=3):
    """
    ランダムフォレスト回帰による学習/予測
    """
    # 例外処理 : 学習データ点数が分割数より少ない場合
    if len(train_key) < n_cv:
        n_cv = len(train_key)
    
    #-------------
    # 正規化
    #-------------
    x = input_data[train_key,:]
    y = output_data[train_key]
    
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    x_scaler.fit(x) #正規化
    y_scaler.fit(y.reshape(-1,1)) #正規化
    
    y_ = y_scaler.transform(y.reshape(-1,1))
    y_ = y_.reshape(-1)
    
    #-------------
    # 学習
    #-------------
    # パラメータ範囲指定
    clf = RFR(n_jobs=1, random_state=2525)
    
    #params = {"max_depth": [None, 2, 3],
    #          "n_estimators":[5, 10, 20],
    #          #"max_features": [1, 3],
    #          "min_samples_split": [2, 3],
    #          "min_samples_leaf": [1, 3],
    #          "bootstrap": [True, False]}
    
    params = {"max_depth": [None, 2, 3],
              "n_estimators":[5, 10],
              "min_samples_split": [2, 3],
              "min_samples_leaf": [1, 3],
              "bootstrap": [True, False]}
    
    cv = GridSearchCV(clf, params, cv=n_cv, scoring=None, n_jobs=1)
    
    #import pdb; pdb.set_trace()
    
    clf.fit(x_scaler.transform(x), y_)
    
    p = clf.feature_importances_
    
    #-------------
    # 予測
    #-------------
    x = input_data[test_key,:]
    
    # 例外処理 : xのデータ点数 = 1の場合 ⇒配列を整形
    if x.ndim == 1:
        x = x.reshape(1,-1)
    
    # 予測
    tmp = clf.predict(x_scaler.transform(x))
    y_pred = y_scaler.inverse_transform(tmp) #非正規化
    return y_pred, p


#%%
def train_test_xgb(input_data, output_data, train_key, test_key, n_cv=5):
    """
    xgboost回帰による学習/予測
    """
    # 例外処理 : 学習データ点数が分割数より少ない場合
    if len(train_key) < n_cv:
        n_cv = len(train_key)
    
    #-------------
    # 正規化
    #-------------
    x = input_data[train_key,:]
    y = output_data[train_key]
    
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    x_scaler.fit(x) #正規化
    y_scaler.fit(y.reshape(-1,1)) #正規化
    
    y_ = y_scaler.transform(y.reshape(-1,1))
    y_ = y_.reshape(-1)
    
    #-------------
    # 学習
    #-------------
    # パラメータ範囲指定
    params = {'learning_rate': [0.05, 0.1], 'max_depth': [3, 5, 10, 100], 'subsample': [0.8, 0.85, 0.9, 0.95], 'colsample_bytree': [0.5, 1.0]}
    #params = {'learning_rate': [0.05, 0.1], 'colsample_bytree': [0.5, 1.0]}
    #params = {'reg_lambda': 10**np.arange(-2, 1, 0.5), 'learning_rate': [0.05, 0.1], 'colsample_bytree': [0.5, 1.0]}
    #params = {'reg_lambda': 10**np.arange(-2, 1, 0.5), 'learning_rate': [0.05, 0.1]}
    clf = XGBRegressor(objective='reg:squarederror')
    
    cv = GridSearchCV(clf, params, cv=n_cv, scoring=None, n_jobs=1)
    
    #import pdb; pdb.set_trace()
    
    clf.fit(x_scaler.transform(x), y_)
    
    #-------------
    # 予測
    #-------------
    x = input_data[test_key,:]
    
    # 例外処理 : xのデータ点数 = 1の場合 ⇒配列を整形
    if x.ndim == 1:
        x = x.reshape(1,-1)
    
    # 予測
    tmp = clf.predict(x_scaler.transform(x))
    y_pred = y_scaler.inverse_transform(tmp) #非正規化
    return y_pred


# %% 
def plot_time_series(fname, t, y, y_pred, ref_data, yabel_name, xlimit=None, interval=1, fontsize=14, save_flag=1):
    """
    時系列プロット
    実績、内示、予測の3種類の時系列をプロット
    """
    tmp = np.concatenate([y, ref_data, y_pred], 0)
    ylimit = [0, max(tmp)]
    #ylimit = [min(tmp), max(tmp)]
    #ylimit = [0.5*min(tmp), max(tmp)]
    
    fig, ax = plt.subplots(figsize=(12,7))
    
    ax.plot(t, y, 'ko-') #実績
    ax.plot(t, ref_data, 'bo-') #内示
    ax.plot(t, y_pred, 'ro-') #予測
    
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval, tz=None))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m"))
    
    plt.setp(ax.get_xticklabels(), rotation=45) #x軸目盛回転
    
    ax.grid()
    
    if xlimit != None:
        plt.xlim(xlimit)
    plt.ylim(ylimit)
    plt.rcParams["font.size"] = fontsize
    plt.ylabel(yabel_name)
    
    # グラフ保存
    if save_flag:
        plt.savefig(fname)

#%%
def output_result_detail(fname, actual, forecast, prediction, test_key):
    """結果明細の出力
    """
    columns = ['timestamp', 'test_flag', '実績', 'N+2月内示', '予測', '内示誤差率 [%]', '予測誤差率 [%]']
    n = len(actual) #データ長
    res = pd.DataFrame(np.zeros([n,len(columns)]), columns=columns, index=np.arange(1,n+1)) #記録領域
    
    res['timestamp'] = actual.index
    res['test_flag'].iloc[test_key] = 1
    
    res['実績'] = actual.values
    res['N+2月内示'] = forecast.values
    
    # 予測データの整形
    tmp = np.zeros(n)
    tmp[test_key] = prediction
    res['予測'] = tmp
    
    # 誤差率の計算
    res['内示誤差率 [%]'] = abs((actual.values - forecast.values) / actual.values) * 100
    res['予測誤差率 [%]'].iloc[test_key] = abs((actual.iloc[test_key].values - res['予測'].iloc[test_key].values) / actual.iloc[test_key].values) * 100
    
    ### save ###
    res.to_csv(fname, encoding='cp932')
    
    return res


#%%
def get_high_corr_var_idx(output_data, corr_abs_flag=1, threshold=0.8, n_in=10):
    """
    相関係数を計算し、相関係数の大きい変数のインデックスを取得する。
    """
    ### 相関係数の算出 ###
    N = output_data.shape[0] #品目数
    
    corrmat = np.zeros(N) #記録領域
    
    # 品目の指定
    for j in range(N):
        data = output_data.iloc[[output_id,j],:].T
        
        corrmat[j] = np.corrcoef(data.iloc[:,0], data.iloc[:,1])[0,1]
    
    corrmat[output_id] = 0 #自分自身の相関係数=0
    
    
    ### 相関の高い品目の選択 ###
    if corr_abs_flag == 1:
        input_idx = np.where(abs(corrmat) >= threshold)[0]
    elif corr_abs_flag == 0:
        input_idx = np.where(corrmat >= threshold)[0]
    
    # 例外処理 : 相関の強い説明変数が存在しなかった場合
    if len(input_idx) == 0:
        if corr_abs_flag == 1:
            input_idx = np.argsort(abs(corrmat))[::-1]
        elif corr_abs_flag == 0:
            input_idx = np.argsort(corrmat)[::-1]
        
        # tmp = corrmat[input_idx]
        input_idx = input_idx[0:n_in]
        print('例外処理 : 相関の強い説明変数が存在しなかった場合')
    return corrmat, input_idx


#%% 
def get_forecast(OutputNames, y, forecast):
    """
    内示の取得
    """
    for i in range(len(OutputNames)):
        output_name = OutputNames[i]
        index = np.where(output_name == forecast['品番'].values)[0]
        
        tmp = forecast[['年月', 'N+2月']].iloc[index,:]
        
        tmp = pd.DataFrame(tmp['N+2月'].values, columns=[output_name], index=tmp['年月'])
        
        if i == 0:
            input_data = copy.deepcopy(tmp)
        else:
            input_data = pd.concat([input_data, tmp], axis=1)
    
    # タイムスタンプを一致させる
    input_data = pd.concat([y, input_data], axis=1, join_axes=[y.index])
    input_data = input_data.iloc[:,1:input_data.shape[1]]
    
    return input_data


#%% 
def get_r2_score_var_idx(y, input_data, n_cv=3, n_disp=10):
    """
    目的変数と類似する品目候補の検索 : 決定係数による検索
    """
    N = input_data.shape[1] #品目数
    
    valid_key = np.where(y.values != 0)[0]
    train_key, test_key = train_test_split(valid_key, test_size=0.3, shuffle=False)
    
    scores = pd.Series(np.zeros(N), index=input_data.columns, name='決定係数') #記録領域
    for i in range(N):
        if i%n_disp == 0:
            print('NO.', str(i), '/', str(N))
        
        x = input_data.iloc[:, [i]]
        
        y_pred, p = train_test_ridge(x.values, y.values, train_key, test_key, n_cv)
        
        scores.iloc[i] = r2_score(y.iloc[test_key].values, y_pred)
    
    return scores


#%%
def gen_lag_data(output_data, n_lag=[0,8]):
    """過去の目的変数を説明変数に追加
    """
    t = output_data.index
    n = len(t) #データ長
    
    # 記録領域
    n_col = n_lag[1] - n_lag[0]
    X = np.zeros([n, n_col])
    columns = []
    
    cnt = 0 #初期化
    
    # ラグの指定
    for lag in range(n_lag[0], n_lag[1]):
        for k in range(n):
            t_old = t[k] - relativedelta(months=lag)
            idx = np.where(t_old == t)[0]
            
            # ラグと一致するタイムスタンプを持つレコードが見つかった場合
            if len(idx) != 0:
                X[k,cnt] = output_data.iloc[idx]
        
        cnt = cnt + 1 #更新
        columns.append('lag' + str(lag)) #store
    
    X = pd.DataFrame(X, columns=columns, index=t)
    return X

#%%
def rmse(predictions, targets):
    """rmseの計算
    """
    return np.sqrt(((predictions - targets) ** 2).mean()) 

#%%
def select_var_dim(input_data, y, train_idx, test_idx, var_pattern='x', n_cv=2):
    """変数の次元選択
    """
    col_idx = [] #記録領域
    for i in range(input_data.shape[1]):
        if var_pattern in input_data.columns[i]:
            col_idx.append(i)
    
    n_lag = len(col_idx)
    scores = np.zeros(n_lag) #記録領域
    for lag in range(0,n_lag):
        x = input_data.iloc[:,col_idx[0:lag+1]]
        ym, _ = train_test_ridge(x.values, y, train_idx, test_idx, n_cv) #myfun
        
        # RMSEの計算
        scores[lag] = rmse(ym, y[test_idx]) #myfun
    
    idx = scores.argmin()
    col_idx = col_idx[0:idx+1]
    return col_idx, scores

#%%
def make_set_item_data(output_dir, all_data, item_tbl, save_flag=1):
    """
    指定した商品コードが出荷実績データに存在するか確認
    """
    num = item_tbl_o.shape[0]
    
    # 記録領域
    index = []
    item_key = []
    
    for i in range(num):
        idx = np.where(item_tbl['品番'].iloc[i] == all_data['商品コード'].values)[0]
        if len(idx) != 0:
            index.append(i)
            item_key.append(idx[0])
    
    item_tbl_new = item_tbl.iloc[index]
    
    data = all_data.iloc[item_key,:]
    data.index = np.arange(data.shape[0])
    
    if save_flag == 1:
        # save : 紐付け対応表
        fname = os.path.join(output_dir, '紐付対応表.csv')
        item_tbl_new.to_csv(fname, encoding='cp932')
        
        # save : 納入実績
        fname = os.path.join(output_dir, '納入実績.csv')
        data.to_csv(fname, encoding='cp932')
    
    return data, item_tbl_new, item_key, index

#%%
def select_output_var(data, threshold=20):
    """
    目的変数の選択
    指定した条件で対象品目を抽出
    """
    N = data.shape[0] #品目数
    columns = ['出荷数量一定以上の月数', '出荷月数']
    stats = pd.DataFrame(np.zeros([N,len(columns)]), columns=columns) #記録領域
    col_name = '商品コード'
    stats.insert(0, col_name, output_data_o[col_name].values)
    
    
    ### 出荷頻度が一定以上の品目を検索 ###
    stats['出荷月数'] = np.sum(data.values != 0, axis=1)
    
    index1 = np.where(stats['出荷月数'].values >= threshold)[0]
    
    ### 出荷数量が一定以上の月数が一定上存在する品目を検索 ###
    output_list = [] #記録領域
    for i in range(len(index1)):
        x = data.iloc[index1[i],:]
        if np.sum(x.values >= 10) >= threshold:
            output_list.append(i)
    
    output_list = index1[output_list]
    
    return output_list, stats

#%%
def GetIdx_including_SettingCarType(item_tbl, input_data, col_name='車種1'):
    """
    車種名のリストを生成。生成したリストと一致する車種を持つレコードのインデックスを取得する。
    """
    # 車種のリスト生成
    car_types = []
    for i in range(item_tbl.shape[0]):
        if col_name in item_tbl.index[i]:
            car_type = item_tbl.iloc[i]
            
            if car_type != '':
                car_types.append(car_type)
    
    index = []
    for i in range(len(car_types)):
        idx = np.where(car_types[i] == input_data[col_name].values)[0]
        index = index + list(idx)
    
    index = np.sort(np.unique(index))
    
    return index, car_types

#%%
def Gen_InputData_from_AllCarproductSchedule(input_data_o, item_tbl):
    """
    車種生産計画の全データから、説明変数を抽出する
    """
    #############
    # 品番が紐付く車種情報の生成
    #############
    # 車種名のリストを生成。
    # 生成したリストと一致する車種を持つレコードのインデックスを取得する。
    index1, car_type1 = GetIdx_including_SettingCarType(item_tbl.iloc[0,:], input_data_o, '車種1') #myfun
    
    index2, car_type2 = GetIdx_including_SettingCarType(item_tbl.iloc[0,:], input_data_o, '車種2') #myfun
    
    if len(car_type2) == 0:
        index = copy.deepcopy(index1)
    else:
        index = np.intersect1d(index1, index2)
    
    input_data_ = input_data_o.iloc[index,:]
    
    
    #############
    # 指定した車種1,2の生産計画を抽出し、説明変数とする
    #############
    columns = ['車種1', '車種2']
    
    #import pdb; pdb.set_trace() #デバッグ
    
    comb_car_type = get_comb_key(input_data_[columns].values) #myfun
    comb_car_type = pd.DataFrame(comb_car_type, columns=columns)
    
    # 車種1,2の組合せの指定
    for i in range(comb_car_type.shape[0]):
        # 指定した車種1,2を持つレコードのインデックス
        col_name = '車種1'
        idx1 = np.where(comb_car_type[col_name].iloc[i] == input_data_[col_name].values)[0]
        
        col_name = '車種2'
        idx2 = np.where(comb_car_type[col_name].iloc[i] == input_data_[col_name].values)[0]
        
        index = np.intersect1d(idx1,idx2)
        x_ = input_data_.iloc[index,:]
        
        #columns = ['N+2月市販', 'N+2月輸出']
        columns = ['生産計画']
        x_ = x_[columns]
        
        # カラム名の変更
        for j in range(len(columns)):
            #columns[j] = comb_car_type.iloc[i,0] + '_' + comb_car_type.iloc[i,1] + '_' + columns[j]
            columns[j] = comb_car_type.iloc[i,0] + '_' + comb_car_type.iloc[i,1]
        x_.columns = columns
        
        if i == 0:
            x = copy.deepcopy(x_)
        else:
            x = pd.concat([x, x_], axis=1)
    
    input_data = copy.deepcopy(x)
    
    return input_data, input_data_, comb_car_type


#%%
def agg_540A_341B(data):
    """
    車種名に'540A', '341B'が含まれる場合、車種名を置換し、集約
    """
    columns = list(data.columns.values)
    
    # 車種名を置換
    columns = [columns[i].replace('540A', '341B') for i in range(len(columns))]
    
    uniques = list(np.unique(columns)) #ユニークなカラム名
    
    n = data.shape[0] #データ長
    dim = len(uniques)
    
    new_data = pd.DataFrame(np.zeros([n,dim]), columns=uniques, index=data.index) #記録領域
    
    for i in range(dim):
        col_name = uniques[i]
        
        index = np.where(col_name == np.array(columns))[0]
        
        #import pdb; pdb.set_trace()
        
        if len(index) == 1:
            new_data[col_name] = data[col_name].values
        elif len(index) >= 2:
            tmp = data.iloc[:,index].sum(axis=1)
            new_data[col_name] = tmp.values
        else:
            print('ERROR!')
    
    return new_data

#%%
def var_select_based_contr_rate(p, threshold=1):
    """
    #elastic netの変数の寄与率による変数選択
    説明変数の選択
    """
    ### 寄与率の高い変数の選択 ###
    #_, p = train_test_en(x.iloc[:,col_idx1].values, y.values, train_key1, 0, 3)
    #_, p = train_test_en(x, y, train_key1, 0, n_cv)
    
    # 寄与率の算出
    p_rate = p[0:-1]**2
    p_rate = p_rate / sum(p_rate) * 100
    
    # 寄与率の大きい変数が存在する場合 ⇒その変数のみを利用する
    col_idx = np.where(threshold < p_rate)[0] # 寄与率一定以上の変数のみ利用する
    return col_idx, p_rate

#%%
def get_1year_before_data(output_data):
    """同月の1年前のデータを説明変数に追加
    """
    t = output_data.index
    
    # 月を取得
    months = [t[k].month for k in range(len(t))]
    months = np.array(months)
    
    n = len(months) #データ長
    columns = ['同月1年前', '同月1年前_mean', '同月1年前_max']
    x = pd.DataFrame(np.zeros([n,3]), columns=columns, index=output_data.index) #記録領域
    for k in range(n):
        index = np.where(months[k] == months[0:k])[0]
        if len(index) != 0:
            index = index[-1]
            x['同月1年前'].iloc[k] = output_data.iloc[index]
            x['同月1年前_mean'].iloc[k] = output_data.iloc[index-1:index+2].mean()
            x['同月1年前_max'].iloc[k] = output_data.iloc[index-1:index+2].max()
    return x

#%%
def gen_up_down_state(output_data, threshold=8):
    """stay / up / downの状態
    """
    dy = output_data.diff() / output_data.values * 100
    
    n = len(output_data)
    bools = np.zeros(n) #記録領域
    bools[bools == 0] = np.nan
    
    # up
    index = np.where(dy >= threshold)[0]
    bools[index] = np.ones(len(index))
    
    # down
    index = np.where(dy <= -threshold)[0]
    bools[index] = -1 * np.ones(len(index))
    
    # stay
    index = np.where(abs(dy) <= threshold)[0]
    bools[index] = 0 * np.ones(len(index))
    
    bools = pd.Series(bools, name='up_down_label', index=output_data.index) #記録領域
    return bools

#%%
def count_up_down_state(x, windows=[2,3,4]):
    """直近Xヶ月間のstay / up / downの回数
    """
    n = len(x)
    
    feature_labels = []
    for i in range(len(windows)):
        col_name = [x.name, 'w' + str(windows[i])]
        feature_labels.append(col_name)
    
    columns = [feature_labels[i][0] + '_' + feature_labels[i][1] for i in range(len(feature_labels))]
    
    feature = pd.DataFrame(np.zeros([n, len(windows)],int), columns=columns, index=x.index) #記録領域
    for i in range(len(windows)):
        w = windows[i]
        tmp = x.fillna(0)
        feature.iloc[:,i] = tmp.rolling(w).sum()
    
    return feature, feature_labels

#%%
def mydwt(y):
    """離散ウェーブレット分解
    """
    ca, cd = pywt.dwt(y.values, 'db1')
    
    n = len(y)
    columns = ['dwt_approx', 'dwt_detail']
    y_dwt = pd.DataFrame(np.zeros([n,len(columns)]), columns=columns, index=y.index)
    
    y_dwt['dwt_approx'] = pywt.upcoef('a', ca, 'db1')
    y_dwt['dwt_detail'] = pywt.upcoef('d', cd, 'db1')
    
    #y_dwt.columns = [columns[0] + '_' + columns[1] for i in range(len(columns))]
    
    #feature_labels = [[y.name, columns[i]] for i in range(len(columns))]
    feature_labels = [[str(y.name), columns[i]] for i in range(len(columns))]
    
    y_dwt.columns = [str(y.name) + '_' + columns[i] for i in range(len(columns))]
    
    return y_dwt, feature_labels

#%%
def seasonal_decompose_feature(output_data, freq=3):
    """トレンド/周期成分に分解し、説明変数を生成
    """
    labels = [['seasonal_freq' + str(freq), 'lag' + str(i)] for i in range(freq)]
    columns = [labels[i][0] + '_' + labels[i][1] for i in range(freq)]
    
    n = len(output_data)
    x = pd.DataFrame(np.zeros([n,freq]), columns=columns, index=output_data.index) #記録領域
    
    for k in range(freq+1,n):
        y = output_data.iloc[0:k]
        
        y_sd = sm.tsa.seasonal_decompose(y.values, freq=freq, extrapolate_trend=2) #成分分解
        
        # トレンド/季節成分の結合
        y_sd = np.concatenate([y_sd.trend.reshape(-1,1), y_sd.seasonal.reshape(-1,1)], axis=1)
        y_sd = pd.DataFrame(y_sd, columns=['trend', 'seasonal'])
        
        x.iloc[k,:] = y_sd['seasonal'].iloc[k-freq:k].values[::-1]
    return x, labels

#%%
def calcAIC(x,y):
    """
    """
    n_cv = 3
    
    n = len(y)
    key = np.arange(n)
    y_pred, _ = train_test_ridge(x, y, key, key, n_cv)
    
    Se = sum((y - y_pred)**2)
    
    p = x.shape[1]
    aic = n*(np.log(2*np.pi*Se/n) + 1) + 1*(p+2)
    return aic


#%%
def stepwise_var_select(x, y):
    """stepwise
    """
    aic = calcAIC(x.values, y) #myfun
    
    # 初期化
    x_new = copy.deepcopy(x)
    cnt = 0
    
    while True:
        col_idx1 = np.arange(x_new.shape[1])
        dim = x_new.shape[1]
        
        AICs = np.zeros(dim-1) #記録領域
        for i in range(dim-1):
            col_idx2 = np.delete(col_idx1, i)
            AICs[i] = calcAIC(x_new.iloc[:,col_idx2].values, y) #myfun
        
        #import pdb; pdb.set_trace()
        
        best_idx = AICs.argmin()
        col_idx2 = np.delete(col_idx1, best_idx)
        
        delta_aic = aic - AICs.min()
        aic = AICs.min()
        
        x_new = x_new.iloc[:,col_idx2]
        
        # 終了条件 : aicが減少しなくなった場合
        #threshold = 0.1
        #if delta_aic > 0 and (delta_aic >= threshold):
        if delta_aic < 0:
            break
        
        print('stepwise iter NO.', str(cnt), ' : AIC=', aic)
        
        cnt = cnt + 1 #更新
    
    dim = len(x_new.columns)
    col_idx = np.zeros(dim, int) #記録領域
    for i in range(dim):
        col_idx[i] = np.where(x_new.columns[i] == x.columns)[0][0]
    
    return col_idx, x_new


###############################################################################
#%% initialize
root_dir = 'D:/Work/TAC/20191209'
data_dir = os.path.join(root_dir, 'data')
input_dir = os.path.join(root_dir, 'input')
output_dir = os.path.join(root_dir, 'output/20191220_stepwise')
#output_dir = os.path.join(root_dir, 'output/20191220_変数選択有b')
dir_path = os.path.join(output_dir, 'fig')
if os.path.isdir(dir_path) == False:
    os.makedirs(dir_path)

# データ読込 : 納入実績
fname = os.path.join(data_dir, '納入実績.csv')
output_data_o = pd.read_csv(fname, parse_dates=True, header=0, index_col=None, encoding='cp932')

# データ読込 : 内示
fname = os.path.join(data_dir, 'forecast.csv')
forecast_o = pd.read_csv(fname, parse_dates=True, header=0, index_col=0, encoding='cp932')

# データ読込 : 車両生産計画
fname = os.path.join(data_dir, '車両生産計画.csv')
input_data_o = pd.read_csv(fname, parse_dates=True, header=0, index_col=0, encoding='UTF-8')

# データ読込 : 車種-品番紐づけ表
fname = os.path.join(input_dir, '車種-品番紐づけ表HV_2.csv')
item_tbl_o = pd.read_csv(fname, header=0, index_col=None, encoding='cp932')

# タイムスタンプの型変換
col_name = '年月'
forecast_o[col_name] = conv_timestamp(forecast_o[col_name], date_format='%Y-%m-%d') #my fun


#%% 前処理(目的変数)
# タイムスタンプの型変換
start_idx = np.where('規格' == output_data_o.columns)[0][0] + 1

n_col = output_data_o.shape[1]
t = output_data_o.columns[start_idx:n_col]
t = conv_timestamp(t, date_format='%Y/%m/%d') #my fun

columns = list(output_data_o.columns)
columns[start_idx:n_col] = t
output_data_o.columns = columns

# 重複のある商品コードをリストアップ
comb_indexs = get_overlap_item_idxs(output_data_o['商品コード'].values) #my fun

# 商品コードに重複のない出荷実績データを生成
output_data_o = make_no_overlap_data(output_data_o, comb_indexs) #my fun


### 指定した商品コードが出荷実績データに存在するか確認 ###
_, item_tbl, item_key, index = make_set_item_data(output_dir, output_data_o, item_tbl_o, 1) #my fun

### 前処理(紐づけ表) ###
# NANの要素を空白に置換
item_tbl = item_tbl_o.fillna('')


#%% 前処理(説明変数)
input_data_o = input_data_o.drop(columns={'N+2月輸出'})
input_data_o = input_data_o.rename(columns={'N+2月市販' : '生産計画'})


### nanの置換 ###
# 車種1がNANの要素を削除
index = np.where(input_data_o['車種1'].str.contains('', na=False))[0]
input_data_o = input_data_o.iloc[index,:]

# 車種2列がNANの要素を空白に置換
input_data_o['車種2'] = input_data_o['車種2'].fillna('')

replace_tbl = [['VOXY ○特HV', 'VOXY○特HV'],
['VOXY ○特', 'VOXY○特']]

for i in range(len(replace_tbl)):
    input_data_o['車種2'] = input_data_o['車種2'].replace(replace_tbl[i][0], replace_tbl[i][1])


# タイムスタンプの取得
t = conv_timestamp(input_data_o['年月'].values, date_format='%Y年%m月') #myfun

# タイムスタンプを２ヶ月ずらす
for k in range(len(t)):
    t[k] = t[k] + relativedelta(months=2)

input_data_o.index = t
input_data_o = input_data_o.drop(columns={'年月'})


#%% 目的変数の選択
threshold = 20

data = output_data_o.iloc[:, start_idx:output_data_o.shape[1]]

# 指定した条件で対象品目を抽出
index, stats = select_output_var(data, threshold) #myfun

index = np.intersect1d(item_key, index)

tmp = list(output_data_o['商品コード'].iloc[index].values)

index = []
for i in range(item_tbl.shape[0]):
    if item_tbl['品番'].iloc[i] in tmp:
        index.append(i)

item_tbl = item_tbl.iloc[index,:]

N = item_tbl.shape[0] #対象品目数


index = []
for i in range(item_tbl.shape[0]):
    idx = np.where(item_tbl['品番'].iloc[i] == output_data_o['商品コード'].values)[0][0]
    index.append(idx)


### 平均誤差率の記録領域 ###
columns = ['内示誤差率 [%]', '予測誤差率 [%]']
all_err_rate = output_data_o.iloc[index, 0:start_idx]
all_err_rate.index = np.arange(N) #index振り直し

for i in range(len(columns)):
    col_name = columns[i]
    all_err_rate[col_name] = np.zeros(N)

start = 30
#lists = np.arange(start,start+1)
lists = np.arange(35,N)
#lists = np.arange(N)


#%% main
# 目的変数の指定
for output_id_ in lists:
    output_name = item_tbl['品番'].iloc[output_id_]
    
    output_id = np.where(output_name == output_data_o['商品コード'].values)[0][0]
    
    output_data = output_data_o.iloc[[output_id], start_idx:output_data_o.shape[1]]
    output_data = output_data.iloc[0,:]
    output_data.name = output_name
    
    date = "{0:%H:%M:%S}".format(dt.datetime.now())
    print('[', date, '] START -> No.', output_id_, '/', N)    
    
    
    ##############
    # 内示の取得
    ##############
    index = np.where(output_name == forecast_o['品番'].values)[0]
    forecast = forecast_o.iloc[index,:]['N+2月']
    
    # タイムスタンプの取得
    t = forecast_o.iloc[index,:]['年月']
    t = pd.to_datetime(t)
    
    forecast.index = t #indexをタイムスタンプに設定
    
    # 目的変数の形状に合うように、内示をリサイズ
    data = pd.concat([output_data, forecast], axis=1, join_axes=[output_data.index])
    forecast = data.iloc[:,1]
    
    forecast = forecast.fillna(0)
    
    
    #%% 説明変数の抽出
    # 車種生産計画の全データから、説明変数を抽出する
    car_data, _, comb_car_type = Gen_InputData_from_AllCarproductSchedule(input_data_o, item_tbl.iloc[[output_id_],:])
    
    # カラム名に含まれるN+2月を削除
    columns = car_data.columns
    columns = [columns[i].replace('N+2月', '') for i in range(len(columns))]
    car_data.columns = columns
    
    car_data = car_data.fillna(0)
    
    
    ### 車種名に'540A', '341B'が含まれる場合、車種名を置換し、集約 ###
    car_data = agg_540A_341B(car_data) #myfun
    
    
    ### 目的変数の形状に合うように、内示をリサイズ ###
    data = pd.concat([output_data, car_data], axis=1, join_axes=[output_data.index])
    car_data = data.iloc[:,1:data.shape[1]]
    
    input_data = copy.deepcopy(car_data) #説明変数に設定
    
    columns = input_data.columns
    input_data.columns = [str(columns[i]) for i in range(len(columns))]
    
    
    #%% 説明変数の生成
    #################
    # 内示を説明変数に追加
    #################
    tmp = copy.deepcopy(forecast)
    tmp.name = '内示'
    input_data = pd.concat([input_data, tmp], axis=1)
    
    
    #################
    # 平滑化した説明変数を追加
    #################
    windows = [3, 6]
    cnt = 0 #初期化
    for w in windows:
        x_ = input_data.fillna(0)
        
        #--- approximationの計算
        approx = x_.rolling(w).mean()
        
        columns = input_data.columns
        columns = [columns[i] + '_approx' + str(w) for i in range(len(columns))]
        approx.columns = columns
        
        #--- detailの計算
        detail = input_data.values - approx.values
        
        columns = input_data.columns
        columns = [columns[i] + '_detail' + str(w) for i in range(len(columns))]
        detail = pd.DataFrame(detail, columns=columns, index=input_data.index)
        
        #--- approximation/detailの結合
        x_ = pd.concat([approx, detail], axis=1)
        
        if cnt == 0:
            x = copy.deepcopy(x_)
        else:
            x = pd.concat([x, x_], axis=1)
        
        cnt += 1
    
    input_data = pd.concat([input_data, x], axis=1)
    
    
    #################
    # 過去の時系列(目的変数)
    #################
    n_lag = [2,6] #ラグ数
    x = gen_lag_data(output_data, n_lag) #my fun
    
    # カラム名の変更
    columns = list(x.columns)
    for i in range(len(columns)):
        columns[i] = 'y_' + columns[i]
    x.columns = columns
    
    input_data = pd.concat([input_data, x], axis=1)
    input_labels = list(input_data.columns)
    
    
    #################
    # 過去の時系列(車両生産計画)
    #################
    n_lag = [1,4] #ラグ数
    for i in range(car_data.shape[1]):
        car_name = car_data.columns[i]
        x_ = gen_lag_data(car_data.iloc[:,i], n_lag) #my fun
        
        # カラム名の変更
        columns = list(x_.columns)
        for j in range(len(columns)):
            columns[j] = car_name + '_' + columns[j]
        x_.columns = columns
        
        if i >= 1:
            x = pd.concat([x, x_], axis=1)
        else:
            x = copy.deepcopy(x_)
    
    input_data = pd.concat([input_data, x], axis=1)
    input_labels = input_labels + list(x.columns) #説明変数のカラム名の追加
    
    
    ### 同月1年前(目的変数) ###
    x = get_1year_before_data(output_data) #myfun
    
    input_data = pd.concat([input_data, x], axis=1) #説明変数の追加
    input_labels = input_labels + list(x.columns) #説明変数のカラム名の追加
    
    
    ### up/stay/downの状態(目的変数) ###
    x = gen_up_down_state(output_data, 8) #myfun
    x = x.shift(-2) #2ヶ月シフト
    x.name = x.name + '_y'
    
    input_data = pd.concat([input_data, x], axis=1) #説明変数の追加
    input_labels.append(x.name) #説明変数のカラム名の追加
    
    ### up/stay/downの状態(車両生産計画) ###
    x = gen_up_down_state(car_data, 8) #myfun
    x.name = x.name + '_x'
    
    input_data = pd.concat([input_data, x], axis=1) #説明変数の追加
    input_labels.append(x.name) #説明変数のカラム名の追加
    
    
    ### 直近Xヶ月間のup/stay/downの回数 ###
    windows = [2, 3, 4]
    x = input_data['up_down_label_y']
    x, labels = count_up_down_state(x, windows) #myfun
    
    input_data = pd.concat([input_data, x], axis=1) #説明変数の追加
    input_labels = input_labels + list(x.columns) #説明変数のカラム名の追加
    
    
    ### 離散ウェーブレット分解(目的変数) ###
    y = copy.deepcopy(output_data)
    y.name = 'y'
    x, labels = mydwt(y) #myfun
    x = x.shift(-2) #2ヶ月シフト
    
    input_data = pd.concat([input_data, x], axis=1) #説明変数の追加
    input_labels = input_labels + list(labels) #説明変数のカラム名の追加
    
    
    ### 離散ウェーブレット分解(車両生産計画) ###
    for i in range(car_data.shape[1]):
        x_, labels_ = mydwt(car_data.iloc[:,i]) #myfun
        
        if i == 0:
            x = copy.deepcopy(x_)
            labels = copy.deepcopy(labels_)
        else:
            x = pd.concat([x, x_], axis=1)
            labels = labels + labels_
    
    input_data = pd.concat([input_data, x], axis=1) #説明変数の追加
    input_labels = input_labels + list(labels) #説明変数のカラム名の追加
    
    
    ### トレンド/周期成分に分解(目的変数) ###
    freqs = [3]
    for i in range(len(freqs)):
        # 目的変数をトレンド/周期成分に分解し、説明変数を生成
        x, labels = seasonal_decompose_feature(output_data, freqs[i]) #myfun
        x = x.shift(-2) #2ヶ月シフト
        
        input_data = pd.concat([input_data, x], axis=1) #説明変数の追加
        input_labels = input_labels + list(labels) #説明変数のカラム名の追加
    
    
    #%% 説明変数の選択
    x = input_data.fillna(0)
    index = np.where(np.sum(x != 0, axis=0) >= 8)[0]
    
    input_data = input_data.iloc[:,index]
    
    ### 同月1年前の説明変数を除去 ###
    #col_name = '同月1年前'
    #index = np.where(col_name == np.array(input_labels))[0]
    #input_data = input_data.drop(columns=col_name, axis=1)
    
    
    #%% 説明変数の評価
    n_cv = 5
    
    x = input_data.fillna(0)
    y = copy.deepcopy(output_data)
    
    valid_key = np.where(output_data.values != 0)[0]
    
    train_key, test_key = train_test_split(valid_key, test_size=0.2, shuffle=False)
    
    n_in = x.shape[1]
    #n_in = 5
    scores = np.zeros(n_in) #記録領域
    
    # 説明変数の指定
    for input_id in range(n_in):
        y_pred, p = train_test_ridge(x.iloc[:,[input_id]].values, y.values, valid_key, valid_key, n_cv)
        
        y_pred = pd.Series(y_pred, index=output_data.index[valid_key])
        
        scores[input_id] = r2_score(y.iloc[valid_key].values, y_pred.values)
    
    threshold = 0.3
    col_idx = np.where(scores >= threshold)[0]
    
    
    #%%
    scores = np.zeros(n_in) #記録領域
    
    # 説明変数の指定
    for input_id in range(n_in):
        y_pred, p = train_test_ridge(x.iloc[:,[input_id]].values, y.values, train_key, test_key, n_cv)
            
        y_pred = pd.Series(y_pred, index=output_data.index[test_key])
        
        scores[input_id] = r2_score(y.iloc[test_key].values, y_pred.values)
    
    threshold = 0.3
    col_idx = np.where(scores >= threshold)[0]
    
    
    #----
    #r2_score(y.iloc[test_key].values, forecast.iloc[test_key].values)
    r2_score(y.iloc[valid_key].values, forecast.iloc[valid_key].values)
    
    
    #%% 予測
    ### 学習/テストデータのキー ###
    # 有効なキーの取得
    #threshold = 2
    
    valid_key = np.where(output_data.values != 0)[0]
    n = output_data.shape[0] #データ長
    
    # 例外処理
    if len(valid_key) <= 4:
        print('ERROR : 有効なデータ数が少ない!')
        continue
    
    train_key, test_key = train_test_split(valid_key, test_size=0.3, shuffle=False)
    
    # 例外処理 : 出荷量一定以上となる月数が少ない場合
    index = np.where(output_data.iloc[test_key] >= 10)[0] #出荷量一定以上となる月のインデックス
    if len(index) < 5:
        print('ERROR : 出荷量一定以上となる月数が少ない!')
        continue
    
    # 例外処理 : 出荷量一定以上となる月数が少ない場合
    index = np.where(output_data.iloc[train_key] >= 10)[0] #出荷量一定以上となる月のインデックス
    if len(index) < 5:
        print('ERROR : 出荷量一定以上となる月数が少ない!')
        continue
    
    
    #%%
    horizon = 2
    n_cv = 5
    #train_window = [10, 10]
    
    y = copy.deepcopy(output_data)
    y = y.fillna(0)
    t = y.index
    x = input_data.fillna(0)
    
    #dy = y.diff(2) #目的変数の変化量(2回微分)
    
    n = len(test_key) #データ長
    y_pred = np.zeros(n) #予測の記録領域
    
    train_key1 = list(train_key)
    #train_key_ = train_key_[10:len(train_key_)]
    
    
    #col_idx1 = copy.deepcopy(col_idx)
    col_idx1 = np.arange(x.shape[1])
    
    # 例外処理
    if len(col_idx) == 0:
        col_idx1 = np.arange(x.shape[1])
    
    # テストデータの指定
    for k in range(n):
        print('NO.', str(k), '/', str(n-1))
        
        test_key_ = test_key[k]
        
        # 学習データ追加
        if (k >= horizon) and (t[test_key_].month - t[test_key[k-horizon]].month >= 2):
            train_key1.append(test_key[k-horizon])
        
        
        ############
        # 学習データの範囲決定
        ############
        #test_idx = train_key1[-1]
        train_idx1, test_idx = train_test_split(train_key1, test_size=0.1, shuffle=False)
        
        #length = len(train_key1) - 6
        length = len(train_idx1) - 4
        scores = np.zeros(length) #記録領域
        scores = pd.DataFrame(np.zeros(length), index=np.arange(length)) #記録領域
        for start in range(length):
            train_idx2 = train_idx1[start:len(train_idx1)]
            
            # 最小学習データ数の保証
            #if len(train_idx) <= 6:
            #    break
            
            ym, p = train_test_ridge(x.iloc[:,col_idx1].values, y.values, train_idx2, test_idx, 3)
            
            # 評価関数の計算
            scores.iloc[start] = r2_score(y[test_idx], ym) #決定係数
            #scores.iloc[start] = rmse(ym, y[test_idx]) # RMSE
        
        #start = np.argmin(scores.values)
        start = np.argmax(scores.values)
        # scores[start]
        
        train_key2 = train_key1[start:len(train_key1)]
        
        
        ############
        # ステップワイズ法による変数選択
        ############
        col_idx2, _ = stepwise_var_select(x.iloc[train_key2,col_idx1], y.iloc[train_key2].values)
        
        
        ############
        # 寄与率の高い変数の選択
        ############
        if 0:
            _, p = train_test_en(x.iloc[:,col_idx1].values, y.values, train_key2, 0, 5)
            
            index, p_rate = var_select_based_contr_rate(p, 1) #myfun
            
            if len(index) != 0:
                col_idx2 = col_idx1[index]
            else:
                col_idx2 = copy.deepcopy(col_idx1)
            
            col_idx2 = copy.deepcopy(col_idx1)
        
        
        ############
        # 予測
        ############
        y_pred[k], p = train_test_ridge(x.iloc[:,col_idx2].values, y.values, train_key2, test_key_, n_cv)
        #y_pred[k], p = train_test_lasso(x.iloc[:,col_idx].values, y.values, train_key_, test_key_, n_cv)
        #y_pred[k], p = train_test_en(x.iloc[:,col_idx].values, y.values, train_key_, test_key_, n_cv)
        #y_pred[k] = train_test_xgb(x.iloc[:,col_idx2].values, y.values, train_key2, test_key_, n_cv)
        #y_pred[k], p = train_test_randomForest(x.iloc[:,col_idx].values, y.values, train_key_, test_key_, n_cv)
    
    
    #%% データ範囲を超える場合、内示で置換
    ### データ範囲の算出 ###
    w = 4
    
    length = len(y) #データ長
    limit = np.zeros([length,2]) #記録領域
    
    for k in range(horizon+w,length):
        end = k-horizon
        start = end - w
        
        index = np.arange(start, end)
        index = np.intersect1d(index, valid_key)
        tmp = y.iloc[index]
        
        limit[k,0] = tmp.min()
        limit[k,1] = tmp.max()
    
    ###
    y_pred2 = copy.deepcopy(y_pred)
    
    bool1 = y_pred < limit[test_key,0]
    bool2 = limit[test_key,1] < y_pred
    bools = bool1 + bool2
    
    index = np.where(bools == 1)[0]
    
    y_pred2[index] = forecast.iloc[test_key[index]]
    
    
    #%%
    #dy_pred = np.diff(y_pred, 2)
    #dy_pred = np.append(np.nan, dy_pred)
    
    #y_pred - 
    
    #dy = y.diff()
    #dy = np.append(np.nan, dy)
    
    
    #%%
    ############
    # 評価
    ############
    y = copy.deepcopy(output_data)
    
    columns = ['内示誤差率 [%]', '予測誤差率 [%]']
    err_rate = pd.DataFrame(np.zeros([1,len(columns)]), columns=columns, index=[output_name]) #記録領域
    
    tmp = calc_mean_error_rate(y.iloc[test_key].values, forecast.iloc[test_key].values) #my fun
    err_rate['内示誤差率 [%]'] = tmp.mean()
    
    tmp = calc_mean_error_rate(y.iloc[test_key], y_pred2) #my fun
    err_rate['予測誤差率 [%]'] = tmp.mean()
    
    
    ### 平均誤差率の履歴保存 ###
    for i in range(2):
        col_name = err_rate.columns[i]
        all_err_rate[col_name].iloc[output_id_] = err_rate[col_name].iloc[0]
    
    ### 表示 ###
    for i in range(err_rate.shape[1]):
        col_name = err_rate.columns[i]
        print(col_name, '=', err_rate.iloc[0,i], '%')
    
    
    #%% save
    ### 全品目の平均誤差率の保存 ###
    fname = os.path.join(output_dir, 'err_rate.csv')
    all_err_rate.to_csv(fname, encoding='cp932')
    
    ### 平均誤差率の保存 ###
    output_dir_ = os.path.join(output_dir, 'csv')
    if os.path.isdir(output_dir_) == False:
        os.makedirs(output_dir_)
    
    fname = 'NO' + str(output_id_) + '_'+  output_name + '.csv'
    fname = os.path.join(output_dir_, fname)
    err_rate.to_csv(fname, encoding='cp932')
    
    
    ### 予測結果の明細の保存 ###
    output_dir_ = os.path.join(output_dir, 'detail_csv')
    if os.path.isdir(output_dir_) == False:
        os.makedirs(output_dir_)
    
    fname = 'NO' + str(output_id_) + '_'+  output_name + '.csv'
    fname = os.path.join(output_dir_, fname)
    
    # 結果明細の出力
    res = output_result_detail(fname, output_data, forecast, y_pred2, test_key) #myfun
    
    
    #%%
    ###################
    # グラフ表示 : 実績/予測の時系列
    ###################
    save_flag = 1
    fontsize = 15
    
    ### 実績/内示/予測(テスト期間) ###
    # set output filename
    fname = 'NO' + str(output_id_) + '_' + output_name + '.png'
    fname = os.path.join(output_dir, 'fig', fname)
    
    t = y.index
    
    #plot_time_series(fname, t[test_key], y.iloc[test_key], y_pred, forecast.iloc[test_key], output_name, None, 1, fontsize, save_flag)
    plot_time_series(fname, t[test_key], y.iloc[test_key], y_pred2, forecast.iloc[test_key], output_name, None, 1, fontsize, save_flag)
    plt.close()
    
    
    #% %
    ### 実績/内示(全期間) ###
    dir_path = os.path.join(output_dir, 'raw')
    if os.path.isdir(dir_path) == False:
        os.makedirs(dir_path)
    
    # set output filename
    fname = 'NO' + str(output_id_) + '_' + output_name + '.png'
    fname = os.path.join(output_dir, 'raw', fname)
    
    n = len(t) #データ長
    tmp = np.zeros(n)
    tmp[tmp == 0] = np.nan
    plot_time_series(fname, t, y.values, tmp, forecast.values, output_name, None, 3, fontsize, save_flag)
    plt.close()
    
    
    #%%
    plt.figure(figsize=(12,7))
    plt.plot(y)
    
    i = 10
    #i = col_idx[i]
    plt.plot(input_data.iloc[:,i])
    plt.title(input_data.columns[i])
    plt.close()
    
    
    #%%
    #plt.figure(figsize=(10,5))
    #plt.plot(y)
    
    #plt.figure(figsize=(10,5))
    #i = col_idx[i]
    #plt.plot(input_data.iloc[:,i])
    #plt.title(input_data.columns[i])


