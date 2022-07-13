import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import random
random.seed(777)
import xgboost as xgb
from sklearn.metrics import classification_report,roc_auc_score
from sklearn.metrics import r2_score
import copy
import optuna
import seaborn as sns
# from my_library.library import DataFramePreProcessing


def xgb_pred(x_train, y_train, x_test, y_test):
    param_dist = {'objective':'binary:logistic', 'n_estimators':16,'use_label_encoder':False,
                 'max_depth':4}
    
    param_def = {'objective':'binary:logistic','use_label_encoder':False}
    xgb_model = xgb.XGBClassifier(**param_dist)
    hr_pred = xgb_model.fit(x_train.astype(float), np.array(y_train), eval_metric='logloss').predict(x_test.astype(float))
    print("---------------------")
    y_proba_train = xgb_model.predict_proba(x_train)[:,1]
    y_proba = xgb_model.predict_proba(x_test)[:,1]
    print('AUC train:',roc_auc_score(y_train,y_proba_train))    
    print('AUC test :',roc_auc_score(y_test,y_proba))
    print(classification_report(np.array(y_test), hr_pred))
    xgb.plot_importance(xgb_model) 
    return xgb_model


def predict_tomorrow(lq,folder_name):
    path_ = '/Users/rince/Desktop/StockPriceData/%s/*.csv' % folder_name
    file = glob.glob(path_)
    path_tpx = sorted(file)[-1]
    path_ = '/Users/rince/Desktop/StockPriceData/DAW/*.csv'
    file = glob.glob(path_)
    path_daw = sorted(file)[-1]
    lq.predict_tomorrow(path_tpx,path_daw)
    
def plot(g,label='x'):
#     type(g) = pd.DataFrame
    plt.subplots(figsize=(10, 6))
    plt.fill_between(g.index,y1 = g['ma'] - g['std'],y2=g['ma']+g['std'],alpha=0.3)
    plt.plot(g.index,g['ma'])
    plt.xlabel(label)
    plt.ylabel('reward')
    plt.grid(True)
    
def make_plot_data(reward_log, ma=5):
#     type(reward_log)==list

    length = len(reward_log)
    reward_log = np.array(reward_log)
    reward_dict = {}
    if ma%2==0:
        print("ma must be odd number.")
        return 
    
    
    sride = ma//2
    try:
        for i in range(sride,length-sride):
            reward_dict[i] = {'reward':reward_log[i],'ma':reward_log[i-sride:i+sride+1].mean(),
                             'std':reward_log[i-sride:i+sride+1].std()}
    except:
        print("Error.")
    
    return pd.DataFrame(reward_dict).T
     
def easy_plot(df,xlabel='episode',ylabel='reward'):
    plt.subplots(figsize=(10, 6))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.plot(df)
    plt.show()
    
def make_df_con(path_tpx,path_daw):
    df_tpx = DataFramePreProcessing(path_tpx).load_df()
    df_daw = DataFramePreProcessing(path_daw,is_daw=True).load_df()
    daw_p = df_daw.pct_change()
    df_con = pd.concat([daw_p,df_tpx],axis = 1,join='inner').astype(float)
    df_tmp = df_con.drop(df_con[ df_con['volume']==0].index)
    return df_tmp

def grid_search(x_train,y_train,x_test,y_test):
    trains = xgb.DMatrix(x_train.astype(float), label=y_train)
    tests = xgb.DMatrix(x_test.astype(float), label=y_test)

    base_params = {
        'booster': 'gbtree',
        'objective':'binary:logistic',
        'eval_metric': 'rmse',
        'random_state':100,
        'use_label_encoder':False
    }

    watchlist = [(trains, 'train'), (tests, 'eval')]
    tmp_params = copy.deepcopy(base_params)
    
#     インナー関数
    def optimizer(trial):
        eta = trial.suggest_uniform('eta', 0.01, 0.3)
        max_depth = trial.suggest_int('max_depth', 3, 15)
        __lambda = trial.suggest_uniform('lambda', 0.7, 2)
        n_estimators = trial.suggest_int('n_estimators', 3, 20)
        learning_rate = trial.suggest_uniform('lambda', 0.01, 1)
        reg_alpha = trial.suggest_uniform('reg_alpha', 0.01, 1)
        reg_lambda = trial.suggest_uniform('reg_lambda', 0.01, 1)
        importance_type = trial.suggest_categorical('importance_type',
                                                    ['gain', 'weight', 'cover','total_gain','total_cover'])

        tmp_params['eta'] = eta
        tmp_params['max_depth'] = max_depth
        tmp_params['lambda'] = __lambda
        tmp_params['n_estimators'] = n_estimators
        tmp_params['learning_rate'] = learning_rate
        tmp_params['reg_alpha'] = reg_alpha
        tmp_params['reg_lambda'] = reg_lambda
        tmp_params['importance_type'] = importance_type
        model = xgb.train(tmp_params, trains, num_boost_round=50)
        predicts = model.predict(tests)
        r2 = r2_score(y_test, predicts)
        print(f'#{trial.number}, Result: {r2}, {trial.params}')
        return r2
    
    study = optuna.create_study(direction='maximize')
    study.optimize(optimizer, n_trials=500)
    print(study.best_params)
    print(study.best_value)

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

def return_latest_data_path(folder_name):
    path_ = '/Users/rince/Desktop/StockPriceData/%s/*.csv' % folder_name
    file = glob.glob(path_)
    path_tpx = sorted(file)[-1]
    path_ = '/Users/rince/Desktop/StockPriceData/DAW/*.csv'
    file = glob.glob(path_)
    path_daw = sorted(file)[-1]
    return path_tpx, path_daw

def load_csv(load_path):
    df = pd.read_csv(load_path, index_col=0)
    return df

def standarize(df):
    # ddof = 0 : 分散
    # ddof = 1 : 不偏分散
    df = (df - df.mean())/df.std(ddof=0)
    return df

def make_data(x_,y_):
    train = pd.concat([y_,x_],axis = 1,join='inner').astype(float)
    x_ = train[train.columns[1:]]
    y_ = train[train.columns[0]]
    return x_,y_

def inverse(after,std_,mean_):
    after = after*std_ + mean_
    return after 

def show_corr(x_):
    x_corr = x_.corr()
    fig, ax = plt.subplots(figsize=(20, 20)) 
    sns.heatmap(x_corr, square=True, vmax=1, vmin=-1, center=0)
    plt.plot()

def return_corr(df,year=2021):
    x = df[df.index.year==year]['daw_close'].values
    y = df[df.index.year==year]['close'].values
    corr = np.corrcoef(x,y)
    return corr

def return_strong_corr(x_):
    strong_corr = []
    x_corr = x_.corr()
    for idx in x_corr.index:
        for col in x_corr.columns:
            if idx == col:
                continue
            else:
                corr = x_corr.loc[idx][col]
                if abs(corr)>=0.8:
                    strong_corr.append([idx,col])
    return strong_corr

def process_kawase(path_kawase,df_con):
    df_kawase = pd.read_csv(path_kawase, index_col=0,encoding='Shift_JIS')
    column_name = df_kawase.iloc[0]
    df_kawase = df_kawase.set_axis(df_kawase.iloc[1].values.tolist(),axis=1).iloc[2:]
    df_kawase.dropna(how='all',axis=1,inplace=True)
    df_kawase.replace('*****',np.nan,inplace=True)
    df_kawase = df_kawase.astype('float64')
    df_kawase['day'] = pd.to_datetime(df_kawase.index,format='%Y/%m/%d')
    df_kawase.set_index('day',inplace=True)
    return df_kawase


def cos_sim(vec1,vec2):
    inner_product = vec1 @ vec2
    vec1_norm = np.linalg.norm(vec1)
    vec2_norm = np.linalg.norm(vec2)
    norm_product = vec1_norm*vec2_norm
    cos = inner_product/norm_product
    return cos

def show_results_fft(wave_vec):
    N = len(wave_vec)            # サンプル数
    dt = 1          # サンプリング間隔
    t = np.arange(0, N*dt, dt) # 時間軸
    freq = np.linspace(0, 1.0/dt, N) # 周波数軸

    f = wave_vec
    F = np.fft.fft(f)
    # 振幅スペクトルを計算
    Amp = np.abs(F)

    # グラフ表示
    plt.figure()
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10
    plt.plot(t, f, label='f(n)')
    plt.xlabel("Time", fontsize=20)
    plt.ylabel("Signal", fontsize=20)
    plt.grid()
    leg = plt.legend(loc=1, fontsize=15)
    leg.get_frame().set_alpha(1)
    plt.show()

    plt.plot(freq, Amp, label='|F(k)|')
    plt.xlabel('Frequency', fontsize=20)
    plt.ylabel('Amplitude', fontsize=20)
    plt.grid()
    leg = plt.legend(loc=1, fontsize=15)
    leg.get_frame().set_alpha(1)
    plt.show()
    return F


def do_fft(wave_vec):
    N = len(wave_vec)            # サンプル数
    dt = 1          # サンプリング間隔
    t = np.arange(0, N*dt, dt) # 時間軸
    freq = np.linspace(0, 1.0/dt, N) # 周波数軸

    f = wave_vec
    F = np.fft.fft(f)

    # 振幅スペクトルを計算
    Amp = np.abs(F)
    return F

def make_spectrum(wave_vec):
    F = do_fft(wave_vec)
    spectrum = np.concatenate([F.real,F.imag])
    # spectrum = np.abs(F)**2
    return spectrum


def cos_sim(vec1,vec2):
    inner_product = vec1 @ vec2
    vec1_norm = np.linalg.norm(vec1)
    vec2_norm = np.linalg.norm(vec2)
    norm_product = vec1_norm*vec2_norm
    cos = inner_product/norm_product
    return cos





def make_easy_x(ng):
    x = []
    for df in ng:
        lis_ =  df.tolist()
        # print(lis_)
        x.append(lis_)
    x = np.array(x)
    return x



