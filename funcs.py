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
import pickle
from scipy import fftpack
from scipy import signal


def xgb_pred(x_train, y_train, x_test, y_test):
    param_dist = {
        'objective':'binary:logistic',
        'n_estimators':16,
        'use_label_encoder':False,
        'max_depth':4
        }
    
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
    spectrum = np.abs(F)**2
    spectrum = spectrum[:len(spectrum)//2]
    return standarize(spectrum)


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


def save_pickle(save_path,object_):
    with open(save_path, mode="wb") as f:
        pickle.dump(object_, f)

def load_pickle(save_path):
    with open(save_path, mode="rb") as f:
        object_ = pickle.load(f)
    return object_



def show_structure(xgb_model):
    xgb.to_graphviz(xgb_model)


def decode(spe):

    length = len(spe)
    mid = length//2

    real_ = spe[:mid]
    imag_ = spe[mid:]

    c_list = []
    for i in range(len(imag_)):
        c_list.append(complex(0,imag_[i]))

    c = np.array(c_list)
    F = real_ + c
    return F


def get_gyosyu_df():
    path_gyosyu = '/Users/Owner/Desktop/StockPriceData/Gyosyu_encoded/'
    FILE = glob.glob(path_gyosyu+'*.csv')
    df_dict = {}
    for file in FILE:
        name = file.replace(path_gyosyu,'')[:-4]
        df = pd.read_csv(file)
        df = df.rename(columns={df.columns[0]:'nan',df.columns[1]:'nan',df.columns[2]:'nan',\
                                    df.columns[3]:'day',df.columns[4]:'nan',df.columns[5]:'open',\
                                    df.columns[6]:'high',df.columns[7]:'low',df.columns[8]:'close',\
                                        df.columns[9]:'volume',})
        df = df.drop('nan',axis=1)
        df = df.drop(df.index[0])
        df['day'] = pd.to_datetime(df['day'],format='%Y/%m/%d')
        df.set_index('day',inplace=True)
        df_dict[name] = df

    return df_dict,FILE

def make_value_list(lx,start_year,end_year,path_tpx,path_daw,alpha=0.34,width=20,stride=10,start_month=1,end_month=12):

    lc_dummy = LearnClustering(width=width)
    df_con = lc_dummy.make_df_con(path_tpx,path_daw)
    
    df_con = df_con[df_con.index.year<=end_year]
    df_con = df_con[df_con.index.year>=start_year]
    df_con = df_con[df_con.index.month<=end_month]
    df_con = df_con[df_con.index.month>=start_month]
    
    x_,z_ = lc_dummy.make_x_data(df_con['close'],stride=stride,test_rate=1.0,width=width)
    length = len(z_)
    value_list = []

    for i in range(length):
        for strategy in ['normal','reverse']:
            try:
                xl = XGBSimulation2(lx,alpha=alpha)
                xl.simulate(path_tpx,path_daw,strategy=strategy,is_validate=True,start_year=start_year,end_year=end_year,df_=z_[i])
                
                trade_log =  xl.trade_log
                total_profit = trade_log['total_profit'].values[0]
                stock_wave = z_[i]
                vt = ValueTable(strategy,alpha,total_profit,trade_log,stock_wave)
                value_list.append(vt)
                
            except Exception as e:
                print(e)
                continue

    return value_list

def return_clx(Value_list):
    Value_good = sorted(Value_list,key=lambda x :x[2],reverse=True)
    Value_bad = sorted(Value_list,key=lambda x :x[2],reverse=False)
    ng = []
    rg = []
    nb = []
    rb = []
    
    # 1sigam = 外れ値 として処理する
    prf_list=[]
    for vg in Value_good:
        total_profit = vg.total_profit
        prf_list.append(total_profit)      
    prf_array = np.array(prf_list)
    st_prf = standarize(prf_array)

    for idx,v in enumerate(Value_good):
        if v.total_profit<=0:break
        if np.abs(st_prf[idx]) >=1:continue    

        df = v.stock_wave
        strategy = v.strategy
        # print(df)
        # break
        if strategy=="normal":
            ng.append(standarize(df))
        else:
            rg.append(standarize(df))

    prf_list=[]
    for vb in Value_bad:
        total_profit = vb.total_profit
        prf_list.append(total_profit)      
    prf_array = np.array(prf_list)
    st_prf = standarize(prf_array)

    for v in Value_bad:
        if v.total_profit>=0 :break
        if np.abs(st_prf[idx]) >=1:continue  
        
        df = v.stock_wave
        strategy = v.strategy

        if strategy=="normal":
            nb.append(standarize(df))
        else:
            rb.append(standarize(df))

    x_ng = make_easy_x(ng)
    x_nb = make_easy_x(nb)
    x_rg = make_easy_x(rg)
    x_rb = make_easy_x(rb)
    return x_ng,x_nb,x_rg,x_rb

def return_ffs(lx,x_ng,x_nb,x_rg,x_rb,FFT_obj,width=20,stride=5):


    log_dict = {}
    cs_dict = {}
    ffs_dict = {}

    random_state=0

    alpha = 0.33
    n_cluster = 1
        
    Fstrategies = []
    Cstrategies = []
    Phases = []
    lc_rg = LearnClustering(n_cluster=n_cluster,random_state=random_state)
    lc_rg.learn_clustering3(x_rg,width=width,stride=stride)
    lc_rb = LearnClustering(n_cluster=n_cluster,random_state=random_state)
    lc_rb.learn_clustering3(x_rb,width=width,stride=stride)
    lc_ng = LearnClustering(n_cluster=n_cluster,random_state=random_state)
    lc_ng.learn_clustering3(x_ng,width=width,stride=stride)
    lc_nb = LearnClustering(n_cluster=n_cluster,random_state=random_state)
    lc_nb.learn_clustering3(x_nb,width=width,stride=stride)

    strategy_list = ['normal','stay','reverse','stay']

    j=0
    fft_dummy = FFT_obj(lx,None,width=width)
    for lc in [lc_ng,lc_nb,lc_rg,lc_rb]:
        
        for _,key in enumerate(lc.wave_dict):
            wave = lc.wave_dict[key]
            spe = fft_dummy.make_spectrum(wave)
            F,Amp = fft_dummy.do_fft(wave)
            F = F[:len(F)//2]
            phase = np.degrees(np.angle(F))
            strategy = strategy_list[j]
            fs  = Fstrategy(strategy,alpha,spe)
            cs = Fstrategy(strategy,alpha,wave)
            ph = Fstrategy(strategy,alpha,phase)
            Fstrategies.append(fs)
            Cstrategies.append(cs)
            Phases.append(ph)
        j+=1

    return Fstrategies,Phases

def return_fft_list(lx,x_,FFT_obj,width=20):

    fft_list = []
    fft_dummy = FFT_obj(lx,None,width=width)
    for wave in x_:
        spe = fft_dummy.make_spectrum(wave)
        fft_list.append(spe)
        
    return fft_list        

def norm(spectrum):
    N = len(spectrum)
    spectrum = spectrum / (N/2)
    return spectrum


def butter_lowpass(lowcut, fs, order=4):
    '''
    バターワースローパスフィルタを設計する関数
    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = signal.butter(order, low, btype='low')
    return b, a

def butter_lowpass_filter(x, lowcut, fs, order=4):
    '''データにローパスフィルタをかける関数
    '''
    b, a = butter_lowpass(lowcut, fs, order=order)
    y = signal.filtfilt(b, a, x)
    return y

def butter_highpass( highcut, fs, order=4):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = signal.butter(order, high, btype = "high", analog = False)
    return b, a

def butter_highpass_filter(self, x, highcut, fs, order=4):
    b, a = butter_highpass(highcut, fs, order=order)
    y = signal.filtfilt(b, a, x)
    return y