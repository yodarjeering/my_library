from importlib.resources import path
import numpy as np
import glob
import matplotlib.pyplot as plt
from enum import Enum
import pandas as pd
from collections import namedtuple
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from collections import namedtuple
from collections import deque
import tensorflow as tf
# from tensorflow.python import keras as K
import matplotlib.pyplot as plt
import random
random.seed(777)
import xgboost as xgb
from sklearn.metrics import classification_report,roc_auc_score
import pickle
import datetime
from func_library import *
from sim_library import *

Experience = namedtuple("Experience", ["s","a","r","n_s","n_a","d"])

class DataFramePreProcessing():

    
    def __init__(self, path_, is_daw=False):
        self.path_ = path_
        self.is_daw = is_daw

        
    def load_df(self):
        if self.is_daw:
            d='d'
        else:
            d=''
        FILE = glob.glob(self.path_)
        df = pd.read_csv(FILE[0])
        df = df.rename(columns={df.columns[0]:'nan',df.columns[1]:'nan',df.columns[2]:'nan',\
                                    df.columns[3]:'day',df.columns[4]:'nan',df.columns[5]:d+'open',\
                                    df.columns[6]:d+'high',df.columns[7]:d+'low',df.columns[8]:d+'close',\
                                       df.columns[9]:d+'volume',})
        df = df.drop('nan',axis=1)
        df = df.drop(df.index[0])
        df['day'] = pd.to_datetime(df['day'],format='%Y/%m/%d')
        df.set_index('day',inplace=True)

        return df.astype(float)
    
class PlotTrade():
    
    
    def __init__(self, df_chart,label=''):
        self.df_chart = df_chart
        plt.clf()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.plot(self.df_chart,label=label)
        plt.legend()
        
    def add_span(self, start_time,end_time):
        self.ax.axvspan(start_time, end_time, color="gray", alpha=0.3)
        
    
    def add_plot(self, df_plot,label=''):
        self.ax.plot(df_plot,label=label)
        plt.legend()
        
        
    def show(self):
        self.ax.grid()
        labels = self.ax.get_xticklabels()
        plt.setp(labels, rotation=15, fontsize=12)
        plt.show()
         
class ValidatePlot(PlotTrade):


    
    
    def __init__(self, df_chart, is_validate=False):
        pass
        
    def add_span(self, start_time,end_time):
        pass
        
    
    def add_plot(self, df_plot):
        pass
        
        
    def show(self):
        pass
    
class MakeTrainData():
    

    def __init__(self, df_con, test_rate=0.9, questions_index = [], is_bit_search=False,is_category=True,ma_short=5,ma_long=25):
        self.df_con = df_con
        self.test_rate = test_rate
        self.questions_index = questions_index
        self.is_bit_search = is_bit_search
        self.is_category = is_category
        self.ma_short = ma_short
        self.ma_long = ma_long
                
                
    def add_ma(self):
        df_process = self.df_con.copy()
        df_process['ma_short'] = df_process['close'].rolling(self.ma_short).mean()
        df_process['ma_long']  = df_process['close'].rolling(self.ma_long).mean()
        df_process['std_short'] = df_process['close'].rolling(self.ma_short).std()
        df_process['std_long']  = df_process['close'].rolling(self.ma_long).std()
        df_process['ema_short'] = df_process['close'].ewm(span=self.ma_short, adjust=False).mean()
        df_process['ema_long'] = df_process['close'].ewm(span=self.ma_long, adjust=False).mean()
        df_process['macd'] = df_process['ema_short'] - df_process['ema_long']
        df_process['macd_signal_short'] = df_process['macd'].ewm(span=self.ma_short, adjust=False).mean()
        df_process['macd_signal_long'] = df_process['macd'].ewm(span=self.ma_long, adjust=False).mean()
        return df_process
                
   
        
    def make_data(self,is_check=False):
        x = pd.DataFrame(index=self.df_con.index)
        y = []
        # この書き方は環境によってはエラー
        # x.index = self.df_con.index
        df_con = self.df_con.copy()
        df_ma = self.add_ma()
        end_point = -1
        if is_check:
            end_point = len(self.df_con)
        else:
            end_point = len(self.df_con)-1
        
        dawp_5 = df_con['dclose'].iloc[:-5]
        dawp_5.index = df_con.index[5:]
        x['dawp_5'] = dawp_5
#         dawp_4 = df_con['dclose'].iloc[:-4]
#         dawp_4.index = df_con.index[4:]
#         x['dawp_4'] = dawp_4
#         dawp_3 = df_con['dclose'].iloc[:-3]
#         dawp_3.index = df_con.index[3:]
#         x['dawp_3'] = dawp_3
#         dawp_2 = df_con['dclose'].iloc[:-2]
#         dawp_2.index = df_con.index[2:]
#         x['dawp_2'] = dawp_2
#         dawp_1 = df_con['dclose'].iloc[:-1]
#         dawp_1.index = df_con.index[1:]
#         x['dawp_1'] = dawp_1
        dawp_0 = df_con['dclose']
        x['dawp_0'] = dawp_0
        
        nikkeip_5 = df_con['pclose'].iloc[:-5]
        nikkeip_5.index = df_con.index[5:]
        x['nikkeip_5'] = nikkeip_5
        
#         nikkeip_4 = df_con['pclose'].iloc[:-4]
#         nikkeip_4.index = df_con.index[4:]
#         x['nikkeip_4'] = nikkeip_4
        
#         nikkeip_3 = df_con['pclose'].iloc[:-3]
#         nikkeip_3.index = df_con.index[3:]
#         x['nikkeip_3'] = nikkeip_3 
#         nikkeip_2 = df_con['pclose'].iloc[:-2]
#         nikkeip_2.index = df_con.index[2:]
#         x['nikkeip_2'] = nikkeip_2
#         nikkeip_1 = df_con['pclose'].iloc[:-1]
#         nikkeip_1.index = df_con.index[1:]
#         x['nikkeip_1'] = nikkeip_1
        nikkeip_0 = df_con['pclose']
        x['nikkeip_0'] = nikkeip_0
        
        high_low = (df_con['high']-df_con['low'])/df_con['close']
        x['diff_rate'] = high_low
        
        close_open = (df_con['close']-df_con['open'])/df_con['close']
        x['close_open'] = close_open
        
        nikkei_volumep = df_con['volume'].pct_change()
        x['nikkei_volumep'] = nikkei_volumep
        
        std_s_5 = df_ma['std_short'].iloc[:-5]
        std_s_5.index = df_ma.index[5:]
        x['std_s_5'] = std_s_5
#         std_s_4 = df_ma['std_short'].iloc[:-4]
#         std_s_4.index = df_ma.index[4:]
#         x['std_s_4'] = std_s_4
#         std_s_3 = df_ma['std_short'].iloc[:-3]
#         std_s_3.index = df_ma.index[3:]
#         x['std_s_3'] = std_s_3
#         std_s_2 = df_ma['std_short'].iloc[:-2]
#         std_s_2.index = df_ma.index[2:]
#         x['std_s_2'] = std_s_2
#         std_s_1 = df_ma['std_short'].iloc[:-1]
#         std_s_1.index = df_ma.index[1:]
#         x['std_s_1'] = std_s_1
        std_s_0 = df_ma['std_short']
        x['std_s_0'] = std_s_0
        
        
        std_l_5 = df_ma['std_long'].iloc[:-5]
        std_l_5.index = df_ma.index[5:]
        x['std_l_5'] = std_l_5
#         std_l_4 = df_ma['std_long'].iloc[:-4]
#         std_l_4.index = df_ma.index[4:]
#         x['std_l_4'] = std_l_4
#         std_l_3 = df_ma['std_long'].iloc[:-3]
#         std_l_3.index = df_ma.index[3:]
#         x['std_l_3'] = std_l_3
#         std_l_2 = df_ma['std_long'].iloc[:-2]
#         std_l_2.index = df_ma.index[2:]
#         x['std_l_2'] = std_l_2
#         std_l_1 = df_ma['std_long'].iloc[:-1]
#         std_l_1.index = df_ma.index[1:]
#         x['std_l_1'] = std_l_1
        std_l_0 = df_ma['std_long']
        x['std_l_0'] = std_l_0
        
# このままの変換だと, 相関しすぎているので, 変化率 or 基準の値で割るなど, 操作が必要
        vec_s_5 = (df_ma['ma_short'].diff(5)/5)
        x['vec_s_5'] = vec_s_5
#         vec_s_4 = (df_ma['ma_short'].diff(4)/4)
#         x['vec_s_4'] = vec_s_4
#         vec_s_3 = (df_ma['ma_short'].diff(3)/3)
#         x['vec_s_3'] = vec_s_3
#         vec_s_2 = (df_ma['ma_short'].diff(2)/2)
#         x['vec_s_2'] = vec_s_2
        vec_s_1 = (df_ma['ma_short'].diff(1)/1)
        x['vec_s_1'] = vec_s_1
        
    
        vec_l_5 = (df_ma['ma_long'].diff(5)/5)
        x['vec_l_5'] = vec_l_5
#         vec_l_4 = (df_ma['ma_long'].diff(4)/4)
#         x['vec_l_4'] = vec_l_4
#         vec_l_3 = (df_ma['ma_long'].diff(3)/3)
#         x['vec_l_3'] = vec_l_3
#         vec_l_2 = (df_ma['ma_long'].diff(2)/2)
#         x['vec_l_2'] = vec_l_2
        vec_l_1 = (df_ma['ma_long'].diff(1)/1)
        x['vec_l_1'] = vec_l_1
        
#         移動平均乖離率
        x['d_MASL'] = df_ma['ma_short']/df_ma['ma_long']
#             ema のベクトル

        emavec_s_5 = (df_ma['ema_short'].diff(5)/5)
        x['emavec_s_5'] = emavec_s_5
#         emavec_s_4 = (df_ma['ema_short'].diff(4)/4)
#         x['emavec_s_4'] = emavec_s_4
#         emavec_s_3 = (df_ma['ema_short'].diff(3)/3)
#         x['emavec_s_3'] = emavec_s_3
#         emavec_s_2 = (df_ma['ema_short'].diff(2)/2)
#         x['emavec_s_2'] = emavec_s_2
        emavec_s_1 = (df_ma['ema_short'].diff(1)/1)
        emavec_s_1.index = df_ma.index
        x['emavec_s_1'] = emavec_s_1
    
        emavec_l_5 = (df_ma['ema_long'].diff(5)/5)
        x['emavec_l_5'] = emavec_l_5
#         emavec_l_4 = (df_ma['ema_long'].diff(4)/4)
#         x['emavec_l_4'] = emavec_l_4
#         emavec_l_3 = (df_ma['ema_long'].diff(3)/3)
#         x['emavec_l_3'] = emavec_l_3
#         emavec_l_2 = (df_ma['ema_long'].diff(2)/2)
#         x['emavec_l_2'] = emavec_l_2
        emavec_l_1 = (df_ma['ema_long'].diff(1)/1)
        x['emavec_l_1'] = emavec_l_1

        #         EMA移動平均乖離率
        x['d_EMASL'] = df_ma['ema_short']/df_ma['ema_long']
        
        macd = df_ma['macd']
        x['macd'] = macd
        macd_signal_short = df_ma['macd_signal_short']
        x['macd_signal_short'] = macd_signal_short
        macd_signal_long = df_ma['macd_signal_long']
        x['macd_signal_long'] = macd_signal_long
            
        
        df_tmp1 = df_con[['close','daw_close']].rolling(self.ma_short).corr()
        corr_short = df_tmp1.drop(df_tmp1.index[0:-1:2])['close']
        corr_short = corr_short.reset_index().set_index('day')['close']
        x['corr_short'] = corr_short
        
        
        df_tmp2 = df_con[['close','daw_close']].rolling(self.ma_long).corr()
        corr_long = df_tmp2.drop(df_tmp2.index[0:-1:2])['close']
        corr_long = corr_long.reset_index().set_index('day')['close']
        x['corr_long'] = corr_long
        
        
        skew_short = df_con['close'].rolling(self.ma_short).skew()
        x['skew_short'] = skew_short
        skew_long = df_con['close'].rolling(self.ma_long).skew()
        x['skew_long'] = skew_long
        
        
        kurt_short = df_con['close'].rolling(self.ma_short).kurt()
        x['kurt_short'] = kurt_short
        kurt_long = df_con['close'].rolling(self.ma_long).kurt()
        x['kurt_long'] = kurt_long
        
        
        df_up = df_con['dclose'].copy()
        df_down = df_con['dclose'].copy()
        df_up[df_up<0] = 0
        df_down[df_down>0] = 0
        df_down *= -1
        sims_up = df_up.rolling(self.ma_short).mean()
        sims_down = df_down.rolling(self.ma_short).mean()
        siml_up = df_up.rolling(self.ma_long).mean()
        siml_down = df_down.rolling(self.ma_long).mean()
        RSI_short = sims_up / (sims_up + sims_down) * 100
        RSI_long = siml_up / (siml_up + siml_down) * 100
        x['RSI_short'] = RSI_short
        x['RSI_long'] = RSI_long
        
        
        open_ =  df_con['open']
        high_ = df_con['high']
        low_ = df_con['low']
        close_ = df_con['close']
#         Open Close 乖離率
#         x['d_OC'] = open_/close_
#     High low 乖離率
# ATR の計算ミスってた 5/31
        x['d_HL'] = high_/low_
        df_atr = pd.DataFrame(index=high_.index)
        df_atr['high_low'] = high_ - low_
        df_atr['high_close'] = high_ - close_
        df_atr['close_low_abs'] =  (close_ - low_).abs()
        tr = pd.DataFrame(index=open_.index)
        tr['TR'] = df_atr.max(axis=1)


#         x['ATR_short'] = tr['TR'].rolling(self.ma_short).mean()
#         x['ATR_long'] =  tr['TR'].rolling(self.ma_long).mean()
# #         ATR乖離率
#         x['d_ATR'] = x['ATR_short']/x['ATR_long']
#         x['ATR_vecs5'] = (x['ATR_short'].diff(5)/1)
#         x['ATR_vecs1'] = (x['ATR_short'].diff(1)/1)
#         x['ATR_vecl5'] = (x['ATR_long'].diff(5)/1)
#         x['ATR_vecl1'] = (x['ATR_long'].diff(1)/1)
        
        today_close = df_con['close']
        yesterday_close = df_con['close'].iloc[:-1]
        yesterday_close.index = df_con.index[1:]
#         騰落率
#         x['RAF'] =  (today_close/yesterday_close -1)

        
#         coef_short tmp1
        x = x.iloc[self.ma_long:end_point]
        x_check = x
#         この '4' は　std_l5 など, インデックスをずらす特徴量が, nanになってしまう分の日数を除くためのもの
# yについても同様
        x_train = x.iloc[self.ma_short-1:int(len(x)*self.test_rate)]
        x_test  = x.iloc[int(len(x)*self.test_rate):]


        if not is_check:
            for i in range(self.ma_long,end_point):
                tommorow_close = self.df_con['close'].iloc[i+1]
                today_close    = self.df_con['close'].iloc[i]
                if tommorow_close>today_close:
                    y.append(1)
                else:
                    y.append(0)
        
            
            y_train = y[self.ma_short-1:int(len(x)*self.test_rate)]
            y_test  = y[int(len(x)*self.test_rate):]
            return x_train, y_train, x_test, y_test
        
        
        else:
            x_check = x_check.iloc[self.ma_short-1:]
            chart_ = self.df_con.loc[x_check.index]
            
            return x_check,chart_

class LearnXGB():
    
    
    def __init__(self):
        self.model = xgb.XGBClassifier()
        self.x_test = None
    
    
    def learn_xgb(self, path_tpx, path_daw, test_rate=0.8, param_dist='None'):
        x_train,y_train,x_test,y_test = self.make_xgb_data(path_tpx,path_daw,test_rate)
        
        if param_dist=='None':
#             Grid search で求めたパラメタ 2021/11/21
            param_dist = { 'n_estimators':16,'use_label_encoder':False,
                 'max_depth':4}

        xgb_model = xgb.XGBClassifier(**param_dist)
        hr_pred = xgb_model.fit(x_train.astype(float), np.array(y_train), eval_metric='logloss').predict(x_test.astype(float))
        print("---------------------")
        y_proba_train = xgb_model.predict_proba(x_train)[:,1]
        y_proba = xgb_model.predict_proba(x_test)[:,1]
        print('AUC train:',roc_auc_score(y_train,y_proba_train))    
        print('AUC test :',roc_auc_score(y_test,y_proba))
        print(classification_report(np.array(y_test), hr_pred))
        _, ax = plt.subplots(figsize=(12, 10))
        xgb.plot_importance(xgb_model,ax=ax) 
        self.model = xgb_model
        

    def make_state(self,path_tpx,path_daw):
        df_con = self.make_df_con(path_tpx,path_daw)
        mk = MakeTrainData(df_con)
        x_check, chart_ = mk.make_data(is_check=True)
        state_ = self.model.predict_proba(x_check.astype(float))
        chart_ = df_con['close'].loc[x_check.index[0]:x_check.index[-1]]
        return state_, chart_
        
        
    def make_xgb_data(self, path_tpx, path_daw, test_rate):
        df_con = self.make_df_con(path_tpx,path_daw)
        mk = MakeTrainData(df_con,test_rate=test_rate)
        x_train, y_train, x_test, y_test = mk.make_data()
        return x_train,y_train,x_test,y_test
    
    
    def make_df_con(self,path_tpx,path_daw):
        df_tpx = DataFramePreProcessing(path_tpx).load_df()
        df_daw = DataFramePreProcessing(path_daw,is_daw=True).load_df()
        daw_p = df_daw.pct_change()
        tpx_p = df_tpx.pct_change()
        tpx_p = tpx_p.rename(columns={'close':'pclose'})
        df_daw = df_daw.rename(columns={'dopen':'daw_close'})
        df_con = pd.concat([df_daw['daw_close'],df_tpx,daw_p['dclose']],axis = 1,join='inner').astype(float)
        df_con['pclose'] = df_con['close'].pct_change()
        df_con = df_con.drop(df_con[ df_con['volume']==0].index)
        return df_con
    
    
    def make_check_data(self, path_tpx, path_daw):
        df_con = self.make_df_con(path_tpx,path_daw)

        mk = MakeTrainData(df_con)
        x_check, chart_ = mk.make_data(is_check=True)
        state_ = self.model.predict_proba(x_check.astype(float))

        chart_ = mk.df_con['close'].loc[x_check.index[0]:x_check.index[-1]]
        state_ = pd.DataFrame(state_)
        state_['day'] = chart_.index
        
        state_.reset_index(inplace=True)
        state_.set_index('day',inplace=True)
        state_.drop('index',axis=1,inplace=True)
        return state_, chart_
    
    
    def predict_tomorrow(self, path_tpx, path_daw, alpha=0.5, strategy='normal', is_online=False, start_year=2021,start_month=1,end_month=12,is_observed=False,is_validate=False):
        xl = XGBSimulation(self.model,alpha=alpha)
        xl.simulate(path_tpx,path_daw,is_validate=is_validate,strategy=strategy,start_year=start_year,start_month=start_month,end_month=end_month,is_observed=is_observed)
        self.xl = xl
        df_con = self.make_df_con(path_tpx,path_daw)
        mk = MakeTrainData(df_con)
        x_check, chart_ = mk.make_data(is_check=True)
        tomorrow_predict = self.model.predict_proba(x_check)
        print("is_bought",xl.is_bought)
        print("df_con in predict_tomorrow",df_con.index[-1])
        print("today :",x_check.index[-1])
        print("tomorrow UP possibility", tomorrow_predict[-1,1])
        
class StrategyMaker():
    
    
#     騰貴下落判断用XGBは作成済み仮定
    def __init__(self,lx):
        self.lx = lx
        self.ma_short = -1
        self.ma_long = -1
        self.model = None
    
#     StrategyMakerの学習データ用の関数
    def _predict_proba(self,path_tpx,path_daw):
        df_con = self.lx.make_df_con(path_tpx,path_daw)
        mk = MakeTrainData(df_con)
        self.ma_short = mk.ma_short
        self.ma_long = mk.ma_long
        x_check, chart_ = mk.make_data(is_check=True)
        proba_ = self.lx.model.predict_proba(x_check.astype(float))
        df_proba = pd.DataFrame(proba_)
        df_proba.index = chart_.index
        return df_proba,chart_,x_check
    
    
    def return_split_data(self,df,year):
        return df[df.index.year>=year]
    
    
    def return_column_df(self, df_base, df_column,column_name):
        df_base[column_name] = df_column
        return df_base
    
#     訓練用データ作成
    def make_train_data(self, path_tpx, path_daw,year=2019,theta=0.0001):
        df_proba,chart_,x_check = self._predict_proba(path_tpx,path_daw)
        up_possibility = self.return_split_data(df_proba[1],year=year)
        moving_average_short = self.return_split_data(chart_['close'].rolling(self.ma_short).mean(),year=year)
        moving_average_long = self.return_split_data(chart_['close'].rolling(self.ma_long).mean(),year=year)
        tmp_ma_short = chart_['close'].rolling(self.ma_short).mean()
        tmp_ma_long = chart_['close'].rolling(self.ma_long).mean()
        grad_short = self.return_split_data(tmp_ma_short.pct_change(),year=year)
        grad_long = self.return_split_data(tmp_ma_long.pct_change(),year=year)
        std_short = self.return_split_data(chart_['close'].rolling(self.ma_short).std(),year=year)
        std_long = self.return_split_data(chart_['close'].rolling(self.ma_long).std(),year=year)
        df_proba = self.return_split_data(df_proba,year=year)
        chart_ = self.return_split_data(chart_,year=year)
        x_check = self.return_split_data(x_check,year=year)
        
        predict_ = self.lx.model.predict(x_check)
#         predict_ : np.array
        is_bought = False
        prf = 0
        index_buy = 0
        y_ = np.array([-1 for i in range(len(x_check))])
        is_hold = False
#         y_ : answer
#         y_ = 0 -> 買わず, y_ = 1 -> 買う
        
        for i in range(len(x_check)-1):
            buy_label = predict_[i]
            index_buy = chart_['close'].iloc[i]
            prf = 0
            
            
            if buy_label == 0:
                sell_label = 1
                for j in range(i+1,len(x_check)):
                    
                    
                    if sell_label==predict_[j]:
                        index_sell = chart_['close'].iloc[j]
                        prf = index_sell - index_buy
                        break
                    if j == len(x_check)-1:
                        is_hold = True
            else: # lable==1:
                sell_label = 0
                for j in range(i+1,len(x_check)):
                    
                    
                    if sell_label==predict_[j]:
                        index_sell = chart_['close'].iloc[j]
                        prf = index_sell - index_buy
                        break
                    if j == len(x_check)-1:
                        is_hold = True
                    
#             ラベル付作業
#             一定以上の収益を上げられたら, 「買い」 のサイン
            current_price = chart_['close'].iloc[i]
            if is_hold:
                continue
            else:
#                 時価に対してどれだけの利益か, それを超えたら良い取引
                if prf > theta*current_price:
                    y_[i] = 1
                else:
                    y_[i] = 0
        x_ = pd.DataFrame()
        x_ = self.return_column_df(x_,up_possibility,'up_possibility')
#         x_ = self.return_column_df(x_,moving_average_short,'moving_average_short')
#         x_ = self.return_column_df(x_,moving_average_long,'moving_average_long')
        x_ = self.return_column_df(x_,grad_short,'grad_short')
        x_ = self.return_column_df(x_,grad_long,'grad_long')
        x_ = self.return_column_df(x_,std_short,'std_short')
        x_ = self.return_column_df(x_,std_long,'std_long')
#         x_ = self.return_column_df(x_,df_proba[1],'df_proba')
        y_ = pd.DataFrame(y_)
        y_.index = x_.index
#         とりあえず, 移動平均の勾配は前日変化率を用いて, 算出する
#*         将来的には, n日変化率とか希望

        return x_, y_
    
    
#     学習
    def learn(self,path_tpx, path_daw,train_year=2019,test_year=2021,theta=0.0001):
        x_, y_ = self.make_train_data(path_tpx=path_tpx, path_daw=path_daw,year=train_year,theta=theta)
        x_train = x_[x_.index.year>=train_year]
        y_train = y_[y_.index.year>=train_year]
        x_train = x_train[x_train.index.year<test_year]
        y_train = y_train[y_train.index.year<test_year]
        y_test = y_[y_.index.year==test_year]
        y_test = y_test[y_test[0]!=-1]
        x_test = x_[x_.index.year==test_year]
        x_test = x_test.loc[y_test.index]
        self.model = xgb_pred(x_train, y_train, x_test, y_test)
        
class Action(Enum):
    BUY  = -1
    STAY = 0
    SELL = 1
    
class Environment():
    
    
    def __init__(self, x_train, price_chart):
        self.x_train = x_train # state list
        self.is_holding = False
        self.time = 0 # x_trainのindex
        self.price_chart = price_chart
        self.bought_price = 0
        
        
    def reset(self):
        self.time=0
        self.is_holding = False
        self.bought_price = 0
        return self.x_train.iloc[self.time].tolist()
        
    
    def actions(self):
        return [Action.BUY, Action.STAY, Action.SELL]
    
    
#* ERROR!!!
    def state(self):
        try:
            return self.x_train.iloc[self.time].tolist()
        except:
            print("Error index :",self.time)
            print("length      :",len(self.x_train))
            print("x_train:",self.x_train)
    
    def reward_func(self, action):
        reward=0
#* for solving the state(self)'s error, self.time -3 . 
        if self.time >= len(self.price_chart)-2: #売り切らずにエピソードを終えた時は評価額を報酬とする
            
            
            if self.is_holding:
                reward = self.price_chart.iloc[self.time+1] - self.bought_price
            else:
                reward = 0
                
            return reward, True
        
        
        else:     
            
       
            if action==Action.BUY:
                reward=0
                if not self.is_holding:
                    self.is_holding = True
                    self.bought_price = self.price_chart.iloc[self.time+1]
            
            elif action==Action.STAY:
                reward=0
            
            elif action==Action.SELL:
                reward = 0
                if self.is_holding:
                    reward = self.price_chart.iloc[self.time+1] - self.bought_price
                    self.bought_price = 0
                    self.is_holding=False
            
            return reward, False
            
    
    
    def step(self, action):
        reward, done = self.reward_func(action)
        self.time += 1
        next_state = self.state()
        return next_state, reward, done
    

    def func(self,x):
        return x
    
class SigmoidEnv(Environment):

    
    def sigmoid(self, x_):
        return 1/(1+np.exp(-x_))
    
    
    def func(self,x):
        return self.sigmoid(x)
    
     
    def reward_func(self, action):
        reward=0
        
        if self.time >= len(self.price_chart)-2: #売り切らずにエピソードを終えた時は評価額を報酬とする
            
            
            if self.is_holding:
                reward = self.price_chart.iloc[self.time+1] - self.bought_price
            else:
                reward = 0
                
            return 2*self.sigmoid(reward)-1, True
        
        
        else:     
            
            
            if action==Action.BUY:
                reward=0
                if not self.is_holding:
                    self.is_holding = True
                    self.bought_price = self.price_chart.iloc[self.time+1]
            
            elif action==Action.STAY:
                reward=0
            
            elif action==Action.SELL:
                reward = 0
                if self.is_holding:
                    reward = self.price_chart.iloc[self.time+1] - self.bought_price
                    self.bought_price = 0
                    self.is_holding=False
            
            return 2*self.sigmoid(reward)-1, False
        
class TanhEnv(Environment):

    
    def func(self,x):
        return np.tanh(x)
    
 
    def reward_func(self, action):
        reward=0
        
        if self.time >= len(self.price_chart)-2: #売り切らずにエピソードを終えた時は評価額を報酬とする
            
            
            if self.is_holding:
                reward = self.price_chart.iloc[self.time+1] - self.bought_price
            else:
                reward = 0
                
            return np.tanh(reward), True
        
        
        else:     
            
            
            if action==Action.BUY:
                reward=0
                if not self.is_holding:
                    self.is_holding = True
                    self.bought_price = self.price_chart.iloc[self.time+1]
            
            elif action==Action.STAY:
                reward=0
            
            elif action==Action.SELL:
                reward = 0
                if self.is_holding:
                    reward = self.price_chart.iloc[self.time+1] - self.bought_price
                    self.bought_price = 0
                    self.is_holding=False
            
            return np.tanh(reward), False

class FNAgent():
    
    
    def __init__(self, epsilon, actions):
        self.epsilon = epsilon
        self.actions = actions
        self.model = None
        self.estimate_probs = False
#         self.estimate_probs = True
        self.initialized = False
#         学習が終わったら, ε-greedy法をやめるためのフラグ変数
        self.is_test = False
        self.is_sarsa = False
        
        
    def save(self, model_path):
        self.model.save(model_path, overwrite=True, include_optimizer=False)
        
        
    def policy(self, s):# 買ってたら戦略が変わる

        if (np.random.random() < self.epsilon or not self.initialized) and not self.is_test:
            return np.random.randint(len(self.actions))
        else:
            estimates = self.estimate(s)
            if self.estimate_probs:
                return np.random.choice(self.actions,size=1, p=softmax(estimates))[0]
            else:
                return np.argmax(estimates)
        
    # @classmethod
    # def load(cls, env, model_path, epsilon=0.0001):
    #     actions = list(range(len(Action)))
    #     agent = cls(epsilon, actions)
    #     agent.model = K.models.load_model(model_path)
    #     agent.initialized = True
    #     return agent
    
    
    def initialize(self, experiences):
        pass
        
        
    def estimate(self, s):
        pass
        
        
    def update(self, experiences, gamma):
        pass
        
        
    def play(self, env, episode_count=1):
        pass
    
class ValueFunctionAgent(FNAgent):

    
    # @classmethod
    # def load(cls, env, model_path, epsilon=0.0001):
    #     actions = list(range(len(Action)))
    #     agent = cls(epsilon, actions)
    #     agent.model = joblib.load(model_path)
    #     agent.initialized = True
    #     agent.is_test = True
    #     return agent

    
    def initialize(self, experiences):
        scaler = StandardScaler() # 特徴料(列)ごとに標準化してる
#*         estimator = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1)
        estimator = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1, early_stopping=True)
        estimator.best_loss_ = 10**5
#*         self.model = Pipeline([("scaler", scaler), ("estimator", estimator)])
#*       自作関数で, state_　標準化済み
        self.model = estimator
        states = np.vstack([e.s for e in experiences])
#*         self.model.named_steps["scaler"].fit(states)

        
        if not self.is_sarsa:
            self.update([experiences[0]], gamma=0)
        else:
            self.update_sarsa([experiences[0]], gamma=0)
        self.initialized = True
        print("Done initialization. From now, begin training!")

    
    def estimate(self, s):
        s = np.array(s).reshape(1,-1)
        # standard scaler してないけどいいのか？
        estimated = self.model.predict(s)[0]
        return estimated

    
    def _predict(self, states):
        if self.initialized:
            predicteds = self.model.predict(states)
        else:
            size = len(self.actions) * len(states)
            predicteds = np.random.uniform(size=size)
            predicteds = predicteds.reshape((-1, len(self.actions)))
        return predicteds

    
    def update(self, experiences, gamma):        
        states = np.vstack([e.s for e in experiences])
        n_states = np.vstack([e.n_s for e in experiences])

        estimateds = self._predict(states)
        future = self._predict(n_states)


        for i, e in enumerate(experiences):
            reward = e.r
            if not e.d:
                reward += gamma * np.max(future[i])
            estimateds[i][e.a] = reward

        estimateds = np.array(estimateds)
        self.model.partial_fit(states, estimateds)
#*         states = self.model.named_steps["scaler"].transform(states)
#*         self.model.named_steps["estimator"].partial_fit(states, estimateds)
    

#         ******************* SARSA法
    def update_sarsa(self,experiences, gamma):
        states = np.vstack([e.s for e in experiences])
        n_states = np.vstack([e.n_s for e in experiences])

        estimateds = self._predict(states)
        future = self._predict(n_states)

        for i, e in enumerate(experiences):
            reward = e.r
            if not e.d:
                reward += gamma * future[i][e.n_a]
            estimateds[i][e.a] = reward

        
        estimateds = np.array(estimateds)
        self.model.partial_fit(states, estimateds)
#*         states = self.model.named_steps["scaler"].transform(states)
#*         self.model.named_steps["estimator"].partial_fit(states, estimateds)

        
    def play(self, env, episode_count=1,is_validate=False):
        actions = env.actions()
#       学習が終わったら, ε-greedy法をやめる
        self.pr_log = pd.DataFrame(index=env.price_chart.index)
        # self.pr_log.index = env.price_chart.index
        self.pr_log['reward'] = [0.0] * len(self.pr_log)
        self.pr_log['eval_reward'] = self.pr_log['reward'].tolist()
        eval_price = 0
        total_eval_price = 0
        if self.initialized:
            self.is_test = True
            
            
        for e in range(episode_count):

            s = env.reset()
            done = False
            total_reward = 0
            reward_log = []
            if not is_validate:
                pl = PlotTrade(env.price_chart)
            else:
                pl = ValidatePlot(None,is_validate=is_validate)
            trade_count = 0
            is_bought = False
            start_time = env.price_chart.index[0]
            end_time = env.price_chart.index[0]
                
            
            while not done:
                a = self.policy(s)
                action = actions[a]
                n_state, reward, done = env.step(action)
                total_reward += reward    
#                 *******
                total_eval_price = total_reward
    
                self.pr_log['reward'].iloc[env.time] = total_reward
                self.pr_log['eval_reward'].iloc[env.time] = total_eval_price
                s = n_state
                reward_log.append(total_reward)
                #============ render ============

                
                if not is_bought:
                    if env.is_holding and action == Action.BUY:
                        start_time = env.price_chart.index[env.time]
                        is_bought = True
#                         ******
                        eval_price = env.price_chart.iloc[env.time] - env.bought_price
                        total_eval_price += eval_price
                        self.pr_log['eval_reward'].iloc[env.time] = total_eval_price
                else:
                    if not env.is_holding and action == Action.SELL:
                        end_time = env.price_chart.index[env.time]
                        is_bought = False
                        pl.add_span(start_time,end_time)
                        trade_count += 1
                    elif env.is_holding:
                        eval_price = env.price_chart.iloc[env.time] - env.bought_price
                        total_eval_price += eval_price
                        self.pr_log['eval_reward'].iloc[env.time] = total_eval_price
                        
                        
            else:
                self.reward_log = reward_log
                self.pr_log['reward'].iloc[-1] = total_reward
                self.pr_log['eval_reward'].iloc[-1] = total_eval_price
                if is_bought and env.is_holding:
                    end_time = env.price_chart.index[-1]
#                     **これが悪さしている気がしてならない
                    eval_price = env.price_chart.iloc[-1] - env.price_chart.iloc[-2]
                    self.pr_log['eval_reward'].iloc[-1] = self.pr_log['eval_reward'].iloc[-2] + eval_price
                    pl.add_span(start_time,end_time)
                    trade_count+=1
                
                    
                if not is_validate:
                    print("==================")
                    print("episode :",e) 
                    print("Get reward {}.".format(total_reward))
                    print("Trade count {}.".format(trade_count))
                    print("Tomorrow action :",action)
                    pl.show()
                return  total_reward, trade_count
            
    
    def return_profit_rate(self,env_check,wallet=2500):
        #         wallet      : 元本のこと 2500は25万円のこと
        #         reward      : 実現収益
        #         eval_reward : 評価損益も含んだ収益率 
        self.play(env_check,is_validate=True)
        self.pr_log['reward'] = self.pr_log['reward'].map(lambda x: x/wallet)
        self.pr_log['eval_reward'] = self.pr_log['eval_reward'].map(lambda x: x/wallet)
        return self.pr_log
            
        
    def return_trade_log(self):
        return self.reward_log
            
class Trainer():

#     buffer_size = 1024 -> 4096二増やした
    def __init__(self, buffer_size=4096, batch_size=32,gamma=0.9, teacher_update_freq=3,patience=500):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.experiences = deque(maxlen=buffer_size)
        self.training = False
        self.training_count = 0
        self.teacher_update_freq = teacher_update_freq
        self.loss = 0
        self.training_episode = 0
#*         何回最高rewardを更新しなかったらstopさせるか
        self.patience = patience
        self.best_rewards = 0
        self.best_model = None
        self.check_point = 0
        self.initial_epsilon = 0.1
        self.final_epsilon = 0.001
        
        
#* EarlyStopping 実装
    def train_loop(self, env,agent, episode=200, initial_count=-1,env_sample=None,is_sarsa=False):
        self.experiences = deque(maxlen=self.buffer_size)
        self.training = False
        self.training_count = 0
        actions = env.actions()
        reward_log = []
        validate_reward_log = []
        self.is_sarsa=is_sarsa
#*      best_rewardを更新したとき, best_modelも更新する
        best_rewards = -10**7
        best_model = None
        check_point = 0 # type(episode)
        update_count = 0
        
        
#         is_bought とか, bought_price とかが必要
        
        for i in range(episode):
            s = env.reset()
            done = False
            step_count = 0
            self.episode_begin(i, agent)
            total_rewards = 0
            is_bought = False
            eval_price = 0
            
            
            if i%100==0:
                print("----------------------")
                print("episode :",i)
            
            while not done:
                

                a = agent.policy(s)
                action = actions[a]
                n_state, reward, done = env.step(action)
                n_a = agent.policy(n_state)
#               評価損益を reward にあたえる
                eval_price = 0
#               評価損益rewardにかけるバイアス
                bias = 0.5
                
                if not is_bought:
                    if env.is_holding and action == Action.BUY:
                        start_time = env.price_chart.index[env.time]
                        is_bought = True
#                         ******
                        eval_price = env.price_chart.iloc[env.time] - env.bought_price
                else:
                    if not env.is_holding and action == Action.SELL:
                        end_time = env.price_chart.index[env.time]
                        is_bought = False
                    elif env.is_holding:
#                         評価損益の計算はここで
                        eval_price = env.price_chart.iloc[env.time] - env.bought_price
                        
        

        # """ 評価損益をAgentに観測させる場合 """

                if reward == 0:
#                     env.func とおして, 与える
                    eval_price *= bias
                    reward = env.func(eval_price)
        
                e = Experience(s, a, reward, n_state, n_a, done)
                total_rewards += reward
                self.experiences.append(e)
                        
            
                
                if not self.training and len(self.experiences) == self.buffer_size:
                    self.begin_train(i, agent)
                    self.training = True

                self.step(i, step_count, agent, e,is_sarsa)

                s = n_state
                step_count += 1
            else:
                self.episode_end(i, step_count, agent)
                reward_log.append(total_rewards)
#         検証用のlog
                if env_sample!=None:
                    validate_reward, dummy = agent.play(env_sample,is_validate=True)
                    validate_reward_log.append(validate_reward)
                if not self.training and initial_count > 0 and i >= initial_count:
                    self.begin_train(i, agent)
                    self.training = True


                if self.training:
                    self.training_count += 1
                
#*       best_reward, best_modelの更新                  
                if best_rewards < total_rewards and i>=20:
                    check_point = i
                    best_rewards = total_rewards
                    best_model = agent.model
                    update_count += 1
                    self.check_point = check_point
                    self.best_rewards = best_rewards
                    self.best_model = best_model
##        early stopping 実装
                if check_point + self.patience <= i:
                    print("Done Early Stopping")
                    print("check_point episode :",check_point)
                    print("update_count        :",update_count)
                    print("best rewards        :",best_rewards)
                    agent.model = best_model
                    break
                
        agent.model = self.best_model
        print("train reward")         
        plt.clf() 
        plt.plot(reward_log)
        plt.xlabel('episode')
        plt.ylabel('reward')
        plt.show()
        
        
        if env_sample!=None:
            print("validate data reward")
            plt.clf() 
            plt.plot(validate_reward_log)
            plt.xlabel('episode')
            plt.ylabel('reward')
            plt.show()
        
        self.reward_log = reward_log
        self.validate_reward_log = validate_reward_log
        
        
    def return_reward_log(self):
        return self.reward_log, self.validate_reward_log
    
    
    def episode_begin(self, episode, agent):
        self.loss = 0

    
    def begin_train(self, episode, agent):
        pass

    
    def step(self, episode, step_count, agent, experience):
        pass

    
    def episode_end(self, episode, step_count, agent):
#         減衰探索
        diff = (self.initial_epsilon - self.final_epsilon)
        decay = diff / self.training_episode
        agent.epsilon = max(agent.epsilon -decay, self.final_epsilon)


    
    def is_event(self, count, interval):
        return True if count != 0 and count % interval == 0 else False

    
    def get_recent(self, count):
        recent = range(len(self.experiences) - count, len(self.experiences))
        return [self.experiences[i] for i in recent]
    
class ValueFunctionTrainer(Trainer):

    
    def train(self, env, episode_count=250, epsilon=0.1, initial_count=-1,env_sample=None,is_sarsa=False):
        actions = list(range(len(Action)))
        
        agent = ValueFunctionAgent(epsilon, actions)
        self.training_episode = episode_count
        self.train_loop(env, agent, episode_count, initial_count,env_sample,is_sarsa)
        return agent

   
    def begin_train(self, episode, agent):
        agent.initialize(self.experiences)

    
    def step(self, episode, step_count, agent, experience,is_sarsa):
        if self.training:
#*             ここでPrioritized Experience Replay 実装？
            batch = random.sample(self.experiences, self.batch_size)
            if not is_sarsa:
                agent.update(batch, self.gamma)
            else:
                agent.update_sarsa(batch,self.gamma)
                          
class ImitationTrainer(ValueFunctionTrainer):
    
    
    def __init__(self,buffer_size=1024,batch_size=32,teacher_agent=None):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.teacher_agent = teacher_agent
        super().__init__(buffer_size,batch_size)
        
    
    def train_loop(self, env,agent, episode=200, initial_count=-1,env_sample=None,is_sarsa=False):
        self.experiences = deque(maxlen=self.buffer_size)
        self.training = False
        self.training_count = 0
        actions = env.actions()
        reward_log = []
        validate_reward_log = []
#         教師データ 100episodeとした
        initial_episode=100
        teacher_agent = self.teacher_agent
#*      best_rewardを更新したとき, best_modelも更新する
        best_rewards = -10**7
        best_model = None
        check_point = 0 # type(episode)
        update_count = 0
        self.is_sarsa=is_sarsa
        estimateds = []
        
        for ie in range(initial_episode):
            s = env.reset()
            done = False
            while not done:
                

                a = teacher_agent.policy(s)
                action = actions[a]
                n_state, reward, done = env.step(action)
                n_a = teacher_agent.policy(n_state)
                e = Experience(s, a, reward, n_state, n_a, done)
                self.experiences.append(e)
                estimateds.append(a)
#         *****estimateds list型だけど大丈夫か？
        self.begin_train(i, agent)
        self.training = True
        #******** ここ要注意
        states = np.vstack([e.s for e in self.experiences])
        self.model.partial_fit(states, estimateds)

        
        for i in range(episode):
            s = env.reset()
            done = False
            step_count = 0
            self.episode_begin(i, agent)
            total_rewards = 0
            is_bought = False
            eval_price = 0
            
            
            if i%100==0:
                print("----------------------")
                print("episode :",i)
            
            while not done:
                

                a = agent.policy(s)
                action = actions[a]
                n_state, reward, done = env.step(action)
                n_a = agent.policy(n_state)
#               評価損益を reward にあたえる
                eval_price = 0
#               評価損益rewardにかけるバイアス
                bias = 0.5
                
                if not is_bought:
                    if env.is_holding and action == Action.BUY:
                        start_time = env.price_chart.index[env.time]
                        is_bought = True
#                         ******
                        eval_price = env.price_chart.iloc[env.time] - env.bought_price
                else:
                    if not env.is_holding and action == Action.SELL:
                        end_time = env.price_chart.index[env.time]
                        is_bought = False
                    elif env.is_holding:
#                         評価損益の計算はここで
                        eval_price = env.price_chart.iloc[env.time] - env.bought_price
                        
        

        # """ 評価損益をAgentに観測させる場合 """

                if reward == 0:
#                     env.func とおして, 与える
                    eval_price *= bias
                    reward = env.func(eval_price)
        
                e = Experience(s, a, reward, n_state, n_a, done)
                total_rewards += reward
                self.experiences.append(e)
                self.step(i, step_count, agent, e,is_sarsa)
                s = n_state
                step_count += 1
            else:
                self.episode_end(i, step_count, agent)
                reward_log.append(total_rewards)
#         検証用のlog
                if env_sample!=None:
                    validate_reward, dummy = agent.play(env_sample,is_validate=True)
                    validate_reward_log.append(validate_reward)
                if not self.training and initial_count > 0 and i >= initial_count:
                    self.begin_train(i, agent)
                    self.training = True


                if self.training:
                    self.training_count += 1
                
#*       best_reward, best_modelの更新                  
                if best_rewards < total_rewards and i>=20:
                    check_point = i
                    best_rewards = total_rewards
                    best_model = agent.model
                    update_count += 1
                    self.check_point = check_point
                    self.best_rewards = best_rewards
                    self.best_model = best_model
##        early stopping 実装
                if check_point + self.patience <= i:
                    print("Done Early Stopping")
                    print("check_point episode :",check_point)
                    print("update_count        :",update_count)
                    print("best rewards        :",best_rewards)
                    agent.model = best_model
                    break
                

        print("train reward")         
        plt.clf() 
        plt.plot(reward_log)
        plt.xlabel('episode')
        plt.ylabel('reward')
        plt.show()
        
        
        if env_sample!=None:
            print("validate data reward")
            plt.clf() 
            plt.plot(validate_reward_log)
            plt.xlabel('episode')
            plt.ylabel('reward')
            plt.show()
        
        self.reward_log = reward_log
        self.validate_reward_log = validate_reward_log
      
class LearnQN():

    
    def __init__(self,lx):
        self.lx = lx
#         ***
        self.trainer = None
        self.QL_agent = None
        self.state = None
        self.env_type = None
        self.ma_long = 0
        self.ma_short = 0
        
        
    def save(self, save_path):
        date_ = datetime.datetime.now().strftime('%Y%m%d')
        state=''
        
        
        if type(self.state)==list:
            for i in self.state:
                state += (i+'_')
        else:
            state = self.state
        
        
        name = 'xqn'+'_'+state+'_'+self.env_type+'_'+date_
        with open(save_path+name+'.pickle', 'wb') as f:
            pickle.dump(self, f)
    
    
    @classmethod
    def load(cls, save_path_qagent):
        with open(save_path_qagent, 'rb') as f:
             lq_copy = pickle.load(f)
        cls(lq_copy.path_tpx,lq_copy.path_daw)
        return lq_copy
        
        
    def show_chart(self,chart_='None'):
        if chart_=='None':
            pl = PlotTrade(self.chart_)
        else:
            pl = PlotTrade(chart_)
        pl.show()
        
        
    def learn(self,path_tpx,path_daw,state_,chart_,env_type='profit',episode_count=100,train_year=2020,test_year=2021,is_sarsa=False,how_many_years=1,is_imitation=False,teacher_agent=None):
        s_train = state_[train_year-how_many_years<=state_.index.year]
        s_train = s_train[s_train.index.year<=train_year]
        s_test = state_[state_.index.year==test_year]
        price_train = chart_[train_year-how_many_years<=chart_.index.year]
        price_train = price_train[price_train.index.year<=train_year]
        price_test  = chart_[chart_.index.year==test_year]
        self.env_type = env_type
        
        
        if env_type=='profit':
            env_train = Environment(s_train,price_train)
        elif env_type=='sigmoid':
            env_train = SigmoidEnv(s_train,price_train)
        elif env_type=='tanh':
            env_train = TanhEnv(s_train,price_train)
        else:
            print("No such env_type.")
            return 
        
        
        env_test = Environment(s_test,price_test)
        if not is_imitation:
            trainer = ValueFunctionTrainer(buffer_size=1024*8, batch_size=32*8*2)
        else:
            trainer = ImitationTrainer(buffer_size=1024*8, batch_size=32*8*2,teacher_agent=teacher_agent)
            
        trained = trainer.train(env_train,episode_count=episode_count,env_sample=env_test,is_sarsa=is_sarsa)   
        trained.play(env_test)
        self.QL_agent = trained
        self.trainer = trainer
        #         reward_log
        self.train_reward_log, self.test_reward_log = trainer.return_reward_log()
        self.test_trade_log  = trained.return_trade_log()
        
    
    def learn_xqn(self,path_tpx,path_daw,state='proba',env_type='profit',episode_count=100,train_year=2020,test_year=2021,is_sarsa=False):
        state_ ,chart_ = self.make_df_state(path_tpx,path_daw) 
        state_, chart_ = self.make_state(path_tpx,path_daw,state_,chart_,state)
        self.learn(path_tpx=path_tpx,path_daw=path_daw,state_=state_,chart_=chart_,env_type=env_type,episode_count=episode_count,train_year=train_year,test_year=test_year,is_sarsa=is_sarsa)
        
        
        
    def return_reward_log(self):
        return self.train_reward_log, self.test_reward_log
    
    
    def return_test_trade_log(self):
        return self.test_trade_log
    
    
    def make_df_state(self,path_tpx,path_daw):
        state_,chart_ = self.lx.make_state(path_tpx,path_daw)
        df_state = pd.DataFrame(state_)
        df_state['day'] = chart_.index
        df_state.reset_index(inplace=True)
        df_state.set_index('day',inplace=True)
        #         下落の確率をdrop
        df_state.drop('index',axis=1,inplace=True)
        return df_state, chart_
    
    
    def concat_df(self,path_tpx,path_daw):
        df_tpx = DataFramePreProcessing(path_tpx).load_df()
        df_daw = DataFramePreProcessing(path_daw,is_daw=True).load_df()
        daw_p = df_daw.pct_change()
        tpx_p = df_tpx.pct_change()
        tpx_p = tpx_p.rename(columns={'close':'pclose'})
        df_daw = df_daw.rename(columns={'dopen':'daw_close'})
        df_con = pd.concat([df_daw['daw_close'],df_tpx,daw_p['dclose']],axis = 1,join='inner').astype(float)
        df_con['pclose'] = df_con['close'].pct_change()
        df_con = df_con.drop(df_con[ df_con['volume']==0].index)
        return df_con

    
#* 属性ごとに, 標準化     
    def make_standard(self,state_,axis='tate'):
#         axis = 1 が横
        if axis=='yoko':
            lambda_ = lambda x:(x-x.mean())/x.std()    
            state_yoko = state_.apply(lambda_,axis=1)
            return state_yoko
        elif axis=='tate':
            lambda_ = lambda x:(x-x.mean())/x.std()    
            state_tate = state_.apply(lambda_,axis=0)
            return state_tate
        else:
            print("No such axis")
            return None
    
    
    def add_state(self,state,state_,df_con):
        
#         dopen == dclose 
        if state=='proba':
            state_ = self.make_standard(state_,axis='tate')
            # state_ = self.make_standard(state_,axis='yoko')
            return state_ 
        
        
        elif state=='change_rate':
#            pclose, dcloseという名前付いてる
            df_pct = df_con.loc[:,['dclose','pclose']].iloc[1:]
            state_ = pd.concat([df_pct,state_],axis = 1,join='inner').astype(float)
            state_ = state_.dropna()
            state_ = self.make_standard(state_,axis='tate')
            # state_ = self.make_standard(state_,axis='yoko')
            return state_
        
        
        elif state=='moving_average':
            
            state_['ma_short'] = df_con['close'].rolling(self.ma_short).mean()
            # state_['ma_long']  = df_con['close'].rolling(self.ma_long).mean()
            state_ = state_.dropna()
            state_ = self.make_standard(state_,axis='tate')
            # state_ = self.make_standard(state_,axis='yoko')
            return state_
        
        
        elif state=='std':
            
            state_['std_short'] = df_con['close'].rolling(self.ma_short).std()
            state_['std_long']  = df_con['close'].rolling(self.ma_long).std()
            state_ = state_.dropna()
            state_ = self.make_standard(state_,axis='tate')
            # state_ = self.make_standard(state_,axis='yoko')
            return state_
        
        
        elif state=='corr':
            df_tmp1 = df_con[['close','daw_close']].rolling(self.ma_short).corr()
            corr_short = df_tmp1.drop(df_tmp1.index[0:-1:2])['close']
            state_['corr_short']=corr_short.reset_index().set_index('day')['close']
            df_tmp2 = df_con[['close','daw_close']].rolling(self.ma_long).corr()
#             このテクは天才的
            corr_long = df_tmp2.drop(df_tmp2.index[0:-1:2])['close']
            state_['corr_long']=corr_long.reset_index().set_index('day')['close']
            state_ = state_.dropna()
            state_ = self.make_standard(state_,axis='tate')
            return state_
        
        
        elif state=='high_low':
            high_low = (df_con['high']-df_con['low'])/df_con['close']
            state_['high_low'] = high_low
            state_ = state_.dropna()
            state_ = self.make_standard(state_,axis='tate')
            return state_
        
        
        elif state=='close_open':
            close_open = (df_con['close']-df_con['open'])/df_con['close']
            state_['close_open'] = close_open
            state_ = state_.dropna()
            state_ = self.make_standard(state_,axis='tate')
            return state_
        
        
        elif state=='vec':
            ma_short = df_con['close'].rolling(self.ma_short).mean() 
            ma_long = df_con['close'].rolling(self.ma_long).mean() 
            vec_s = ma_short.diff(1)
            vec_l = ma_long.diff(1)
            state_['vec_s'] = vec_s
            state_['vec_l'] = vec_l
            state_ = state_.dropna()
            state_ = self.make_standard(state_,axis='tate')
            return state_
        
        
        elif state=='ema_vec':
            ema_short = df_con['close'].ewm(span=self.ma_short, adjust=False).mean()
            ema_long = df_con['close'].ewm(span=self.ma_long, adjust=False).mean()
            emavec_s = ema_short.diff(1)
            emavec_l = ema_long.diff(1)
            state_['emavec_s'] = emavec_s
            state_['emavec_l'] = emavec_l
            state_ = state_.dropna()
            state_ = self.make_standard(state_,axis='tate')
            return state_
        
        
        elif state=='macd':
            ema_short = df_con['close'].ewm(span=self.ma_short, adjust=False).mean()
            ema_long = df_con['close'].ewm(span=self.ma_long, adjust=False).mean()
            macd = ema_short - ema_long
            macd_signal_short = macd.ewm(span=self.ma_short, adjust=False).mean()
            macd_signal_long = macd.ewm(span=self.ma_long, adjust=False).mean()
            state_['macd'] = macd
            state_['macd_signal_short'] = macd_signal_short
            state_['macd_signal_long'] = macd_signal_long
            state_ = state_.dropna()
            state_ = self.make_standard(state_,axis='tate')
            return state_

            
        elif state=='skew':
            state_['skew_short'] = df_con['close'].rolling(self.ma_short).skew()
            state_['skew_long'] = df_con['close'].rolling(self.ma_long).skew()
            state_ = state_.dropna()
            state_ = self.make_standard(state_,axis='tate')
            return state_
        
        
        elif state=='kurt':
            state_['kurt_short'] = df_con['close'].rolling(self.ma_short).kurt()
            state_['kurt_long'] = df_con['close'].rolling(self.ma_long).kurt()
            state_ = state_.dropna()
            state_ = self.make_standard(state_,axis='tate')
            return state_
        
        
        elif state=='RSI':
            df_up,df_down = df_con['pclose'].copy(),df_con['pclose'].copy()
            df_up[df_up<0]=0
            df_down[df_down>0]=0
            df_down *= -1
            sims_up = df_up.rolling(self.ma_short).mean()
            sims_down = df_down.rolling(self.ma_short).mean()
            siml_up = df_up.rolling(self.ma_long).mean()
            siml_down = df_down.rolling(self.ma_long).mean()
            RSI_short = sims_up / (sims_up + sims_down) * 100
            RSI_long = siml_up / (siml_up + siml_down) * 100
            state_['RSI_short'] = RSI_short
            state_['RSI_long'] = RSI_long
            state_ = state_.dropna()
            state_ = self.make_standard(state_,axis='tate')
            return state_
            
       
        # 確率を落とす
        elif state=='drop_proba':
            return state_.drop(1,axis=1)


        elif state=="open":
            # 始値は観測できる
            pass
        
        
        else:
            print("No such state type.")
            return None
    
    
    def make_state(self,path_tpx,path_daw,state_,chart_,states=['proba'], ma_short=5,ma_long=25):
#         if type(state)==list:
        self.ma_long = ma_long
        self.ma_short = ma_short
        self.state = states
        df_con = self.concat_df(path_tpx,path_daw)
        
        
        for state in states:
            state_ = self.add_state(state,state_,df_con)
        
        
        return state_.drop(0,axis=1), chart_.loc[state_.index]

        
    def predict_tomorrow(self,path_tpx,path_daw):
        check_state, check_chart = self.lx.make_check_data(path_tpx,path_daw)
        state = self.state
        state_, chart_ = self.make_state(path_tpx,path_daw,check_state,check_chart,state=state)
        s_check = state_.iloc[-50:]
        price_check = chart_.iloc[-50:]
        print("today :", s_check.index[-1])
        
        env_check = Environment(s_check,price_check)
        self.QL_agent.play(env_check)

        
    def return_profit_rate(self, path_tpx,path_daw,wallet=2500):
        check_state, check_chart = self.lx.make_check_data(path_tpx,path_daw)
#         OK
        state = self.state
        state_, chart_ = self.make_state(path_tpx,path_daw,check_state,check_chart,states=state)
        s_check = state_
        price_check = chart_
        env_check = Environment(s_check,price_check)
        pr_log =  self.QL_agent.return_profit_rate(env_check,wallet)
        return pr_log

