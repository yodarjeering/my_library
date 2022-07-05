from re import X
import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.manifold import trustworthiness
random.seed(777)
import xgboost as xgb
from sklearn.metrics import classification_report,roc_auc_score
from sklearn.metrics import r2_score
import copy
import optuna
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import preprocessing
from collections import namedtuple
from my_library.funcs import *
ValueTable = namedtuple("ValueTable",["strategy","alpha","total_profit","trade_log","stock_wave"])
Fstrategy = namedtuple("Fstrategy",["strategy","alpha","spectrum"])


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
    

    def __init__(self, df_con, test_rate=0.9, is_bit_search=False,is_category=True,ma_short=5,ma_long=25):
        self.df_con = df_con
        self.test_rate = test_rate
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
        # この書き方は環境によってはエラー
        # x.index = self.df_con.index
        df_con = self.df_con.copy()
        df_ma = self.add_ma()
        end_point = -1
        if is_check:
            end_point = len(self.df_con)
        else:
            end_point = len(self.df_con)-1
        
        # ダウ変化率
        dawp_5 = df_con['dclose'].iloc[:-5]
        dawp_5.index = df_con.index[5:]
        x['dawp_5'] = dawp_5
        dawp_0 = df_con['dclose']
        x['dawp_0'] = dawp_0
        
        # 日経変化率
        nikkeip_5 = df_con['pclose'].iloc[:-5]
        nikkeip_5.index = df_con.index[5:]
        x['nikkeip_5'] = nikkeip_5
        nikkeip_0 = df_con['pclose']
        x['nikkeip_0'] = nikkeip_0
        
        # high - low 変化率
        high_low = (df_con['high']-df_con['low'])/df_con['close']
        x['diff_rate'] = high_low
        
        # close - open 変化率
        close_open = (df_con['close']-df_con['open'])/df_con['close']
        x['close_open'] = close_open
        
        # 売買量変化率
        nikkei_volumep = df_con['volume'].pct_change()
        x['nikkei_volumep'] = nikkei_volumep
        
        # 短期標準偏差ベクトル
        std_s_5 = df_ma['std_short'].iloc[:-5]
        std_s_5.index = df_ma.index[5:]
        x['std_s_5'] = std_s_5
        std_s_0 = df_ma['std_short']
        x['std_s_0'] = std_s_0
        
        # 長期標準偏差ベクトル
        std_l_5 = df_ma['std_long'].iloc[:-5]
        std_l_5.index = df_ma.index[5:]
        x['std_l_5'] = std_l_5
        std_l_0 = df_ma['std_long']
        x['std_l_0'] = std_l_0
        
        # 短期移動平均ベクトル
        vec_s_5 = (df_ma['ma_short'].diff(5)/5)
        x['vec_s_5'] = vec_s_5
        vec_s_1 = (df_ma['ma_short'].diff(1)/1)
        x['vec_s_1'] = vec_s_1
        
        # 長期移動平均ベクトル
        vec_l_5 = (df_ma['ma_long'].diff(5)/5)
        x['vec_l_5'] = vec_l_5
        vec_l_1 = (df_ma['ma_long'].diff(1)/1)
        x['vec_l_1'] = vec_l_1
        
#         移動平均乖離率
        x['d_MASL'] = df_ma['ma_short']/df_ma['ma_long']

#          ema短期のベクトル
        emavec_s_5 = (df_ma['ema_short'].diff(5)/5)
        x['emavec_s_5'] = emavec_s_5
        emavec_s_1 = (df_ma['ema_short'].diff(1)/1)
        emavec_s_1.index = df_ma.index
        x['emavec_s_1'] = emavec_s_1
    
        # ema長期ベクトル
        emavec_l_5 = (df_ma['ema_long'].diff(5)/5)
        x['emavec_l_5'] = emavec_l_5
        emavec_l_1 = (df_ma['ema_long'].diff(1)/1)
        x['emavec_l_1'] = emavec_l_1

        #         EMA移動平均乖離率
        x['d_EMASL'] = df_ma['ema_short']/df_ma['ema_long']
        
        # macd
        macd = df_ma['macd']
        x['macd'] = macd
        macd_signal_short = df_ma['macd_signal_short']
        x['macd_signal_short'] = macd_signal_short
        macd_signal_long = df_ma['macd_signal_long']
        x['macd_signal_long'] = macd_signal_long
            
        # 短期相関係数
        df_tmp1 = df_con[['close','daw_close']].rolling(self.ma_short).corr()
        corr_short = df_tmp1.drop(df_tmp1.index[0:-1:2])['close']
        corr_short = corr_short.reset_index().set_index('day')['close']
        x['corr_short'] = corr_short
        
        # 長期相関係数
        df_tmp2 = df_con[['close','daw_close']].rolling(self.ma_long).corr()
        corr_long = df_tmp2.drop(df_tmp2.index[0:-1:2])['close']
        corr_long = corr_long.reset_index().set_index('day')['close']
        x['corr_long'] = corr_long
        
        # 歪度
        skew_short = df_con['close'].rolling(self.ma_short).skew()
        x['skew_short'] = skew_short
        skew_long = df_con['close'].rolling(self.ma_long).skew()
        x['skew_long'] = skew_long
        
        # 尖度
        kurt_short = df_con['close'].rolling(self.ma_short).kurt()
        x['kurt_short'] = kurt_short
        kurt_long = df_con['close'].rolling(self.ma_long).kurt()
        x['kurt_long'] = kurt_long
        
        # RSI 相対力指数
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

#        Open Close 乖離率
        x['d_OC'] = open_/close_

#       High low 乖離率
        x['d_HL'] = high_/low_
        df_atr = pd.DataFrame(index=high_.index)
        df_atr['high_low'] = high_ - low_
        df_atr['high_close'] = high_ - close_
        df_atr['close_low_abs'] =  (close_ - low_).abs()
        tr = pd.DataFrame(index=open_.index)
        tr['TR'] = df_atr.max(axis=1)

        # ATR
        x['ATR_short'] = tr['TR'].rolling(self.ma_short).mean()
        x['ATR_long'] =  tr['TR'].rolling(self.ma_long).mean()
        x['d_ATR'] = x['ATR_short']/x['ATR_long']
        x['ATR_vecs5'] = (x['ATR_short'].diff(5)/1)
        x['ATR_vecs1'] = (x['ATR_short'].diff(1)/1)
        x['ATR_vecl5'] = (x['ATR_long'].diff(5)/1)
        x['ATR_vecl1'] = (x['ATR_long'].diff(1)/1)
        
        today_close = df_con['close']
        yesterday_close = df_con['close'].iloc[:-1]
        yesterday_close.index = df_con.index[1:]
#        騰落率
#       一度も使用されていなかったため, 削除
        # x['RAF'] =  (today_close/yesterday_close -1)
        x = x.iloc[self.ma_long:end_point]
        x_check = x
#        この '4' は　std_l5 など, インデックスをずらす特徴量が, nanになってしまう分の日数を除くためのもの
        # yについても同様
        x_train = x.iloc[self.ma_short-1:int(len(x)*self.test_rate)]
        x_test  = x.iloc[int(len(x)*self.test_rate):]


        if not is_check:
            
            y_train,y_test = self.make_y_data(x,self.df_con,end_point)
            return x_train, y_train, x_test, y_test
        
        
        else:
            x_check = x_check.iloc[self.ma_short-1:]
            chart_ = self.df_con.loc[x_check.index]
            
            return x_check,chart_


    def make_y_data(self,x,df_con,end_point):
        y = []
        for i in range(self.ma_long,end_point):
            tommorow_close = df_con['close'].iloc[i+1]
            today_close    = df_con['close'].iloc[i]
            if tommorow_close>today_close:
                y.append(1)
            else:
                y.append(0)
            
        y_train = y[self.ma_short-1:int(len(x)*self.test_rate)]
        y_test  = y[int(len(x)*self.test_rate):]
        return y_train,y_test

class MakeTrainData3(MakeTrainData):


    def __init__(self,df_con,test_rate=0.8,alpha=0.5,beta=-0.4):
        super(MakeTrainData3,self).__init__(df_con,test_rate=test_rate)
        self.alpha = alpha
        self.beta = beta


    def make_y_data(self,x,df_con,end_point):
        y = []
        for i in range(self.ma_long,end_point):
            tommorow_close = df_con['close'].iloc[i+1]
            today_close    = df_con['close'].iloc[i]
            change_rate = ((tommorow_close-today_close) / today_close) * 100
            
            # UP : 2
            if change_rate > self.alpha:
                y.append(2)
            # DOWN : 0
            elif change_rate < self.beta:
                y.append(0)
            # STAY : 1
            else:
                y.append(1)
            
        y_train = y[self.ma_short-1:int(len(x)*self.test_rate)]
        y_test  = y[int(len(x)*self.test_rate):]
        return y_train,y_test

class Simulation():


    def __init__(self):
        self.model = None
        self.accuracy_df = None
        self.trade_log = None
        self.pr_log = None
        self.MK = MakeTrainData


    def simulate_routine(self, path_tpx, path_daw,start_year=2021,end_year=2021,start_month=1,end_month=12,df_="None",is_validate=False):
        x_check, y_check = self.make_check_data(path_tpx,path_daw)
        y_ = pd.DataFrame(y_check)
        y_.index = x_check.index
        df_con = self.return_df_con(path_tpx,path_daw)
        df_con['ma_short'] = df_con['close'].rolling(self.ma_short).mean()
        df_con['ma_long']  = df_con['close'].rolling(self.ma_long).mean()
        df_con = df_con.iloc[self.ma_long:]
       
        # 任意の期間の df を入力しても対応できる
        if type(df_)==pd.DataFrame or type(df_)==pd.Series:
            start_ = df_.index[0]
            end_ = df_.index[-1]
            x_check = x_check.loc[start_:end_]
            y_ = y_.loc[start_:end_]
            df_con = df_con.loc[start_:end_]
        else:
            x_check = self.return_split_df(x_check,start_year=start_year,end_year=end_year,start_month=start_month,end_month=end_month)
            y_ = self.return_split_df(y_,start_year=start_year,end_year=end_year,start_month=start_month,end_month=end_month)
            df_con = self.return_split_df(df_con,start_year=start_year,end_year=end_year,start_month=start_month,end_month=end_month)

        self.df_con = df_con
        y_check = y_.values.reshape(-1).tolist()
        if not is_validate:
            pl = PlotTrade(df_con['close'],label='close')
            pl.add_plot(df_con['ma_short'],label='ma_short')
            pl.add_plot(df_con['ma_long'],label='ma_long')
        else:
            pl=None
        self.pr_log = pd.DataFrame(index=x_check.index[:-1])
        self.pr_log['reward'] = [0.0] * len(self.pr_log)
        self.pr_log['eval_reward'] = self.pr_log['reward'].tolist()

        return x_check,y_check,y_,df_con,pl


    def set_for_online(self,x_check,y_):
        x_tmp = x_check
        y_tmp = y_
        current_date = x_tmp.index[0]
        acc_df = pd.DataFrame(index=x_tmp.index)
        acc_df['pred'] = [-1] * len(acc_df)
        return x_tmp, y_tmp, current_date, acc_df


        
    def learn_online(self,x_tmp,y_tmp,x_check,current_date,tmp_date):
        x_ = x_tmp[current_date<=x_tmp.index]
        x_ = x_[x_.index<tmp_date]
        y_ = y_tmp[current_date<=y_tmp.index]
        y_ = y_[y_.index<tmp_date]
        self.xgb_model = self.xgb_model.fit(x_,y_)
        predict_proba = self.xgb_model.predict_proba(x_check.astype(float))
        current_date = tmp_date

        return predict_proba, current_date


    def buy(self,df_con,x_check,i):
#   観測した始値が, 予測に反して上がっていた時, 買わない
        index_buy = df_con['close'].loc[x_check.index[i+1]]
        start_time = x_check.index[i+1]
        is_bought = True

        return index_buy, start_time, is_bought


    def sell(self,df_con,x_check,prf,index_buy,prf_list,trade_count,pl,start_time,i,is_validate):
        index_sell = df_con['close'].loc[x_check.index[i+1]]
        end_time = x_check.index[i+1]
        prf += index_sell - index_buy
        prf_list.append(index_sell - index_buy)
        trade_count += 1
        is_bought = False
        if not is_validate:
            pl.add_span(start_time,end_time)
        else:
            pass

        return prf, trade_count, is_bought


    def hold(self,df_con,index_buy,total_eval_price,i):
        eval_price = df_con['close'].iloc[i] - index_buy
        total_eval_price += eval_price
        self.pr_log['eval_reward'].loc[df_con.index[i]] = total_eval_price
        return total_eval_price


# SELL の直後に BUY となる時, シミュレートできてない
    def return_grad(self, df, index, gamma=0, delta=0):
        grad_ma_short = df['ma_short'].iloc[index+1] - df['ma_short'].iloc[index]
        grad_ma_long  = df['ma_long'].iloc[index+1] - df['ma_long'].iloc[index]
        strategy = ''
        
        if grad_ma_long >= gamma:
            strategy = 'normal'
        elif grad_ma_long < delta:
            strategy = 'reverse'
        else:
            print("No such threshold")
        return strategy

    
    def make_df_con(self,path_tpx,path_daw):
        df_tpx = DataFramePreProcessing(path_tpx).load_df()
        df_daw = DataFramePreProcessing(path_daw,is_daw=True).load_df()
        daw_p = df_daw.pct_change()
        tpx_p = df_tpx.pct_change()
        tpx_p = tpx_p.rename(columns={'close':'pclose'})
        df_daw = df_daw.rename(columns={'dopen':'daw_close'})
        df_con = pd.concat([df_daw['daw_close'],df_tpx,daw_p['dclose'],tpx_p['pclose']],axis = 1,join='inner').astype(float)
        df_con = df_con.drop(df_con[ df_con['volume']==0].index)
        return df_con

    
    def make_check_data(self,path_tpx,path_daw):
        df_con = self.make_df_con(path_tpx,path_daw)
        mk = self.MK(df_con,test_rate=1.0)
        x_check, y_check, _, _ = mk.make_data()
        self.ma_short = mk.ma_short
        self.ma_long = mk.ma_long
        return x_check, y_check
    
    
    def return_df_con(self,path_tpx,path_daw):
        df_con = self.make_df_con(path_tpx,path_daw)
        return df_con


    def calc_acc(self, acc_df, y_check):
        df = pd.DataFrame(columns = ['score','Up precision','Down precision','Up recall','Down recall','up_num','down_num'])
        acc_dict = {'TU':0,'FU':0,'TD':0,'FD':0}
    
        for i in range(len(acc_df)):
            
            label = acc_df['pred'].iloc[i]
            if label==-1:continue

            if y_check[i]==label:
                if label==0:
                    acc_dict['TD'] += 1
                else:#label = 1 : UP
                    acc_dict['TU'] += 1
            else:
                if label==0:
                    acc_dict['FD'] += 1
                else:
                    acc_dict['FU'] += 1

        df = self.calc_accuracy(acc_dict,df)
        return df


    def calc_accuracy(self,acc_dict,df):
        denom = 0
        for idx, key in enumerate(acc_dict):
            denom += acc_dict[key]
        
        try:
            TU = acc_dict['TU']
            FU = acc_dict['FU']
            TD = acc_dict['TD']
            FD = acc_dict['FD']
            score = (TU + TD)/(denom)
            prec_u = TU/(TU + FU)
            prec_d = TD/(TD + FD)
            recall_u = TU/(TU + FD)
            recall_d = TD/(TD + FU)
            up_num = TU+FD
            down_num = TD+FU
            col_list = [score,prec_u,prec_d,recall_u,recall_d,up_num,down_num]
            df.loc[0] = col_list
            return df
        except:
            print("division by zero")
            return None


 
# ここ間違ってる
    def return_split_df(self,df,start_year=2021,end_year=2021,start_month=1,end_month=12):
        df = df[df.index.year>=start_year]
        if start_year <= end_year:
            df = df[df.index.year<=end_year]
        if len(set(df.index.year))==1:
            df = df[df.index.month>=start_month]
            df = df[df.index.month<=end_month]
        else:
            df_tmp = df[df.index.year==start_year]
            last_year_index = df_tmp[df_tmp.index.month==start_month].index[0]
#             new_year_index = df[df.index.month==end_year].index[-1]
            df = df.loc[last_year_index:]
        return df


    def return_trade_log(self,prf,trade_count,prf_array,cant_buy):
        log_dict = {
            'total_profit':prf,
            'trade_count':trade_count,
            'max_profit':prf_array.max(),
            'min_profit':prf_array.min(),
            'mean_profit':prf_array.mean(),
            'cant_buy_count':cant_buy
            }
        df = pd.DataFrame(log_dict,index=[1])
        return df


    
    def get_accuracy(self):
        return self.accuracy_df


    def get_trade_log(self):
        return self.trade_log



    def simulate(self):
        pass


# simulate 済みを仮定
    def return_profit_rate(self,wallet=2500):
        self.pr_log['reward'] = self.pr_log['reward'].map(lambda x: x/wallet)
        self.pr_log['eval_reward'] = self.pr_log['eval_reward'].map(lambda x: x/wallet)
        return self.pr_log


class XGBSimulation(Simulation):
    
    
    def __init__(self, lx, alpha=0.70):
        super(XGBSimulation,self).__init__()
        self.lx = lx
        self.xgb_model = lx.model
        self.alpha = alpha
        self.acc_df = None
        self.y_check = None
        self.ma_long = 0
        self.ma_short = 0
        self.is_bought = False
        # 20日以上 ホールドしたら, 自動的に売り
        self.calc_func = self.calc_acc
        self.MK = MakeTrainData
        

    def simulate(self, path_tpx, path_daw, is_validate=False,strategy='normal',is_online=False,start_year=2021,end_year=2021,start_month=1,end_month=12,
                is_variable_strategy=False,is_observed=False,df_="None"):
        
        x_check,y_check,y_,df_con,pl = self.simulate_routine(path_tpx, path_daw,start_year,end_year,start_month,end_month,df_,is_validate)
        x_tmp,y_tmp,current_date,acc_df = self.set_for_online(x_check,y_)
        length = len(x_check)
        prf_list = []
        predict_proba = self.xgb_model.predict_proba(x_check.astype(float))
        is_bought = False
        index_buy = 0
        index_sell = 0
        prf = 0
        hold_day = 0
        trade_count = 0
        trigger_count = 0
        total_eval_price = 0
        cant_buy = 0 # is_observed=True としたことで買えなくなった取引の回数をカウント
        is_trigger=False
        
        for i in range(length-1):
            
            
            row = predict_proba[i]
            label = np.argmax(row)
            prob = row[label]
            total_eval_price = prf
            self.pr_log['reward'].loc[df_con.index[i]] = prf 
            self.pr_log['eval_reward'].loc[df_con.index[i]] = total_eval_price
#             label==0 -> down
#             label==1 -> up
#             オンライン学習
            tmp_date = x_tmp.index[i]   
            if is_online and current_date.month!=tmp_date.month:
                predict_proba, current_date = self.learn_online(x_tmp,y_tmp,x_check,current_date,tmp_date)
# ここのprob は2クラスうち, 出力の大きいほうのクラスの可能性が代入されている
            if prob > self.alpha:
                if label == 0:
                    acc_df.iloc[i] = 0
                else: # label == 1:  
                    acc_df.iloc[i] = 1
            
            if is_variable_strategy:
                strategy = self.return_grad(df_con, index=i-1,gamma=0, delta=0)
            

            if strategy=='reverse':
                is_buy  = (label==0 and prob>self.alpha)
                is_sell = (label==1 and prob>self.alpha)
                is_cant_buy = (is_observed and (df_con['open'].loc[x_check.index[i+1]] > df_con['close'].loc[x_check.index[i]]))
            elif strategy=='normal':
                is_buy  = (label==1 and prob>self.alpha)
                is_sell = (label==0 and prob>self.alpha)
                is_cant_buy = (is_observed and (df_con['open'].loc[x_check.index[i+1]] < df_con['close'].loc[x_check.index[i]]))
            else:
                print("No such strategy.")
                return 

            
            if not is_bought:
                if is_cant_buy:
                    cant_buy += 1
                    continue
                elif is_buy:
                    index_buy, start_time, is_bought = self.buy(df_con,x_check,i)

            else:
                hold_day += 1
                if hold_day>=20:
                    trigger_count+=1
                    is_trigger=True

                if is_sell or is_trigger:
                    prf, trade_count, is_bought = self.sell(df_con,x_check,prf,index_buy,prf_list,trade_count,pl,start_time,i,is_validate)
                    hold_day = 0
                    is_trigger = False
                else:
                    total_eval_price = self.hold(df_con,index_buy,total_eval_price,i)
                    
            
            self.is_bought = is_bought
                  
        
        if is_bought:
            index_sell = df_con['close'].loc[x_check.index[-1]] 
            prf += index_sell - index_buy
            prf_list.append(index_sell - index_buy)
            end_time = x_check.index[-1]
            trade_count+=1
            if not is_validate:
                pl.add_span(start_time,end_time)

        
        self.pr_log['reward'].loc[df_con.index[-1]] = prf 
        self.pr_log['eval_reward'].loc[df_con.index[-1]] = total_eval_price
        prf_array = np.array(prf_list)
        self.acc_df = acc_df
        self.y_check = y_check
            
        
        try:
            df = self.calc_acc(acc_df, y_check)
            self.accuracy_df = df
            log = self.return_trade_log(prf,trade_count,prf_array,cant_buy)
            self.trade_log = log

            if not is_validate:
                print(log)
                print("")
                print(df)
                print("")
                print("trigger_count :",trigger_count)
                pl.show()
        except:
            print("no trade")
    
    def show_result(self, path_tpx,path_daw,strategy='normal'):
        x_check, y_check = self.make_check_data(path_tpx,path_daw)  
        self.simulate(x_check,y_check,strategy)

class XGBSimulation2(XGBSimulation):


    def __init__(self,lx,alpha=0.33):
        super(XGBSimulation2,self).__init__(lx,alpha)
        self.MK = MakeTrainData3

    def calc_acc(self, acc_df, y_check):
        df = pd.DataFrame(columns = ['score','Up precision','Stay precision','Down precision','Up recall','Stay recall','Down recall','up_num','stay_num','down_num'])
        acc_dict = {'TU':0,'FUs':0,'FUd':0,'TS':0,'FSu':0,'FSd':0,'TD':0,'FDu':0,'FDs':0}
    
        for i in range(len(acc_df)):
            
            label = acc_df['pred'].iloc[i]
            if label == -1: continue

            if y_check[i]==label:
                if label==0:
                    acc_dict['TD'] += 1
                elif label==1:
                    acc_dict['TS'] += 1
                else:#label = 2 : UP
                    acc_dict['TU'] += 1
            else:
                if label==0:
                    if y_check[i]==2:
                        # FDu
                        acc_dict['FDu'] += 1
                    else: 
                        # FDs
                        acc_dict['FDs'] += 1
                elif label==1:
                    if y_check[i]==2:
                        # FSu
                        acc_dict['FSu'] += 1
                    else:
                        # FSd
                        acc_dict['FSd'] += 1
                else:
                    if y_check[i]==0:
                        # FUd
                        acc_dict['FUd'] += 1
                    else:
                        # FUs
                        acc_dict['FUs'] += 1

        df = self.calc_accuracy(acc_dict,df)
        return df


    def calc_accuracy(self,acc_dict,df):
        denom = 0
        for idx, key in enumerate(acc_dict):
            denom += acc_dict[key]
        
        try:
            TU = acc_dict['TU']
            FUs = acc_dict['FUs']
            FUd = acc_dict['FUd']
            TS = acc_dict['TS']
            FSu = acc_dict['FSu']
            FSd = acc_dict['FSd']
            TD = acc_dict['TD']
            FDu = acc_dict['FDu']
            FDs = acc_dict['FDs']

            score = (TU + TD + TS)/(denom)
            prec_u = TU/(TU + FUs + FUd)
            prec_s = TS/(TS + FSu + FSd)
            prec_d = TD/(TD + FDu + FDs)
            recall_u = TU/(TU + FSu + FDu)
            recall_s = TS/(TS + FUs + FDs)
            recall_d = TD/(TD + FUd + FSd)
            # recall の分母
            up_num = TU+FSu+FDu
            stay_num = TS+FUs+FDs
            down_num = TD+FUd+FSd
            col_list = [score,prec_u,prec_s,prec_d,recall_u,recall_s,recall_d,up_num,stay_num,down_num]
            df.loc[0] = col_list
            return df
        except:
            print("division by zero")
            return None



    def simulate(self, path_tpx, path_daw, is_validate=False,strategy='normal',is_online=False,start_year=2021,end_year=2021,start_month=1,end_month=12,
            is_variable_strategy=False,is_observed=False,df_="None"):
    
        x_check,y_check,y_,df_con,pl = self.simulate_routine(path_tpx, path_daw,start_year,end_year,start_month,end_month,df_,is_validate)
        x_tmp,y_tmp,current_date,acc_df = self.set_for_online(x_check,y_)
        length = len(x_check)
        prf_list = []
        predict_proba = self.xgb_model.predict_proba(x_check.astype(float))
        is_bought = False
        index_buy = 0
        index_sell = 0
        prf = 0
        hold_day = 0
        trigger_count = 0
        is_trigger = False
        trade_count = 0
        total_eval_price = 0
        cant_buy = 0 # is_observed=True としたことで買えなくなった取引の回数をカウント

        
        for i in range(length-1):
            
            
            row = predict_proba[i]
            label = np.argmax(row)
            prob = row[label]
            total_eval_price = prf
            self.pr_log['reward'].loc[df_con.index[i]] = prf 
            self.pr_log['eval_reward'].loc[df_con.index[i]] = total_eval_price
            tmp_date = x_tmp.index[i]   

            if is_online and current_date.month!=tmp_date.month:
                predict_proba, current_date = self.learn_online(x_tmp,y_tmp,x_check,current_date,tmp_date)

# ここのprob は2クラスうち, 出力の大きいほうのクラスの可能性が代入されている
            if prob > self.alpha:
                if label == 0:
                    acc_df.iloc[i] = 0
                elif label == 1:  
                    acc_df.iloc[i] = 1
                else: #  label==2
                    acc_df.iloc[i] = 2
            
            if is_variable_strategy:
                strategy = self.return_grad(df_con, index=i-1,gamma=0, delta=0)
            

            if strategy=='reverse':
                is_buy  = (label==0 and prob>self.alpha)
                is_sell = (label==2 and prob>self.alpha)
                is_cant_buy = (is_observed and (df_con['open'].loc[x_check.index[i+1]] > df_con['close'].loc[x_check.index[i]]))
            elif strategy=='normal':
                is_buy  = (label==2 and prob>self.alpha)
                is_sell = (label==0 and prob>self.alpha)
                is_cant_buy = (is_observed and (df_con['open'].loc[x_check.index[i+1]] < df_con['close'].loc[x_check.index[i]]))
            else:
                print("No such strategy.")
                return 

            
            if not is_bought:
                if is_cant_buy:
                    cant_buy += 1
                    continue
                elif is_buy:
                    index_buy, start_time, is_bought = self.buy(df_con,x_check,i)

            else:
                hold_day += 1
                if hold_day>=20:
                    trigger_count+=1
                    is_trigger = True

                if is_sell or is_trigger:
                    prf, trade_count, is_bought = self.sell(df_con,x_check,prf,index_buy,prf_list,trade_count,pl,start_time,i,is_validate)
                    hold_day = 0
                    is_trigger = False
                else:
                    total_eval_price = self.hold(df_con,index_buy,total_eval_price,i)
                    
            
            self.is_bought = is_bought
                
        
        if is_bought:
            index_sell = df_con['close'].loc[x_check.index[-1]] 
            prf += index_sell - index_buy
            prf_list.append(index_sell - index_buy)
            end_time = x_check.index[-1]
            trade_count+=1
            if not is_validate:
                pl.add_span(start_time,end_time)

        
        self.pr_log['reward'].loc[df_con.index[-1]] = prf 
        self.pr_log['eval_reward'].loc[df_con.index[-1]] = total_eval_price
        prf_array = np.array(prf_list)
        self.acc_df = acc_df
        self.y_check = y_check
            
        
        # try:
        df = self.calc_acc(acc_df, y_check)
        self.accuracy_df = df
        log = self.return_trade_log(prf,trade_count,prf_array,cant_buy)
        self.trade_log = log

        if not is_validate:
            print(log)
            print("")
            print(df)
            print("")
            print("trigger_count :",trigger_count)
            pl.show()

class TechnicalSimulation(Simulation):
    
    
    def __init__(self,ma_short=5, ma_long=25, hold_day=5, year=2021):
        super(TechnicalSimulation,self).__init__()
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.hold_day = hold_day
        self.year = year
        
    
    def is_buyable(self, short_line, long_line, index_):
#         1=<index<=len-1 仮定
        long_is_upper = long_line.iloc[index_-1]>short_line.iloc[index_-1]
        long_is_lower = long_line.iloc[index_]<=short_line.iloc[index_]
        buyable = long_is_upper and long_is_lower
        return buyable
    
    
    def is_sellable(self, short_line, long_line, index_):
        long_is_lower = long_line.iloc[index_-1]<short_line.iloc[index_-1]
        long_is_upper = long_line.iloc[index_]>=short_line.iloc[index_]
        sellable = long_is_upper and long_is_lower
        return sellable

        
        
    def simulate(self,path_tpx,path_daw,is_validate=False,start_year=2021,end_year=2021,start_month=1,end_month=12):
        _,_,_,df_con,pl = self.simulate_routine(path_tpx, path_daw,start_year,end_year,start_month,end_month)
        prf_list = []
        is_bought = False
        index_buy = 0
        prf = 0
        trade_count = 0
        eval_price = 0
        total_eval_price = 0
        short_line = df_con['ma_short']
        long_line = df_con['ma_long']
        length = len(df_con)
  
        for i in range(1,length):
            
            total_eval_price = prf
            self.pr_log['reward'].loc[df_con.index[i]] = prf 
            self.pr_log['eval_reward'].loc[df_con.index[i]] = total_eval_price
            if not is_bought:
                
                if self.is_buyable(short_line,long_line,i):
                    index_buy = df_con['close'].iloc[i]
                    is_bought = True
                    start_time = df_con.index[i]
                    hold_count_day = 0
                else:
                    continue
            
            
            else:
                
                if self.is_sellable(short_line,long_line,i) or hold_count_day==self.hold_day:
                    index_cell = df_con['close'].iloc[i]
                    end_time = df_con.index[i]
                    prf += index_cell - index_buy
                    prf_list.append(index_cell - index_buy)
                    total_eval_price = prf
                    self.pr_log['reward'].loc[df_con.index[i]] = prf 
                    self.pr_log['eval_reward'].loc[df_con.index[i]] = total_eval_price
                    trade_count+=1
                    is_bought = False
                    hold_count_day = 0
                    pl.add_span(start_time,end_time)
                else:
                    hold_count_day+=1
                    eval_price = df_con['close'].iloc[i] - index_buy
                    total_eval_price += eval_price
                    self.pr_log['eval_reward'].loc[df_con.index[i]] = total_eval_price
                    
        
        if is_bought:
            end_time = df_con['close'].index[-1]
            index_sell = df_con['close'].iloc[-1]
            pl.add_span(start_time,end_time)
            eval_price = index_sell - index_buy
            prf += eval_price
            prf_list.append(prf)
            total_eval_price += eval_price
            self.pr_log['eval_reward'].loc[df_con.index[-1]] = total_eval_price
        
        prf_array = np.array(prf_list)
        log = self.return_trade_log(prf,trade_count,prf_array,0)
        self.trade_log = log

        if not is_validate:        
            print(log)
            print("")
            pl.show()    


class FFTSimulation(XGBSimulation2):


    def __init__(self, lx, Fstrategies, alpha=0.33):
        super(FFTSimulation,self).__init__(lx,alpha)
        self.Fstrategies = Fstrategies
        

    def choose_strategy(self,x_spe):
        cos_sim_list = []
        """
            spectrums[0] : normal_good のスペクトラム
            spectrums[1] : normal_bad のスペクトラム
                         :

            strategy_list[0] : ('normal',alpha=0.3)
            strategy_list[1] : ('stay',alpha=0.3)
                             : 
        """
        for fs in self.Fstrategies:
            cos = cos_sim(fs.spectrum,x_spe)
            cos_sim_list.append(cos)

        max_index = np.argmax(np.array(cos_sim_list))
        strategy = self.Fstrategies[max_index].strategy
        self.alpha = self.Fstrategies[max_index].alpha
        return strategy


    def make_x_dict(self,path_tpx,path_daw):
        df_con = self.make_df_con(path_tpx,path_daw)
        x_dict = {}
        lc = LearnClustering()
        x_, z_ = lc.make_x_data(df_con['close'],stride=1,test_rate=1.0)
        length = len(z_)

        for i in range(length):
            time_ = z_[i].index[-1]
            x_dict[time_] = standarize(x_[i])
        
        return x_dict


    def do_fft(self,wave_vec):
        N = len(wave_vec)            # サンプル数
        dt = 1          # サンプリング間隔
        t = np.arange(0, N*dt, dt) # 時間軸
        freq = np.linspace(0, 1.0/dt, N) # 周波数軸

        f = wave_vec
        F = np.fft.fft(f)

        # 振幅スペクトルを計算
        Amp = np.abs(F)
        return F

    
    def make_spectrum(self,wave_vec):
        F = self.do_fft(wave_vec)
        # spectrum = np.concatenate([F.real,F.imag])
        spectrum = np.abs(F)**2
        return spectrum


    def simulate(self, path_tpx, path_daw, is_validate=False,is_online=False,start_year=2021,end_year=2021,start_month=1,end_month=12,
    is_observed=False):
        
        x_check,y_check,y_,df_con,pl = self.simulate_routine(path_tpx, path_daw,start_year,end_year,start_month,end_month,'None',is_validate)
        self.x_check = x_check
        x_tmp,y_tmp,current_date,acc_df = self.set_for_online(x_check,y_)
        x_dict = self.make_x_dict(path_tpx,path_daw)
        length = len(x_check)
        prf_list = []
        predict_proba = self.xgb_model.predict_proba(x_check.astype(float))
        is_bought = False
        index_buy = 0
        index_sell = 0
        prf = 0
        hold_day = 0
        trigger_count = 0
        is_trigger = False
        trade_count = 0
        total_eval_price = 0
        cant_buy = 0
        buy_count = 0
        sell_count = 0

        
        for i in range(length-1):

            time_ = df_con.index[i]
            x_spe = self.make_spectrum(x_dict[time_])

            if not is_bought:
                strategy = self.choose_strategy(x_spe)

            row = predict_proba[i]
            label = np.argmax(row)
            prob = row[label]
            total_eval_price = prf
            self.pr_log['reward'].loc[df_con.index[i]] = prf 
            self.pr_log['eval_reward'].loc[df_con.index[i]] = total_eval_price
            tmp_date = x_tmp.index[i]   

            if is_online and current_date.month!=tmp_date.month:
                predict_proba, current_date = self.learn_online(x_tmp,y_tmp,x_check,current_date,tmp_date)


            if prob > self.alpha:
                if label == 0:
                    acc_df.iloc[i] = 0
                elif label == 1:  
                    acc_df.iloc[i] = 1
                else: #  label==2
                    acc_df.iloc[i] = 2

            if strategy=='reverse':
                is_buy  = (label==0 and prob>self.alpha)
                is_sell = (label==2 and prob>self.alpha)
                is_cant_buy = (is_observed and (df_con['open'].loc[x_check.index[i+1]] > df_con['close'].loc[x_check.index[i]]))
            elif strategy=='normal':
                is_buy  = (label==2 and prob>self.alpha)
                is_sell = (label==0 and prob>self.alpha)
                is_cant_buy = (is_observed and (df_con['open'].loc[x_check.index[i+1]] < df_con['close'].loc[x_check.index[i]]))
            elif strategy=='stay' :
                is_buy = False
                is_sell =  False
                is_cant_buy = False

            

            
            if not is_bought:
                if is_cant_buy:
                    cant_buy += 1
                    continue
                elif is_buy:
                    index_buy, start_time, is_bought = self.buy(df_con,x_check,i)
                    buy_count += 1

            else:
                hold_day += 1
                if hold_day>=20:
                    trigger_count+=1
                    is_trigger = True

                if is_sell or is_trigger:
                    prf, trade_count, is_bought = self.sell(df_con,x_check,prf,index_buy,prf_list,trade_count,pl,start_time,i,is_validate)
                    hold_day = 0
                    is_trigger = False
                    sell_count += 1
                else:
                    total_eval_price = self.hold(df_con,index_buy,total_eval_price,i)
                    
            
            self.is_bought = is_bought
                
        
        if is_bought:
            index_sell = df_con['close'].loc[x_check.index[-1]] 
            prf += index_sell - index_buy
            prf_list.append(index_sell - index_buy)
            end_time = x_check.index[-1]
            trade_count+=1
            if not is_validate:
                pl.add_span(start_time,end_time)

        
        self.pr_log['reward'].loc[df_con.index[-1]] = prf 
        self.pr_log['eval_reward'].loc[df_con.index[-1]] = total_eval_price
        prf_array = np.array(prf_list)
        self.acc_df = acc_df
        self.y_check = y_check
            
        
        # try:
        df = self.calc_acc(acc_df, y_check)
        self.accuracy_df = df
        log = self.return_trade_log(prf,trade_count,prf_array,cant_buy)
        self.trade_log = log

        if not is_validate:
            print(log)
            print("")
            print(df)
            print("")
            print("trigger_count :",trigger_count)
            # print("buy_count",buy_count)
            # print("sell_count",sell_count)
            pl.show()
            

    
class LearnXGB():
    
    
    def __init__(self, num_class=2):
        self.model = xgb.XGBClassifier()
        self.x_test = None
        self.num_class = num_class
        if num_class==2:
            self.MK = MakeTrainData
        elif num_class==3:
            self.MK = MakeTrainData3
    

    def learn_xgb(self, path_tpx, path_daw, test_rate=0.8, param_dist='None'):
        x_train,y_train,x_test,y_test = self.make_xgb_data(path_tpx,path_daw,test_rate)
        
        if param_dist=='None':
#             Grid search で求めたパラメタ 2021/11/21
            param_dist = { 
            'n_estimators':16,
            'max_depth':4}

        xgb_model = xgb.XGBClassifier(**param_dist)
        hr_pred = xgb_model.fit(x_train.astype(float), np.array(y_train), eval_metric='logloss').predict(x_test.astype(float))
        print("---------------------")
        y_proba_train = xgb_model.predict_proba(x_train)[:,1]
        y_proba = xgb_model.predict_proba(x_test)[:,1]

        if self.num_class==2:
            print('AUC train:',roc_auc_score(y_train,y_proba_train))    
            print('AUC test :',roc_auc_score(y_test,y_proba))

        print(classification_report(np.array(y_test), hr_pred))
        _, ax = plt.subplots(figsize=(12, 10))
        xgb.plot_importance(xgb_model,ax=ax) 
        self.model = xgb_model


    def learn_xgb2(self,x_train,y_train,x_test,y_test,param_dist='None'):
        if param_dist=='None':
#             Grid search で求めたパラメタ 2021/11/21
            param_dist = { 
                'n_estimators':16,
                'max_depth':4
                 }

        xgb_model = xgb.XGBClassifier(**param_dist)
        hr_pred = xgb_model.fit(x_train.astype(float), np.array(y_train), eval_metric='logloss').predict(x_test.astype(float))
        print("---------------------")
        y_proba_train = xgb_model.predict_proba(x_train)[:,1]
        y_proba = xgb_model.predict_proba(x_test)[:,1]

        if self.num_class==2:
            print('AUC train:',roc_auc_score(y_train,y_proba_train))    
            print('AUC test :',roc_auc_score(y_test,y_proba))

        print(classification_report(np.array(y_test), hr_pred))
        _, ax = plt.subplots(figsize=(12, 10))
        xgb.plot_importance(xgb_model,ax=ax) 
        self.model = xgb_model
        

    def make_state(self,path_tpx,path_daw):
        df_con = self.make_df_con(path_tpx,path_daw)
        mk = self.MK(df_con)
        x_check, chart_ = mk.make_data(is_check=True)
        state_ = self.model.predict_proba(x_check.astype(float))
        chart_ = df_con['close'].loc[x_check.index[0]:x_check.index[-1]]
        return state_, chart_
        
        
    def make_xgb_data(self, path_tpx, path_daw, test_rate=0.8):
        df_con = self.make_df_con(path_tpx,path_daw)
        mk = self.MK(df_con,test_rate=test_rate)
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
    
    
    def for_ql_data(self, path_tpx, path_daw):
        df_con = self.make_df_con(path_tpx,path_daw)

        mk = self.MK(df_con)
        x_check, chart_ = mk.make_data(is_check=True)
        state_ = self.model.predict_proba(x_check.astype(float))

        chart_ = mk.df_con['close'].loc[x_check.index[0]:x_check.index[-1]]
        state_ = pd.DataFrame(state_)
        state_['day'] = chart_.index
        
        state_.reset_index(inplace=True)
        state_.set_index('day',inplace=True)
        state_.drop('index',axis=1,inplace=True)
        return state_, chart_
    
    
    def predict_tomorrow(self, path_tpx, path_daw, alpha=0.5, strategy='normal', is_online=False, is_valiable_strategy=False,start_year=2021,start_month=1,end_month=12,is_observed=False,is_validate=False):
        xl = XGBSimulation(self,alpha=alpha)
        xl.simulate(path_tpx,path_daw,is_validate=is_validate,strategy=strategy,is_variable_strategy=is_valiable_strategy,start_year=start_year,start_month=start_month,end_month=end_month,is_observed=is_observed,is_online=is_online)
        self.xl = xl
        df_con = self.make_df_con(path_tpx,path_daw)
        mk = self.MK(df_con)
        x_check, chart_ = mk.make_data(is_check=True)
        tomorrow_predict = self.model.predict_proba(x_check)
        label = self.get_tomorrow_label(tomorrow_predict,strategy, is_valiable_strategy)
        print("is_bought",xl.is_bought)
        print("df_con in predict_tomorrow",df_con.index[-1])
        print("today :",x_check.index[-1])
        print("tomorrow UP possibility", tomorrow_predict[-1,1])
        print("label :",label)


    def get_tomorrow_label(self, tomorrow_predict,strategy, is_valiable_strategy):
        label = "STAY"
        df_con = self.xl.df_con
        if is_valiable_strategy:
            i = len(df_con)-2
            strategy = self.xl.return_grad(df_con, index=i,gamma=0, delta=0)
        
        if strategy == 'normal':
            if tomorrow_predict[-1,1] > self.xl.alpha:
                label = "BUY"
            elif 1-tomorrow_predict[-1,1] > self.xl.alpha:
                label =  "SELL"
            else:
                label = "STAY"
        
        elif strategy == 'reverse':
            if 1-tomorrow_predict[-1,1] > self.xl.alpha:
                label = "BUY"
            elif tomorrow_predict[-1,1] > self.xl.alpha:
                label = "SELL"
            else:
                label = "STAY"

        return label

class LearnClustering(LearnXGB):


    def __init__(self,n_cluster=8,width=20,stride=5,strategy_table=None):
        super(LearnClustering,self).__init__()
        self.model : KMeans = None
        self.n_cluster = n_cluster
        self.width = width
        self.stride = stride
        self.n_label = None
        self.wave_dict = None
        self.strategy_table = strategy_table



    def make_x_data(self,close_,width=20,stride=5,test_rate=0.8):
        length = int(len(close_)*test_rate)
        close_ = close_.iloc[:length]
        # close_ = standarize(close_)
        # close_list = close_.tolist()

        x = []
        z = []
        for i in range(0,length-width,stride):
            x.append(standarize(close_.iloc[i:i+width]).tolist())
            z.append(close_.iloc[i:i+width])
        x = np.array(x)
        return x,z


    def make_wave_dict(self,x,y,width):
        n_label = list(set(y))
        self.n_label = n_label
        wave_dict = {i:np.array([0.0 for j in range(width)]) for i in n_label}
        
        # クラス波形の総和
        for i in range(len(x)):
            wave_dict[y[i]] += x[i]
        
        # 平均クラス波形
        for i in range(len(y)):
            count_class = list(y).count(y[i])
            wave_dict[y[i]] /= count_class
            wave_dict[y[i]] = preprocessing.scale(wave_dict[y[i]])
        return wave_dict


    def learn_clustering(self,path_tpx,path_daw,width=20,stride=5,test_rate=0.8):
        df_con = self.make_df_con(path_tpx,path_daw)
        close_ = df_con['close']
        x,_ = self.make_x_data(close_,width=width,stride=stride,test_rate=test_rate)
        model = KMeans(n_clusters=self.n_cluster)
        model.fit(x)
        self.model = model
        y = model.labels_
        wave_dict = self.make_wave_dict(x,y,width)
        self.wave_dict = wave_dict


    def learn_clustering2(self,close_,width=20,stride=5):
        x,_ = self.make_x_data(close_,width=width,stride=stride,test_rate=1.0)
        model = KMeans(n_clusters=self.n_cluster)
        model.fit(x)
        self.model = model
        y = model.labels_
        wave_dict = self.make_wave_dict(x,y,width)
        self.wave_dict = wave_dict


    def learn_clustering3(self,x,width=20,stride=5):
        model = KMeans(n_clusters=self.n_cluster)
        model.fit(x)
        self.model = model
        y = model.labels_
        wave_dict = self.make_wave_dict(x,y,width)
        self.wave_dict = wave_dict
    

    def show_class_wave(self):
        for i in range(self.n_cluster):
            print("--------------------")
            print("class :",i)
            plt.plot(self.wave_dict[i])
            plt.show()
            plt.clf()


    def predict(self,path_tpx,path_daw,stride=2,test_rate=1.0):
        df_con = self.make_df_con(path_tpx,path_daw)
        close_ = df_con["close"]
        x,z = self.make_x_data(close_,stride=stride,test_rate=test_rate)
        y_pred  = self.model.predict(x)
        return y_pred,z


    def predict2(self,df_con,stride=2,test_rate=1.0):
        close_ = df_con["close"]
        x,z = self.make_x_data(close_,stride=stride,test_rate=test_rate)
        y_pred  = self.model.predict(x)
        return y_pred,z

    
    def return_y_pred(self,path_tpx,path_daw,stride=2,test_rate=1.0):
        df_con = self.make_df_con(path_tpx,path_daw)
        close_ = df_con["close"]
        x,z = self.make_x_data(close_,stride=stride,test_rate=test_rate)
        y_pred  = self.model.predict(x)
        return y_pred

    def encode(self, strategy, alpha, wave_dict):
        pass

class LearnTree(LearnXGB):
    
    
    def __init__(self):
        super(LearnTree,self).__init__()
        self.model : tree.DecisionTreeClassifier() = None
        self.x_test = None
    
    
    def learn_tree(self, path_tpx, path_daw, test_rate=0.8, param_dist='None'):
        x_train,y_train,x_test,y_test = self.make_xgb_data(path_tpx,path_daw,test_rate)
        tree_model = tree.DecisionTreeClassifier(random_state=0)
        hr_pred = tree_model.fit(x_train.astype(float), np.array(y_train)).predict(x_test.astype(float))
        print("---------------------")
        y_proba_train = tree_model.predict_proba(x_train)[:,1]
        y_proba = tree_model.predict_proba(x_test)[:,1]
        print('AUC train:',roc_auc_score(y_train,y_proba_train))    
        print('AUC test :',roc_auc_score(y_test,y_proba))
        print(classification_report(np.array(y_test), hr_pred))
        self.model = tree_model
        
class LearnRandomForest(LearnXGB):
    
    
    def __init__(self):
        super(LearnRandomForest,self).__init__()
        self.model : RandomForestClassifier = None
        self.x_test = None
    
    
    def learn_forest(self, path_tpx, path_daw, test_rate=0.8, param_dist='None'):
        x_train,y_train,x_test,y_test = self.make_xgb_data(path_tpx,path_daw,test_rate)
        tree_model = self.model = RandomForestClassifier(max_depth=2, random_state=0)
        hr_pred = tree_model.fit(x_train.astype(float), np.array(y_train)).predict(x_test.astype(float))
        print("---------------------")
        y_proba_train = tree_model.predict_proba(x_train)[:,1]
        y_proba = tree_model.predict_proba(x_test)[:,1]
        print('AUC train:',roc_auc_score(y_train,y_proba_train))    
        print('AUC test :',roc_auc_score(y_test,y_proba))
        print(classification_report(np.array(y_test), hr_pred))
        self.model = tree_model

class LearnLogisticRegressor(LearnXGB):
    
    
    def __init__(self):
        super(LearnLogisticRegressor,self).__init__()
        self.model : LogisticRegression = None
        self.x_test = None
    
    
    def learn_logistic(self, path_tpx, path_daw, test_rate=0.8, param_dist='None'):
        x_train,y_train,x_test,y_test = self.make_xgb_data(path_tpx,path_daw,test_rate)
        logistic_model = LogisticRegression(max_iter=2000)
        hr_pred = logistic_model.fit(x_train.astype(float), np.array(y_train)).predict(x_test.astype(float))
        print("---------------------")
        y_proba_train = logistic_model.predict_proba(x_train)[:,1]
        y_proba = logistic_model.predict_proba(x_test)[:,1]
        print('AUC train:',roc_auc_score(y_train,y_proba_train))    
        print('AUC test :',roc_auc_score(y_test,y_proba))
        print(classification_report(np.array(y_test), hr_pred))
        self.model = logistic_model

class LearnLinearRegression(LearnXGB):


    def __init__(self):
        super(LearnLinearRegression,self).__init__()
        self.model : LinearRegression = None
        self.x_test = None
        self.x_val = None
        self.y_val = None
        plt.clf()

    # データリーク確認
    # 直すように
    # diff_date : 何日後の予測をするか指定するパラメタ
    def make_regression_data(self,path_tpx,path_daw,test_rate=0.8,diff_date=5):
        df = self.make_df_con(path_tpx,path_daw)
        x_train,_,x_test,_ = self.make_xgb_data(path_tpx,path_daw,test_rate=test_rate)
        x_train, y_train = make_data(x_train,df['close'])
        x_test,y_test = make_data(x_test,df['close'])
        x_train = standarize(x_train)
        x_test = standarize(x_test)
        y_train = standarize(y_train)
        y_test = standarize(y_test)
        x_train = x_train.iloc[:-diff_date]
        y_train = y_train.iloc[diff_date:]
        x_test = x_test.iloc[:-diff_date]
        y_test = y_test.iloc[diff_date:]
        self.x_val = x_test
        self.y_val = y_test
        return x_train,y_train,x_test,y_test



    def learn_linear_regression(self,path_tpx,path_daw,test_rate=0.8):
        x_train,y_train,x_test,y_test = self.make_regression_data(path_tpx,path_daw,test_rate=test_rate)
        lr = LinearRegression()
        lr.fit(x_train,y_train)
        self.model = lr
        y_pred = lr.predict(x_test)
        print("True")
        plt.plot(y_test.iloc[:20])
        plt.show()
        print("Predict")
        plt.plot(y_pred[:20])
        plt.grid()
        plt.show()


    def show_model_summary(self):
        x_add_const = sm.add_constant(self.x_val)
        model_sm = sm.OLS(self.y_val, x_add_const).fit()
        print(model_sm.summary())

class RandomSimulation(Simulation):


    def __init__(self,ma_short=5,ma_long=25,random_num=2):
        super(RandomSimulation,self).__init__()
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.random_num = random_num


    def make_check_data(self,path_tpx,path_daw):
        df = self.make_df_con(path_tpx,path_daw)
        x_check = df.iloc[1:]
        length = len(df)
        y_check = []
        for i in range(1,length):
            if df['pclose'].iloc[i] > 0:
                y_check.append(1)
            else:
                y_check.append(0)
        
        return x_check, y_check

    def random_func(self,random_num):
        # 0 or 1 を返す関数
        return random.randint(0,random_num-1)


    # ランダムに上がるか, 下がるか予測する
    def simulate(self,path_tpx,path_daw,is_validate=False,start_year=2021,end_year=2021,start_month=1,end_month=12):
        x_check, y_check = self.make_check_data(path_tpx,path_daw)
        y_ = pd.DataFrame(y_check)
        y_.index = x_check.index
        x_check = self.return_split_df(x_check,start_year=start_year,end_year=end_year,start_month=start_month,end_month=end_month)
        y_ = self.return_split_df(y_,start_year=start_year,end_year=end_year,start_month=start_month,end_month=end_month)
        y_check = y_.values.reshape(-1).tolist()
        length = len(x_check)
        is_bought = False
        index_buy = 0
        index_sell = 0
        prf = 0
        prf_list = []
        trade_count = 0
        df_con = self.return_df_con(path_tpx,path_daw)
        df_con['ma_short'] = df_con['close'].rolling(self.ma_short).mean()
        df_con['ma_long']  = df_con['close'].rolling(self.ma_long).mean()
        df_con = df_con.iloc[self.ma_long:]
        df_con = self.return_split_df(df_con,start_year=start_year,end_year=end_year,start_month=start_month,end_month=end_month)
        pl = PlotTrade(df_con['close'],label='close')
        pl.add_plot(df_con['ma_short'],label='ma_short')
        pl.add_plot(df_con['ma_long'],label='ma_long')
        pl.add_plot(df_con['open'],label='open')
        self.pr_log = pd.DataFrame(index=x_check.index[:-1])
        self.pr_log['reward'] = [0.0] * len(self.pr_log)
        self.pr_log['eval_reward'] = self.pr_log['reward'].tolist()
        eval_price = 0
        total_eval_price = 0
        #********* acc_df?
        acc_df = pd.DataFrame(index=x_check.index)
        acc_df['pred'] = [-1] * len(acc_df)
#* 判定不能は -1, 騰貴予測は 1, 下落予測は 0
# is_observed=True としたことで買えなくなった取引の回数をカウント
        cant_buy = 0
        pclose = x_check['pclose']


        for i in range(length-1):
            
            total_eval_price = prf
            self.pr_log['reward'].loc[df_con.index[i]] = prf 
            self.pr_log['eval_reward'].loc[df_con.index[i]] = total_eval_price
            #*******  self.random_num で　up, down(あるいはstay） を 返す関数を実装
            label = self.random_func(self.random_num)

            if label==1 and pclose.iloc[i+1]>0:
                acc_df.iloc[i] = 1
            else: #label == 0 
                acc_df.iloc[i] = 0
                # x_check['dclose'].iloc[i] は観測可能 

            if not is_bought:
                # 買いのサイン
                if label==1:
                    index_buy = df_con['close'].loc[x_check.index[i+1]]
                    start_time = x_check.index[i+1]
                    is_bought = True
            else:
                # 売りのサイン
                if label==0:
                    index_sell = df_con['close'].loc[x_check.index[i+1]]
                    end_time = x_check.index[i+1]
                    prf += index_sell - index_buy
                    prf_list.append(index_sell - index_buy)
                    is_bought = False
                    trade_count += 1
                    pl.add_span(start_time,end_time)
                else:
                    eval_price = df_con['close'].iloc[i] - index_buy
                    total_eval_price += eval_price
                    self.pr_log['eval_reward'].loc[df_con.index[i]] = total_eval_price

            
            self.is_bought = is_bought
                  
        
        if is_bought:
            index_sell = df_con['close'].loc[x_check.index[-1]] 
            prf += index_sell - index_buy
            prf_list.append(index_sell - index_buy)
            end_time = x_check.index[-1]
            trade_count+=1
            pl.add_span(start_time,end_time)

        
        self.pr_log['reward'].loc[df_con.index[-1]] = prf 
        self.pr_log['eval_reward'].loc[df_con.index[-1]] = total_eval_price
        prf_array = np.array(prf_list)
        self.acc_df = acc_df
        self.y_check = y_check
            
        
        try:
            df = self.calc_acc(acc_df, y_check)
            self.accuracy_df = df
            log = self.return_trade_log(prf,trade_count,prf_array,cant_buy)
            self.trade_log = log

            if not is_validate:
                # ここも df 化できるように 
                print(log)
                print("")
                print(df)
                print("")
                pl.show()
        except:
            print("no trade")

# alpha, beta よくよく働きを吟味するように
# まだ仕組み理解してない
class DawSimulation(Simulation):

    def __init__(self,alpha=0,beta=0):
        super(DawSimulation,self).__init__()
        # 買いの閾値をalpha
        self.alpha = alpha
        # 売りの閾値をbeta
        self.beta = beta
        self.ma_short = 5
        self.ma_long = 25


    def make_check_data(self,path_tpx,path_daw):
        df = self.make_df_con(path_tpx,path_daw)
        x_check = df.iloc[1:]
        length = len(df)
        y_check = []
        for i in range(1,length):
            if df['pclose'].iloc[i] > 0:
                y_check.append(1)
            else:
                y_check.append(0)
        
        return x_check, y_check



    def simulate(self, path_tpx, path_daw, is_validate=False,strategy='normal',start_year=2021,end_year=2021,start_month=1,end_month=12,
                 is_variable_strategy=False,is_observed=False):
        x_check, y_check = self.make_check_data(path_tpx,path_daw)
        y_ = pd.DataFrame(y_check)
        y_.index = x_check.index
        x_check = self.return_split_df(x_check,start_year=start_year,end_year=end_year,start_month=start_month,end_month=end_month)
        y_ = self.return_split_df(y_,start_year=start_year,end_year=end_year,start_month=start_month,end_month=end_month)
        y_check = y_.values.reshape(-1).tolist()
        length = len(x_check)
        is_bought = False
        index_buy = 0
        index_sell = 0
        prf = 0
        trade_count = 0
        df_con = self.return_df_con(path_tpx,path_daw)
        df_con['ma_short'] = df_con['close'].rolling(self.ma_short).mean()
        df_con['ma_long']  = df_con['close'].rolling(self.ma_long).mean()
        df_con = df_con.iloc[self.ma_long:]
        df_con = self.return_split_df(df_con,start_year=start_year,end_year=end_year,start_month=start_month,end_month=end_month)
        pl = PlotTrade(df_con['close'],label='close')
        pl.add_plot(df_con['ma_short'],label='ma_short')
        pl.add_plot(df_con['ma_long'],label='ma_long')
        pl.add_plot(df_con['open'],label='open')
        prf_list = []
        self.pr_log = pd.DataFrame(index=x_check.index[:-1])
        self.pr_log['reward'] = [0.0] * len(self.pr_log)
        self.pr_log['eval_reward'] = self.pr_log['reward'].tolist()
        eval_price = 0
        total_eval_price = 0

        #********* acc_df?
        acc_df = pd.DataFrame(index=x_check.index)
        acc_df['pred'] = [-1] * len(acc_df)


#* 判定不能は -1, 騰貴予測は 1, 下落予測は 0
# is_observed=True としたことで買えなくなった取引の回数をカウント
        cant_buy = 0
        dclose = x_check['dclose']
        pclose = x_check['pclose']


        for i in range(length-1):
            
            total_eval_price = prf
            self.pr_log['reward'].loc[df_con.index[i]] = prf 
            self.pr_log['eval_reward'].loc[df_con.index[i]] = total_eval_price
            
            if is_variable_strategy:
                strategy = self.return_grad(df_con, index=i,gamma=0, delta=0)
            

#   ダウが上がる　-> 日経平均もあがることを仮定している
# 　つまり, dclose>0 -> label = 1
            if dclose.iloc[i+1]>0:
                acc_df.iloc[i] = 1
            else: #label == 0
                acc_df.iloc[i] = 0
                # x_check['dclose'].iloc[i] は観測可能 


            if strategy=='reverse':
            
                if not is_bought:
    #                 下がって買い
    # ダウ平均の変化率(%)が下がって買う -> 逆張り戦略
                    if x_check['dclose'].iloc[i]*100 < self.alpha:
#                         観測した始値が, 下がるという予測に反して上がっていた時, 買わない
                        if is_observed and df_con['open'].loc[x_check.index[i+1]] > df_con['close'].loc[x_check.index[i]]:
                            cant_buy += 1
                            continue
                        index_buy = df_con['close'].loc[x_check.index[i+1]]
                        start_time = x_check.index[i+1]
                        is_bought = True
                        
#                     
                else:
    #                 上がって売り
                    if x_check['dclose'].iloc[i]*100 > self.beta:
                        index_sell = df_con['close'].loc[x_check.index[i+1]]
                        end_time = x_check.index[i+1]
                        prf += index_sell - index_buy
                        prf_list.append(index_sell - index_buy)
                        is_bought = False
                        trade_count += 1
                        pl.add_span(start_time,end_time)
                    else:
                        eval_price = df_con['close'].iloc[i] - index_buy
                        total_eval_price += eval_price
                        self.pr_log['eval_reward'].loc[df_con.index[i]] = total_eval_price
                        
                        
            elif strategy=='normal':
                
                if not is_bought:
    #                 上がって買い
                    if x_check['dclose'].iloc[i]*100 > self.alpha:
#             上がるという予測に反して, 始値が前日終値より下がっていたら買わない
                        if is_observed and df_con['open'].loc[x_check.index[i+1]] < df_con['close'].loc[x_check.index[i]]:
                            cant_buy += 1
                            continue
                        index_buy = df_con['close'].loc[x_check.index[i+1]]
                        start_time = x_check.index[i+1]
                        is_bought = True
                else:
    #                 下がって売り
                    if x_check['dclose'].iloc[i]*100 < self.beta:
                        index_sell = df_con['close'].loc[x_check.index[i+1]]
                        end_time = x_check.index[i+1]
                        prf += index_sell - index_buy
                        prf_list.append(index_sell - index_buy)
                        is_bought = False
                        trade_count += 1
                        pl.add_span(start_time,end_time)
                    else:
                        eval_price = df_con['close'].iloc[i] - index_buy
                        total_eval_price += eval_price
                        self.pr_log['eval_reward'].loc[df_con.index[i]] = total_eval_price
            else:
                print("No such strategy.")
                return 
            
            self.is_bought = is_bought
                  
        
        if is_bought:
            index_sell = df_con['close'].loc[x_check.index[-1]] 
            prf += index_sell - index_buy
            prf_list.append(index_sell - index_buy)
            end_time = x_check.index[-1]
            trade_count+=1
            pl.add_span(start_time,end_time)

        
        self.pr_log['reward'].loc[df_con.index[-1]] = prf 
        self.pr_log['eval_reward'].loc[df_con.index[-1]] = total_eval_price
        prf_array = np.array(prf_list)
        self.acc_df = acc_df
        self.y_check = y_check
            
        
        try:
            df = self.calc_acc(acc_df, y_check)
            self.accuracy_df = df
            log = self.return_trade_log(prf,trade_count,prf_array,cant_buy)
            self.trade_log = log

            if not is_validate:
                print(log)
                print("")
                print(df)
                print("")
                pl.show()
        except:
            print("no trade")

class TPXSimulation(Simulation):


    def __init__(self):
        super(TPXSimulation,self).__init__()


    def simulate(self,path_tpx,path_daw,is_validate=False,start_year=2021,end_year=2021,start_month=1,end_month=12):
        df = self.make_df_con(path_tpx,path_daw)
        x_check = self.return_split_df(df,start_year=start_year,end_year=end_year,start_month=start_month,end_month=end_month)
        length = len(x_check)
        pl = PlotTrade(x_check['close'],label='close')
        prf_list = []
        


        start_time = x_check.index[0]
        end_time = x_check.index[-1]
        pl.add_span(start_time,end_time)
        prf = x_check['close'].loc[x_check.index[-1]] - x_check['close'].loc[x_check.index[0]]
        index_buy = x_check['close'].iloc[0]
        prf_list_diff = x_check['close'].map(lambda x : x - index_buy).diff().fillna(0).tolist()
        prf_list = x_check['close'].map(lambda x : x - index_buy).tolist()
        prf_array = np.array(prf_list)
        prf_array_diff = np.array(prf_list_diff)
        self.pr_log = pd.DataFrame(index=x_check.index[:-1])
        self.pr_log['reward'] = prf_list[:-1]
        self.pr_log['eval_reward'] = prf_list[:-1]
        log = self.return_trade_log(prf,length-1,prf_array_diff,0)
        self.trade_log = log

        if not is_validate:
            print(log)
            print("")
            pl.show()

class StrategymakerSimulation(XGBSimulation):


    def simulate(self, path_tpx, path_daw, sm, is_validate=False,is_online=False,start_year=2021,end_year=2021,start_month=1,end_month=12,ma_short=5,ma_long=25,theta=0.0001):
        x_check, y_check = self.make_check_data(path_tpx,path_daw)
        y_ = pd.DataFrame(y_check)
        y_.index = x_check.index
        x_check = self.return_split_df(x_check,start_year=start_year,end_year=end_year,start_month=start_month,end_month=end_month)
        y_ = self.return_split_df(y_,start_year=start_year,end_year=end_year,start_month=start_month,end_month=end_month)
        y_check = y_.values.reshape(-1).tolist()
        length = len(x_check)
        predict_proba = self.xgb_model.predict_proba(x_check.astype(float))
        is_bought = False
        index_buy = 0
        index_sell = 0
        prf = 0
        trade_count = 0
        df_con = self.return_df_con(path_tpx,path_daw)
        df_con['ma_short'] = df_con['close'].rolling(self.ma_short).mean()
        df_con['ma_long']  = df_con['close'].rolling(self.ma_long).mean()
        df_con = df_con.iloc[self.ma_long:-1]
        df_con = self.return_split_df(df_con,start_year=start_year,end_year=end_year,start_month=start_month,end_month=end_month)
        pl = PlotTrade(df_con['close'],label='close')
        pl.add_plot(df_con['ma_short'],label='ma_short')
        pl.add_plot(df_con['ma_long'],label='ma_long')
        prf_list = []
        self.pr_log = pd.DataFrame(index=x_check.index[:-1])
        self.pr_log.index = x_check.index
        self.pr_log['reward'] = [0.0] * len(self.pr_log)
        self.pr_log['eval_reward'] = self.pr_log['reward'].tolist()
        eval_price = 0
        total_eval_price = 0
#*      オンライン学習用の学習データ   
        x_tmp = x_check.copy()
        y_tmp = y_.copy()
        current_date = x_tmp.index[0]
        acc_df = pd.DataFrame()
        acc_df.index = x_tmp.index
        acc_df['pred'] = [-1] * len(acc_df)
        x_sm, y_sm = sm.make_train_data(path_tpx, path_daw,year=start_year,theta=theta)
        x_sm = self.return_split_df(x_sm,start_year=start_year,end_year=end_year,start_month=start_month,end_month=end_month)
        buy_sign = sm.model.predict(x_sm)
#* 判定不能は -1, 騰貴予測は 1, 下落予測は 0
        
        
        for i in range(length-1):
            
            
            row = predict_proba[i]
            label = np.argmax(row)
            prob = row[label]
            total_eval_price = prf
            
            self.pr_log['reward'].loc[df_con.index[i]] = prf 
            self.pr_log['eval_reward'].loc[df_con.index[i]] = total_eval_price
#             label==0 -> down
#             label==1 -> up
#*          オンライン学習
            tmp_date = x_tmp.index[i]
            if current_date.month!=tmp_date.month and is_online:
#             x_ = x_tmp.loc[:x_tmp.index]
                x_ = x_tmp[current_date<=x_tmp.index]
                x_ = x_[x_.index<tmp_date]
                y_ = y_tmp[current_date<=y_tmp.index]
                y_ = y_[y_.index<tmp_date]
#                 param_dist = {'objective':'binary:logistic', 'n_estimators':16,'use_label_encoder':False,
#                  'max_depth':4}
#                 tmp_xgb = xgb.XGBClassifier(**param_dist)
                self.xgb_model = self.xgb_model.fit(x_,y_)
                predict_proba = self.xgb_model.predict_proba(x_check.astype(float))
                current_date = tmp_date
            
            if prob > self.alpha:
                if label == 0:
                    acc_df.iloc[i] = 0
                else: #l able == 1 
                    acc_df.iloc[i] = 1
                    
#                     「買い」 サインの時
            if buy_sign[i]==1:
                if prob >0.5 :
                    strategy='normal'
                else:
                    strategy='reverse'
            else:
                strategy=None
                

            if strategy=='reverse' and buy_sign[i]==1:
            
                if not is_bought:
    #                 下がって買い
                    if label==0 and prob>self.alpha:
                        index_buy = df_con['close'].loc[x_check.index[i+1]]
                        start_time = x_check.index[i+1]
                        is_bought = True
                else:
    #                 上がって売り
                    if label==1 and prob>self.alpha:
                        index_sell = df_con['close'].loc[x_check.index[i+1]]
                        end_time = x_check.index[i+1]
                        prf += index_sell - index_buy
                        prf_list.append(index_sell - index_buy)
                        is_bought = False
                        trade_count += 1
                        pl.add_span(start_time,end_time)
                    else:
                        eval_price = df_con['close'].iloc[i] - index_buy
                        total_eval_price += eval_price
                        self.pr_log['eval_reward'].loc[df_con.index[i]] = total_eval_price
                        
                        
            elif strategy=='normal' and buy_sign[i]==1:
                
                if not is_bought:
    #                 上がって買い
                    if label==1 and prob>self.alpha:
                        index_buy = df_con['close'].loc[x_check.index[i+1]]
                        start_time = x_check.index[i+1]
                        is_bought = True
                else:
    #                 下がって売り
                    if label==0 and prob>self.alpha:
                        index_sell = df_con['close'].loc[x_check.index[i+1]]
                        end_time = x_check.index[i+1]
                        prf += index_sell - index_buy
                        prf_list.append(index_sell - index_buy)
                        is_bought = False
                        trade_count += 1
                        pl.add_span(start_time,end_time)
                    else:
                        eval_price = df_con['close'].iloc[i] - index_buy
                        total_eval_price += eval_price
                        self.pr_log['eval_reward'].loc[df_con.index[i]] = total_eval_price
            
            
            elif strategy==None:
                continue
                
        
        
        if is_bought:
            index_sell = df_con['close'].loc[x_check.index[-1]] 
            prf += index_sell - index_buy
            prf_list.append(index_sell - index_buy)
            end_time = x_check.index[-1]
            trade_count+=1
            pl.add_span(start_time,end_time)

        
        self.pr_log['reward'].loc[df_con.index[-1]] = prf 
        self.pr_log['eval_reward'].loc[df_con.index[-1]] = total_eval_price
        prf_array = np.array(prf_list)
        self.acc_df = acc_df
        self.y_check = y_check
            
        
        try:
            if not is_validate:
                print("Total profit :{}".format(prf))
                print("Trade count  :{}".format(trade_count))
                print("Max profit   :{}".format(prf_array.max()))
                print("Min profit   :{}".format(prf_array.min()))
                print("Mean profit  :{}".format(prf_array.mean()))
                if not is_online:
                    df = self.eval_proba(x_check,y_check)
                else:
                    df = self.calc_acc(acc_df, y_check)
                print(df)
                print("")
                pl.show()
        except:
            print("no trade")
  
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

class AnalyzePCA():


    def __init__(self):
        self.pca : PCA = None
        self.data = None


    def make_data(self,path_tpx,path_daw,test_rate=0.8):
        lx = LearnXGB()
        x_train,y_train,x_test,y_test = lx.make_xgb_data(path_tpx,path_daw,test_rate=0.8)
        return x_train,y_train,x_test,y_test

    
    def do_pca(self,path_tpx,path_daw,test_rate=0.8):
        x_train,y_train,x_test,y_test = self.make_data(path_tpx,path_daw,test_rate=test_rate)
        x_train = standarize(x_train)
        x_test = standarize(x_test)
        pca = PCA()
        pca.fit(x_train)
        self.pca = pca
        self.data = x_train


    def do_pca2(self,x_train):
        pca = PCA()
        pca.fit(x_train)
        self.pca = pca
        self.data = x_train

    
    def get_loadings(self,is_show=False):
        loadings = pd.DataFrame(self.pca.components_.T, index=self.data.columns)
        if is_show:
            print(loadings.head())
        return loadings

    
    def get_score(self,is_show=False):
        score = pd.DataFrame(self.pca.transform(self.data), index=self.data.index)
        if is_show:
            print(score.head())
        return score

    # 第一主成分, 第二主成分に対するデータのプロット
    def show_data_in_k1k2(self,num=5):
        # num : 可視化するデータ数を指定
        plt.subplots(figsize=(10, 10)) 
        score = self.get_score()
        plt.scatter(score.iloc[:num,0], score.iloc[:num,1]) 
        plt.rcParams["font.size"] = 11
        # プロットしたデータにサンプル名をラベリング
        for i in range(num):
            plt.text(score.iloc[i,0], score.iloc[i,1], score.index[i], horizontalalignment="center", verticalalignment="bottom")
        # 第一主成分
        plt.xlim(-5, 2.5)
        # 第二主成分
        plt.ylim(-3, -1)
        plt.yticks(np.arange(-3, -0.5, 0.5))
        plt.xlabel("t1")
        plt.ylabel("t2")
        plt.grid()
        plt.show()
        plt.clf()
            

    def get_contribution_ratios(self,is_show=False):
        contribution_ratios = pd.DataFrame(self.pca.explained_variance_ratio_)
        if is_show:
            print(contribution_ratios.head())
        return contribution_ratios


    def get_cumulative_contribution_ratios(self,is_show=False):
        contribution_ratios = self.get_contribution_ratios()
        cumulative_contribution_ratios = contribution_ratios.cumsum()
        if is_show:
            print(cumulative_contribution_ratios)
        return cumulative_contribution_ratios


    def show_cont_cumcont_rartios(self):
        contribution_ratios = self.get_contribution_ratios()
        cumulative_contribution_ratios = self.get_cumulative_contribution_ratios()
        cont_cumcont_ratios = pd.concat([contribution_ratios, cumulative_contribution_ratios], axis=1).T
        cont_cumcont_ratios.index = ['contribution_ratio', 'cumulative_contribution_ratio']  # 行の名前を変更
        # 寄与率を棒グラフで、累積寄与率を線で入れたプロット図を重ねて描画
        x_axis = range(1, contribution_ratios.shape[0] + 1)  # 1 から成分数までの整数が x 軸の値
        plt.rcParams['font.size'] = 18
        plt.bar(x_axis, contribution_ratios.iloc[:, 0], align='center')  # 寄与率の棒グラフ
        plt.plot(x_axis, cumulative_contribution_ratios.iloc[:, 0], 'r.-')  # 累積寄与率の線を入れたプロット図
        plt.xlabel('Number of principal components')  # 横軸の名前
        plt.ylabel('Contribution ratio(blue),\nCumulative contribution ratio(red)')  # 縦軸の名前。\n で改行しています
        plt.show()


    def show_band_gap(self):
        # 第 1 主成分と第 2 主成分の散布図 (band_gap の値でサンプルに色付け)
        score = self.get_score()
        plt.scatter(score.iloc[:, 0], score.iloc[:, 1], c=self.data.iloc[:, 0], cmap=plt.get_cmap('jet'))
        clb = plt.colorbar()
        clb.set_label('band_gap', labelpad=-20, y=1.1, rotation=0)
        plt.xlabel('t1')
        plt.ylabel('t2')
        plt.show()



