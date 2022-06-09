import numpy as np
import pandas as pd
# from tensorflow.python import keras as K
import random
random.seed(777)
from library import *
from func_library import *


class Simulation():


    def __init__(self):
        self.model = None
        self.accuracy_df = None
        self.trade_log = None
        self.pr_log = None


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
        mk = MakeTrainData(df_con,test_rate=1.0)
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


class TechnicalSimulation(Simulation):
    
    
    def __init__(self,ma_short=5, ma_long=25, hold_day=25, year=2021):
        super(TechnicalSimulation,self).__init__()
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.hold_day = hold_day
        self.year = year
        
        
    def process(self,df):
        df_process = df.copy()
        df_process['ma_short'] = df_process['close'].rolling(self.ma_short).mean()
        df_process['ma_long']  = df_process['close'].rolling(self.ma_long).mean()
        # return df_process[df_process.index.year==self.year]
        return df_process
    
    
    def is_buyable(self, short_line, long_line, index_):
#         1=<index<=len-1 仮定
        long_is_upper = long_line.iloc[index_-1]>=short_line.iloc[index_-1]
        long_is_lower = long_line.iloc[index_+1]<=short_line.iloc[index_+1]
        buyable = long_is_upper and long_is_lower
        return buyable
    
    
    def is_sellable(self, short_line, long_line, index_):
        long_is_lower = long_line.iloc[index_-1]<=short_line.iloc[index_-1]
        long_is_upper = long_line.iloc[index_+1]>=short_line.iloc[index_+1]
        sellable = long_is_upper and long_is_lower
        return sellable

        
        
    def simulate(self,path_tpx,path_daw,is_validate=False,start_year=2021,end_year=2021,start_month=1,end_month=12):
        df = self.process(self.make_df_con(path_tpx,path_daw))
        df_process = self.return_split_df(df,start_year=start_year,end_year=end_year,start_month=start_month,end_month=end_month)
        is_bought = False
        hold_count_day = 0
        index_buy = 0
        pl = PlotTrade(df_process['close'],label='close')
        pl.add_plot(df_process['ma_short'],label='ma_5')
        pl.add_plot(df_process['ma_long'],label='ma_25')   
        prf = 0
        prf_list = []
        start_time = 0
        end_time = 0
        short_line = df_process['ma_short']
        long_line  = df_process['ma_long']
        trade_count = 0
        self.pr_log = pd.DataFrame(index=df_process.index[self.ma_short:-1])
        self.pr_log['reward'] = [0.0] * len(self.pr_log)
        self.pr_log['eval_reward'] = self.pr_log['reward'].tolist()
        eval_price = 0
        total_eval_price = 0
        
        
        for i in range(self.ma_short,len(df_process)-1):
            
            
            total_eval_price = prf
            self.pr_log['reward'].loc[df_process.index[i]] = prf 
            self.pr_log['eval_reward'].loc[df_process.index[i]] = total_eval_price
            if not is_bought:
                
                if self.is_buyable(short_line,long_line,i):
                    index_buy = df_process['close'].iloc[i]
                    is_bought = True
                    start_time = df_process.index[i]
                    hold_count_day = 0
                else:
                    continue
            
            
            else:
                
                if self.is_sellable(short_line,long_line,i) or hold_count_day==self.hold_day:
                    index_cell = df_process['close'].iloc[i]
                    end_time = df_process.index[i]
                    prf += index_cell - index_buy
                    prf_list.append(index_cell - index_buy)
                    total_eval_price = prf
                    self.pr_log['reward'].loc[df_process.index[i]] = prf 
                    self.pr_log['eval_reward'].loc[df_process.index[i]] = total_eval_price
                    trade_count+=1
                    is_bought = False
                    hold_count_day = 0
                    pl.add_span(start_time,end_time)
                else:
                    hold_count_day+=1
                    eval_price = df_process['close'].iloc[i] - index_buy
                    total_eval_price += eval_price
                    self.pr_log['eval_reward'].loc[df_process.index[i]] = total_eval_price
                    
        
        if is_bought and hold_count_day>0:
            end_time = df_process['close'].index[-1]
            pl.add_span(start_time,end_time)
            eval_price = df_process['close'].iloc[-1] - index_buy
            prf_list.append(df_process['close'].iloc[-1] - index_buy)
            total_eval_price += eval_price
            self.pr_log['eval_reward'].loc[df_process.index[-1]] = total_eval_price
        
        prf_array = np.array(prf_list)
        log = self.return_trade_log(prf,trade_count,prf_array,0)
        self.trade_log = log

        if not is_validate:        
            print(log)
            print("")
            pl.show()    
          
    
class XGBSimulation(Simulation):
    
    
    def __init__(self, xgb_model, alpha=0.70):
        super(XGBSimulation,self).__init__()
        self.xgb_model = xgb_model
        self.alpha = alpha
        self.acc_df = None
        self.y_check = None
        self.ma_long = 0
        self.ma_short = 0
        self.is_bought = False
        
    
    # online 学習の時だけ使ってる
    def eval_proba(self, x_test, y_test):
        predict_proba = self.xgb_model.predict_proba(x_test.astype(float))
        df = pd.DataFrame(columns = ['score','Up precision','Down precision','Up recall','Down recall','up_num','down_num'])
        j=0
        acc_dict = {'TU':0,'FU':0,'TD':0,'FD':0}
        
        
        for i in range(len(predict_proba)):
            row = predict_proba[i]
            label = np.argmax(row)
            proba = row[label]
            if proba > self.alpha:
                if y_test[i]==label:
                    if label==0:
                        acc_dict['TD'] += 1
                    else:
                        acc_dict['TU'] += 1
                else:
                    if label==0:
                        acc_dict['FD'] += 1
                    else:
                        acc_dict['FU'] += 1


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
            down_num
            col_list = [score,prec_u,prec_d,recall_u,recall_d,up_num,down_num]
            df.loc[j] = col_list
            j+=1
            return df
        except:
            print("division by zero")
            return None
        
    
#*    日付変更できるように変更
    def simulate(self, path_tpx, path_daw, is_validate=False,strategy='normal',is_online=False,start_year=2021,end_year=2021,start_month=1,end_month=12,
                is_variable_strategy=False,is_observed=False):
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
#*      オンライン学習用の学習データ   
        x_tmp = x_check.copy()
        y_tmp = y_.copy()
        current_date = x_tmp.index[0]
        acc_df = pd.DataFrame(index=x_tmp.index)
        acc_df['pred'] = [-1] * len(acc_df)
#* 判定不能は -1, 騰貴予測は 1, 下落予測は 0
# is_observed=True としたことで買えなくなった取引の回数をカウント
        cant_buy = 0
        
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
            
# ここのprob は2クラスうち, 出力の大きいほうのクラスの可能性が代入されている
            if prob > self.alpha:
                if label == 0:
                    acc_df.iloc[i] = 0
                else: #l able == 1 
                    acc_df.iloc[i] = 1
            
            
            if is_variable_strategy:
                strategy = self.return_grad(df_con, index=i,gamma=0, delta=0)
            
                
            if strategy=='reverse':
            
                if not is_bought:
    #                 下がって買い
                    if label==0 and prob>self.alpha:
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
                        
                        
            elif strategy=='normal':
                
                if not is_bought:
    #                 上がって買い
                    if label==1 and prob>self.alpha:
#             上がるという予測に反して, 始値が前日終値より下がっていたら買わない
                        if is_observed and df_con['open'].loc[x_check.index[i+1]] < df_con['close'].loc[x_check.index[i]]:
                            cant_buy += 1
                            continue
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
            if not is_online:
                df = self.eval_proba(x_check,y_check)
            else:
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

      
    #  get_accuracy　関数作ったら無用になる
    def return_accuracy(self, path_tpx,path_daw,strategy='normal',is_online=False,start_year=2021,start_month=1):
        self.simulate(path_tpx,path_daw,is_validate=True,strategy=strategy,is_online=is_online)
        y_check = pd.DataFrame(self.y_check)
        y_check.index = self.acc_df.index
        acc_df = self.acc_df.copy()
        acc_df = acc_df[acc_df.index.year==start_year]
        acc_df = acc_df[acc_df.index.month>=start_month]
        y_check = y_check[y_check.index.year==start_year]
        y_check = y_check[y_check.index.month>=start_month]
        df = self.calc_acc(acc_df,y_check.values)
        return df
    
        
    def show_result(self, path_tpx,path_daw,strategy='normal'):
        x_check, y_check = self.make_check_data(path_tpx,path_daw)  
        self.simulate(x_check,y_check,strategy)
        

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

