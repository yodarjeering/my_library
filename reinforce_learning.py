from enum import Enum
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from collections import deque
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import random
random.seed(777)


Experience = namedtuple("Experience", ["s","a","r","n_s","n_a","d"])



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

