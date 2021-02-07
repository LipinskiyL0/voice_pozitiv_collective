# -*- coding: utf-8 -*-
"""
Для управления голосом создаем объект на нечеткой логике. 
R - матрица четких отношений, которую расширяем с помощью нечеткой логике
comands - перечень нечетких команд. Каждая команда описывается нечетким множеством
            comands[i] наименование i-й команды
terms - объект DataFrame содержит нечеткие множества описывающие команды из comands
        terms["u"] - содержит уиверсальное множество
        terms[comands[i]] - содержит функцию принадлежности для команды comands[i]
key_words  - перечень ключевых терминов, которые будут отыскиваться
               в речи и переводиться в нечеткие команды
               представляет собой Series где индекс = термин,
               а значение элемент из универсального множества
               params["U"]
               
params - словарь параметров
        
        params["R"] - матрица четких отношений между командами и действиями
                    R.columns - Значения управляющего параметра
                    R.loc[comands[i], columns[j]] принимает 1 если при команде
                                                  comands[i] управляющий параметр
                                                  должен принимать значение 
                                                  =columns[j], 0 иначе
        params["U"] - массив содержит минимум, максимум и шаг дискретизации для
                      построения универсального множества
        params["p_terms"] - содержит параметры термов для каждой команды
        params["key_words"] - перечень ключевых терминов, которые будут отыскиваться
                               в речи и переводиться в нечеткие команды
                               представляет собой Series где индекс = термин,
                               а значение элемент из универсального множества
                               params["u"]
                                                    

"""
import numpy as np
import pandas as pd
import skfuzzy as fuzz

import tensorflow as tf
from Operation_weights import get_weights
from Operation_weights import set_weights
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from DivClass import DivClass
from OptNelderMid import NelderMid



from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.neural_network import MLPRegressor
#from ConCorCoeff import concordance_correlation_coefficient as CCC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
#from moving_average import moving_average 
# from sklearn.externals import joblib
import joblib


class fuzzy_voice:
    
    def __init__(self):
        self.comands=[]
        self.R=[]
        self.terms=[]
        self.key_words=[]
        self.fuzzy_q=[]
        self.pop=[] #после того как дифэволюция отработае массив заполнится
                    #и будет использоваться как стартовая популяция для
                    #для следующей оптимизации
        self.fit=[] #пригодность для self.pop
        self.simplex=[] #аналогично для нелдера мида
        self.f=[] 
        
        
    
    def ini(self, params):
        
        self.comands=list(params["R"].index)
        
        self.R=params["R"].copy()
        
        t=np.arange(params["U"][0],params["U"][1]+params["U"][2]/2, params["U"][2])
        
        self.terms=pd.DataFrame(t[:, np.newaxis], columns=["U"])
        
        for c in self.comands:
            self.terms[c]=self.fterms(self.terms["U"], params["p_terms"].loc[c, :])
        self.key_words=params["key_words"].copy()
        
        self.fuzzy_q=pd.Series(np.zeros(len(self.comands)), index=self.comands)
        
        return True    
        
    
    def fterms(self, u, par):
        return  fuzz.trimf(u.values, np.array(par))
#        return  fuzz.gaussmf(u, par[0], par[1])
    
    def fuzzyfication(self, word):
        #фазификация 
        for s in self.comands:
            self.fuzzy_q[s] = fuzz.interp_membership(self.terms["U"], 
                                                    self.terms[s], 
                                                    self.key_words[word])
        return True
    
    def fuzzyfication_NN(self, x):
        #фазификация для нейросети
        #когда на входе не слово, а четкое значение
        for s in self.comands:
            self.fuzzy_q[s] = fuzz.interp_membership(self.terms["U"], 
                                                    self.terms[s], 
                                                    x)
        return True
    
    def calc_out(self, word):
        #вычисляем выход
        
        #производим фазификацию
        if self.fuzzyfication(word)==False:
            return False
        
        #нечеткий вывод
        P=list(self.R.columns)
        
        mus=pd.DataFrame(np.zeros([len(self.comands), len(P)+1]), index=self.comands, 
                      columns=["in"]+P)
        mus["in"]=self.fuzzy_q
        mus[P]=self.R
        #рассчитываем по каждому параметру принадлежность
        out=pd.Series()
        for p in P:
            ind=["in"]
            ind.append(p)
            out[p]=np.max(np.min(mus[ind], axis=1))
        
        s=0    
        for p in P:
            s+=out[p]*p
        s=s/np.sum(out)
        return s
    
    def calc_out_NN(self, X):
        #вычисляем выход для нейронной сети
        #когда входом является не ключевое слово, а четкие значения -
        #виртуальный тумлер
        ind=X<-1
        X[ind]=-1
        ind=X>1
        X[ind]=1
        
        result=[]
        for x in X:
            #производим фазификацию
            if self.fuzzyfication_NN(x)==False:
                return False
            
            #нечеткий вывод
            P=list(self.R.columns)
            
            mus=pd.DataFrame(np.zeros([len(self.comands), len(P)+1]), index=self.comands, 
                          columns=["in"]+P)
            mus["in"]=self.fuzzy_q
            mus[P]=self.R
            #рассчитываем по каждому параметру принадлежность
            out=pd.Series()
            for p in P:
                ind=["in"]
                ind.append(p)
                out[p]=np.max(np.min(mus[ind], axis=1))
            
            s=0    
            for p in P:
                s+=out[p]*p
            if np.sum(out)!=0:
                s=s/np.sum(out)
            else:
                s=0
            result.append(s)
        result=np.array(result)
        return result
    
    def ini_NN(self, n_inputs):
        #построение нейронной сети
        tf.compat.v1.reset_default_graph()
        
        n_inputs=n_inputs
        n_hidden1=10
        n_hidden2=5
        n_outputs=1
        
        tf.compat.v1.disable_eager_execution()
        X=tf.compat.v1.placeholder(tf.float32, shape=(None, n_inputs), name="X")
        y=tf.compat.v1.placeholder(tf.float32, shape=(None), name="y")
        
        
        with tf.name_scope("dnn"):
            hidden1=tf.compat.v1.layers.dense(X, n_hidden1, name="hidden1")
            bn1_act=tf.compat.v1.nn.elu(hidden1)
            
            hidden2=tf.compat.v1.layers.dense(bn1_act, n_hidden2, name="hidden2")
            bn2_act=tf.compat.v1.nn.elu(hidden2)
            
#            hidden3=tf.layers.dense(bn2_act, n_outputs, name="outputs")
            y_pred=tf.compat.v1.layers.dense(bn2_act, n_outputs, name="outputs")
#            y_pred=tf.nn.sigmoid(hidden3)
#            y_pred=tf.nn.tanh(hidden3)
        
#        with tf.name_scope("loss"):
#            model_out=self.calc_out_NN(y_pred)
#            error=model_out-y
#            loss=tf.reduce_mean(tf.square(error), name="loss")
#        
        weights=tf.compat.v1.all_variables()
        # init = tf.compat.v1.global_variables_initializer()
        init = tf.compat.v1.global_variables_initializer()
        
        self.NN={"weights":weights,
                 "init":init, 
                 "y_pred":y_pred,
                 "X":X
#                 "loss":loss
                 }
        
        return True
    
    def NN_calc(self, XX):
        #вычисление НН
#        init=tf.global_variables_initializer()
#        saver=tf.train.Saver()
#        with tf.Session() as sess:
#            init.run()
#            saver.restore(sess, "./model_NM.ckpt")
#            y_rez=self.NN["y_pred"].eval(feed_dict={self.NN["X"]:XX})
        
        y_rez=self.NN["y_pred"].eval(feed_dict={self.NN["X"]:XX})
        return y_rez
    
    def rosen(self, x, weights, y_pred, XX, yy):
        if set_weights(weights, x)==False:
            print("False")
        y=y_pred.eval(feed_dict={self.NN["X"]:XX})
        model_out=self.calc_out_NN(y)
        y_rez=np.mean((model_out-yy)**2)
        return y_rez

    def NN_opt(self, vectors, div=True, nm=False):
        vect=vectors.copy()
        yy=vect["transition"].values
        del  vect["transition"]
        try:
            del vect["Unnamed: 0"]
        except:
            pass
        XX=vect.values
        init=tf.global_variables_initializer()
        saver=tf.train.Saver()
        
        with tf.Session() as sess:
            init.run()
#            saver.restore(sess, "./model_NM.ckpt")            
            x=get_weights(self.NN["weights"])
            if div==True:
                print("DivEvolution")
                Div=DivClass()
                Div.inicialization(FitFunct=self.rosen, 
                                   Min=np.full( len(x[0, :]), -0.05), 
                                   Max=np.full(len(x[0, :]), 0.05),
                                   n_ind=100,
                                   args=(self.NN["weights"], self.NN["y_pred"],
                                      XX, yy),
                                   pop=self.pop,
                                   fit=self.fit
                                   )
                                     
                x, self.pop, self.fit =Div.opt(n_iter=100,
                                   f=0.2, 
                                   p=0.25,
                                   args=(self.NN["weights"], self.NN["y_pred"],
                                      XX, yy)
                                   )
                
             
            if nm==True:  
                print("Nelder-Mid")
#                res = minimize(, x, method='nelder-mead', 
#                                options={'xtol': 1e-1, 'disp': True}, 
#                                args=(self.NN["weights"], self.NN["y_pred"],
#                                      XX, yy))
                x, self.simplex, self.f=NelderMid(self.rosen, x, max_iter=10, alpha=1, betta=0.5, 
                            gamma=2, e=1e-10, k=0.1, args=(self.NN["weights"],
                            self.NN["y_pred"], XX, yy), simplex_t=self.simplex, 
                                                           f_t=self.f)

            
            set_weights(self.NN["weights"], x)
            save_path=saver.save(sess, "./model_NM.ckpt")
            
            y=self.NN["y_pred"].eval(feed_dict={self.NN["X"]:XX})
            model_out=self.calc_out_NN(y)
            y_rez=np.mean((model_out-yy)**2)
            print(y_rez)
            plt.figure()
            plt.plot(model_out, 'b')
            plt.plot(yy, 'r')
            if model_out.ndim==1:
                model_out=model_out[:, np.newaxis]
            if yy.ndim==1:
                yy=yy[:, np.newaxis]
            rez=np.hstack([yy, model_out])
            rez=pd.DataFrame(rez, columns=["real", "model"])
            rez.to_csv("rezult.csv")
            
        return y_rez
    
    def NN_from_file(self):
        #сеть должна быть уже инициализирована
        
        # init=tf.global_variables_initializer()
        init = tf.compat.v1.global_variables_initializer()
        saver=tf.compat.v1.train.Saver()
#        with tf.Session() as sess:
#            init.run()
#            saver.restore(sess, "./model_NM.ckpt")
#            self.sess=sess
        
        sess=tf.compat.v1.InteractiveSession()
        init.run()
        saver.restore(sess, "./model_NM.ckpt")
        self.sess=sess
        return
            
        
            
        
        
    
#    def model_sklearn(method="SVR", vectors):
#            if method=="LR":
#                lr=LR()
#            elif method=="Ridge":
#                lr=Ridge()
#            elif method=="Lasso":
#                lr=Lasso()
#            elif method=="MLPRegressor":
#                lr=MLPRegressor()
#            elif method=="SVR":
#                lr=SVR()
#            elif method=="KNR":
#                lr=KNR()
#            elif method=="RFR":
#                lr=RFR()
#            elif method=="GBR":
#                lr=GBR()
#            else:
#                print("unknown method")
#                return False
#            
#            lr=lr.fit(X_train, y_train[:,0])
#            y_mod_train=lr.predict(X_train)
#            y_mod_test=lr.predict(X_test)
#            c_train=CCC(y_train, y_mod_train[:, np.newaxis])
#            c_test=CCC(y_test, y_mod_test[:, np.newaxis])
#            return (lr, c_train, c_test)


        
        
        
        
            
        
        
        
        
        
        
        
        
    
        
        
        