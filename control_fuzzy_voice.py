# -*- coding: utf-8 -*-
"""
управляющий класс для системы fuzzy_voice

case - массив переходов. формат: case[i, :] - информация по i-му переходу
       case[i, 0] - команда на естественном языке после токинезации и 
       лемингизации, case[i, 1] - текущее состояние. т.е. состояние из которого
       необходимое сделать переход, case[i, 2] - текущая цель, т.е. то состояние
       в котором на самом деле должен был быть объект на шаге i, case[i, 3] - 
       цель в которую мы должны перейти. 
       case[i, 0] - команда, которая должна перевести объект из case[i, 1] в
                    case[i, 3]. 
       case[i, 3]-case[i, 1] - разница которую нужно покрыть на текущем переходе
       case[:, 1]-case[:, 2] - разница между желаемыми состояниями и действительными

"""
from t4 import fuzzy_voice
import pandas as pd
import numpy as np
from t4 import fuzzy_voice
import matplotlib.pyplot as plt
from drawnow import drawnow # import lib from drawnow

import string
import pymorphy2
import nltk
from nltk.corpus import stopwords
from tokenize_me import tokenize_me




class control_fuzzy_voice:
    
    def __init__(self, param):
        self.contr=fuzzy_voice()
        self.contr.ini(param)
        self.mode="case"
        
        
        
    def model_sim(self, mode="case",  it=50):
        self.mode=mode
        self.it=it
        self.fig=plt.figure()
        drawnow(self.makeFig)

        
    
    def makeFig(self): # creat a function to make plot
        plt.style.use('classic')
        morph = pymorphy2.MorphAnalyzer()
        fig=self.fig
        mode=self.mode
        it=self.it
        n=it
        xs = [0]
        ys = [0]
        ys_flc=[0]
        k=1
        out=0
        goal1=0
        self.case=[]  
        if mode == "NN":
            #режим нейросети
            #активируем нейросеть запускаем ее из файла по умолчанию
            vectors=pd.read_csv("vectors.csv")
            try:
                del vectors["Unnamed: 0"]
            except:
                pass
            musor,m=vectors.shape
            self.contr.ini_NN(m-1)
            self.contr.NN_from_file()
            words_vector=self.get_vector()
            
        for i in range(1,n):
            fig.clear() 
            plt.ylim(-1, 3)
            plt.xlim(-1, it+1)
            goal0=goal1
            goal1=np.random.rand()
            
            if i==1:
                goal1=0.5
            
            x=k*out
            
            xs.append(i-1)
            xs.append(i)
            ys.append(ys[-1]+x)
            ys.append(ys[-1])
            ys_flc.append(goal0)
            ys_flc.append(goal0)
            plt.plot(xs, ys, "b", linewidth=3)
            plt.plot(xs, ys_flc, "g", linewidth=2)
            plt.plot(np.arange(i, n), np.full(n-i, goal1), "r")
            plt.legend(["object", "goal past", "goal future"])
            plt.pause(0.1)
            
            
            result=input("команда: ")
            result = result.lower()
            
            if "стоп" in result:
                break
            if ("вперед" in result)|("вверх" in result) :
                k=1
            elif ("назад" in result)|("вниз" in result):
                k=-1
            else:
                k=0
            
            
            if mode=="NN":
                #Приводим команду к мешку слов. 
                #проводим токенизацию - парсим на слова и убираем не информативные
                result=tokenize_me(result)
                mus=[]
                for r in result:
                    y=morph.parse(r)[0]
                    mus.append(y.normal_form)
                    
                result=list(set(mus)) #убираем повторяющиеся слова
                #преобразовываем обратно в строку
                mus=""
                for r in result:
                    mus+=r+" "
                    
                self.case.append([mus, ys[-1], goal0, goal1])
            else:
                self.case.append([result, ys[-1], goal0, goal1])
                
                
            if mode=="key_words":
                out=self.control_keywords(result)
            elif mode=="case":
                k=1
                out=goal1-ys[-1]
            elif mode=="NN":
                v=self.rez_to_vector(mus, words_vector)
                y_rez=self.contr.NN_calc( v[np.newaxis, :])
                out=self.contr.calc_out_NN(y_rez)
                out=out[0]
                
            else:
                out=0
                
                
        self.case=pd.DataFrame(self.case)
        self.case.columns=["comands", "real_current", "goal_current", "goal_next"]
        try:
            self.case.to_csv("case.csv")
        except:
            print("ошибка вывода в файл")    
        print("ошибка управления: ", np.mean((self.case["real_current"]-self.case["goal_current"])**2))
        if mode=="NN":
            self.contr.sess.close()
        return
    
    def control_keywords(self, result):
        #управление по ключевому слову
         #составляем массив длин команд для того, что бы проверять сначала
        #самые длинные команды
        key_words_name=list(self.contr.key_words.index)
        mas_len=np.zeros(len(key_words_name), dtype=int)
        for i in range(len(key_words_name)):
            mas_len[i]=len(key_words_name[i])
        ind_mas_len=np.argsort(mas_len)
        ind_mas_len=ind_mas_len[::-1]
        
        word=""
        for ind in ind_mas_len:
            if key_words_name[ind] in result:
                word=key_words_name[ind]
                break
        if word=="":
            out=0
        else:
            out=self.contr.calc_out(word)
        return out
    
    def make_vectors(self):
        #функция работает когда файл case.csv заполнен 
        #этот файл заполняется работой симуляции self.model_sim(mode="case")
        #на оснве технологии "мешок" слов формируем обучающую выборку
        case=pd.read_csv("case.csv")
        case['transition']=case["goal_next"]-case["real_current"]
        case['transition']=case['transition'].abs()
        words=[]
        for comand in case["comands"]:
            result=tokenize_me(comand)
            words=words+result
        words=list(set(words))
        
        words=pd.Series(words)
        words.columns=[""]
        words.to_csv("key_vector.csv")
        
        #формируем вектора
        vectors=[]
        for comand in case["comands"]:
            vector=[]
            for word in words:
                if word in comand:
                    vector.append(1)
                else:
                    vector.append(0)
            vectors.append(vector)
            
        vectors=pd.DataFrame(vectors, columns=list(words))
        
        result=pd.concat([vectors, case[['transition']]], sort=False, axis=1)
            
        result.to_csv("vectors.csv")
        return
    def analisys_vectors(self):
        #Производится анализ векторов полученных в ходе иммитации управления пользователем
        #1. Выделяются уникальные вектора и их частоты
        #2. Если частота больше 1 для вектора считается дисперсия
        #3. результат выводится в файл analisys_vectors.csv
        df=pd.read_csv("vectors.csv")
        col=list(df.columns)
        col=col[:-1]
        
        c1=df.groupby(col)["transition"].mean()
        c2=df.groupby(col)["transition"].std()
        c3=df.groupby(col)["transition"].count()
        
        ban=pd.DataFrame({"mean":c1, "std":c2, "count":c3})
        ban.to_csv("analysis_vectors.csv")
        ban=pd.read_csv("analysis_vectors.csv")
        
        return True
        
        
    def opt_NN(self, div=True, nm=False):
        vectors=pd.read_csv("vectors.csv")
        try:
            del vectors["Unnamed: 0"]
        except:
            pass    
        n,m=vectors.shape
                
        self.contr.ini_NN(m-1)
        self.contr.NN_opt(vectors, div=div, nm=nm)
                        
        return
    
    def get_vector(self):
        #на оснве файла vectors
        #строим вектор ключевых слов
        vectors=pd.read_csv("vectors.csv")
        del vectors["transition"]
        try:
            del vectors["Unnamed: 0"]
        except:
            pass
        words_vector=list(vectors.columns)
        return words_vector
        
        
    def rez_to_vector(self, result, words_vector):
        #принимаем строку result - строка команда на естественном языке
        #и переводим ее в вектор признаков
        result=tokenize_me(result)
        mus=[]
        morph = pymorphy2.MorphAnalyzer()
        for r in result:
            y=morph.parse(r)[0]
            mus.append(y.normal_form)
            
        result=list(set(mus)) #убираем повторяющиеся слова
        #преобразовываем обратно в строку
        mus=""
        for r in result:
            mus+=r+" "
        
        v=[]
        for word in words_vector:
            if word in mus:
                v.append(1)
            else:
                v.append(0)
        v=np.array(v)
        return v
    
   
    
    

        
    