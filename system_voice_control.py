# -*- coding: utf-8 -*-
"""
класс создан для управление классом fuzzy_voice, который и реализует основные 
вычисления



"""
#Готовим контроллер
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
from control_fuzzy_voice import control_fuzzy_voice
    
comands=["сильно-отрицательно", "нормально-отрицательно",
         "ноль", "нормально", "сильно"]
key_words_name=["очень-очень слабо", "очень слабо", "слабо", "неочень слабо", 
            "слабо-ближе к среднему", "средне", "средне - ближе к сильному", 
            "неочень сильно", "сильно", "очень сильно", "очень-очень сильно"]
key_words=pd.Series(np.linspace(0.025, 1, len(key_words_name)), index=key_words_name)
R=pd.DataFrame([[1, 0, 0, 0, 0],[0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],[0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],], 
                columns=[-1,-0.5, 0, 0.5, 1], index=comands)
U=[-1.1, 1.1, 0.1]
p_terms=[[-1,-1, -0.5],[-1, -0.5, 0],[-0.5,0,0.5],[0,0.5,1],[0.5,1,1]]
#p_terms=[[0,0.1],[0.5,0.05],[1,0.1]]
p_terms=pd.DataFrame(p_terms, index=comands)
params={"R":R, "U":U, "p_terms":p_terms, "key_words":key_words}

S=control_fuzzy_voice(params)
# S.model_sim(mode="case", it=200)
# S.model_sim(mode="key_words", it=20)
#S.make_vectors()
#S.analisys_vectors()
#S.analisys_vectors()
#S.opt_NN(div=True, nm=False)
S.model_sim(mode="NN", it=20)
