# -*- coding: utf-8 -*-
"""
функции по чтению и записи весов нейросети по тензору 
weights=tf.all_variables()
при условии что перед построением 
tf.reset_default_graph()

"""
import numpy as np
#*******************************************************************
def get_nweights(weights):
    #считаем количество весов в нейронной сети
    s=0
    for w in weights:
        mus=w.eval() 
        s+=mus.size
        
    return s

#*******************************************************************
def get_weights(weights):
    #считываем веса из нейронной сети
    rez=[]
    for w in weights:
        mus=w.eval()   
        if rez==[]:
            rez=mus.reshape([1, mus.size])
        else:
            rez=np.hstack([rez, mus.reshape([1, mus.size])])
    return rez
#******************************************************************
def set_weights(weights, mas):
    #устанавливаем веса из mas в нейронную сеть
    if mas.ndim==1:
        mas=mas[np.newaxis, :]
    for w in weights:
        mus=w.eval()   
        x, x1=np.hsplit(mas, [mus.size])
        x=x.reshape(mus.shape)
        w.load(x)
        mas=x1
        
    if len(mas[0, :])!=0:
        return False
    return True
    
    
        

    