# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 20:40:07 2020

@author: Леонид
"""
import numpy as np
def rosen( x):
    return np.sum(x**2)

def step2(f):
    #step 2
    #Находим наилучшую -l,  наихудшую -h и вторую по наилучшести -s
    ind=np.argsort(f)
    l=ind[0]
    s=ind[1]
    h=ind[-1]
    return l, s, h

def step3(simplex, h):
    #step3 находим центр тяжести вершин без наихудшей  
    #simplex - вектора расположенные по строкам
    #h - индекс наихудшей вершины
    rez=(np.sum(simplex, axis=0)-simplex[h, :])/(len(simplex[:, 0])-1)
    return rez

def step4(f, fxn2, e):
    #проверка критерия останова
    sigma=(np.sum((f-fxn2)**2)/len(f))**0.5
    if sigma<e:
        return True
    return False
    
def step5(xn2, xh, alpha):
    xn3=xn2+alpha*(xn2-xh)
    return xn3


    
def NelderMid(F, x, max_iter=10, alpha=1, betta=0.5, gamma=2, e=1e-10, k=1, args=[], simplex_t=[], f_t=[]):
    #реализация нелдера мида
    #F - оптимизируемя функция 
    #x - стратовая точка
    #max_iter - максимальное количество итераций
    #alpha, betta, gamma - параметры алгоритма (отражение, сжатие, растяжение)
    #e - критерий останова разброс значений функции
    #k - коэффициент разброса при создании начального симплекса
    
    x=np.array(x)
    if x.ndim>1:
        x=x[0, :]
    n=len(x) #размерность задачи
    
    if simplex_t==[]:
        #создаем стартовый симлпекс. 
        #step 1
        E=np.eye(n)
        E=E*k
    #    E=np.random.random([n,n])*2*k-k
        simplex=x+E
        if x.ndim==1:
            simplex=np.vstack([x[np.newaxis, :], simplex])
        else:
            simplex=np.vstack([x, simplex])
            
        if args!=[]:
            weights=[args[0] for i in range(n+1)]
            y_pred=[args[1] for i in range(n+1)]
            XX=[args[2] for i in range(n+1)]
            yy=[args[3] for i in range(n+1)]
            
            f=np.array(list(map(F, simplex, weights,  y_pred,XX, yy)))
        else:
            f=np.array(list(map(F, simplex)))
    else:
        #если в параметрах пришел стартовый симплекс берем его
        simplex=simplex_t.copy()
        f=f_t.copy()
    #step 2
    #Находим наилучшую наихудшую и среднюю вершины
    l, s, h=step2(f)
    flag=True
    it=0
    while (it<max_iter)&(flag==True): 
           
        #step 3 
        #находим центр тяжести без худшей вершины
        xn2=step3(simplex, h)
        if args!=[]:
            fxn2=F(xn2, args[0], args[1], args[2], args[3])
        else:
            fxn2=F(xn2)
        
        if step4(f, fxn2, e)==True:
            #сработал критерий останова
            flag=False
        #step 4 
        #проверка критерия останова выполняется циклом while
        
        #step 5 
        #выполнить процедуру отражения
        xn3=step5(xn2, simplex[h, :], alpha)
        
        if args!=[]:
            fxn3=F(xn3, args[0], args[1], args[2], args[3])
        else:
            fxn3=F(xn3)
        #step 6
        
        if fxn3<=f[l]:
            #отражение прошло удачно делаем растяжение
            xn4=xn2+gamma*(xn3-xn2)
            
            if args!=[]:
                fxn4=F(xn4, args[0], args[1], args[2], args[3])
            else:
                fxn4=F(xn4)
            
            if fxn4<f[l]:
                #растяжение прошло удачно заменяем худшую вершину
                simplex[h, :]=xn4
                f[h]=fxn4
            else:
                simplex[h, :]=xn3
                f[h]=fxn3
        elif (f[s]<fxn3)&(fxn3<=f[h]):
            #отражение прошло менее удачно выполняем операцию сзатия
            xn5=xn2+betta*(simplex[h, :]-xn2)
            
            if args!=[]:
                fxn5=F(xn5, args[0], args[1], args[2], args[3])
            else:
                fxn5=F(xn5)
            simplex[h, :]=xn5
            f[h]=fxn5
        elif (f[l]<fxn3)&(fxn3<=f[s]):
            simplex[h, :]=xn3
            f[h]=fxn3
        else:
            #отражение совсем прошло неудачно
            #делаем операцию редукции
            simplex=simplex[l, :]+0.5*(simplex[l, :]-simplex)
            if args!=[]:
                weights=[args[0] for i in range(n+1)]
                y_pred=[args[1] for i in range(n+1)]
                XX=[args[2] for i in range(n+1)]
                yy=[args[3] for i in range(n+1)]                    
                f=np.array(list(map(F, simplex, weights,  y_pred,XX, yy)))
            else:
                f=np.array(list(map(F, simplex)))

            
        l, s, h=step2(f)
        
        print("iteration: {0}, best solve: {1}".format(it, f[l]))
        
        it+=1
    return simplex[l, :], simplex, f

#nelder=NelderMid(rosen, [1,2,3, 4,5,6], max_iter=1000, e=1e-10, k=10)
#print(nelder)

      
        
        
    
    
        