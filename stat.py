# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 22:58:56 2020

@author: Леонид
"""

stat=pd.DataFrame([[st0["NN"].min(), st0["KW"].min()],
                    [st0["NN"].max(), st0["KW"].max()],
                    [st0["NN"].mean(), st0["KW"].mean()],
                    [st0["NN"].std(), st0["KW"].std()]],
                 index=["min", "max", "mean", "std"], 
                 columns=["NN", "KW"])

st_rez=pd.concat([st0, stat])
st_rez.to_csv("rez_all.csv")
st_rez=st_rez.round(6)
st_rez.to_csv("rez_all_round.csv")
