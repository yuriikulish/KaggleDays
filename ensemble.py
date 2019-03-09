# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 16:31:08 2019

@author: User
"""

import pandas as pd

subm1 = pd.read_csv("./submissions/subm_m1.csv")
subm2 = pd.read_csv("./submissions/subm_m2.csv")
subm3 = pd.read_csv("./submissions/subm_m2.csv")

average_subm = subm1.copy()
average_subm["target"] = subm1["target"]*0.4 + subm2["target"]*0.3 + subm3["target"]*0.3

average_subm[["ID","target"]].to_csv("submissions/ensemble.csv", index = False)



