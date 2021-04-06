# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 16:33:57 2021

@author: Isabelle
"""

import itertools

p_in = [0.75, 0.5, 0.25, 0.1]
p_rec = [0.25, 0.1, 0.08]
N_c = [40, 30, 20, 10]
seed = [0, 1, 2]

params = itertools.product(p_in, p_rec, N_c, seed)
param_str = [[str(x) for x in p] for p in params]

with open("params.txt", "w") as file:
    for line in param_str:
        file.write(" ".join(line) + "\n")
