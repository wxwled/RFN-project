# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 11:00:03 2020

@author: wxw
"""

from sympy import *

pa,qa,p1,p2,q1,q2 = symbols('pa,qa,p1,p2,q1,q2')
# q1 = p1
# q2 = p2
qa = pa
pi = [pa*q1**2,pa*p1*p2,pa*p1*(q1+q2),qa*p2*(q1+q2),qa*p1*p2,qa*q2**2]

H = 0
# H = 1
for i in pi:
    H += -i*log(i)
    # H *= i

dhdp1 = diff(H,p1)
dhdp2 = diff(H,p2)
dhdq1 = diff(H,q1)
dhdq2 = diff(H,q2)
dhdpa = diff(H,pa)
dhdqa = diff(H,qa)

print(simplify(dhdp1-dhdq1))
print(simplify(dhdp2-dhdq2))
print(simplify(dhdpa-dhdqa))
