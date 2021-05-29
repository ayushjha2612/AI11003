# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1It8nMnr1gVRoVhOxveExQ8-HYHcErXpd
"""

from scipy.stats import gamma
import matplotlib.pyplot as plt


# Checking for different values of lamda
lamda= 0.1
print("For lamda  = 0.1 ")

# option 1
n=int(1e5)
X= gamma.rvs(a=3,scale = 1/lamda,size = n)
summation_1 = sum(1/X)
theta_1 = (2*summation_1)/ n  # estimator
exp_1 = theta_1.mean()      # expectattion of estimator
Bias_1 = exp_1- lamda       # bias calculation
print("The value of bias for lamda= ",lamda ,"is", Bias_1)
print("As it can be seen the value of bias is close to zero,so estimator is unbiased ")

# option 2
n=10   # for small value of n
X= gamma.rvs(a=3,scale = 1/lamda, size=n)

summation_2 = sum(X_2)
xbar= summation_2/n
theta_2 = (3)/ xbar       # estimator
exp_2 = theta_2.mean()     # expectattion of estimator
Bias_2 = exp_2- lamda       # bias calculation
print("The value of bias for lamda= ",lamda ,"is", Bias_2)
print("As it can be seen the value of bias is not much close to zero,so estimator is not unbiased ")

# As MSE limit n tends to infinity should be zero so big value of n is chosen
# option 3
n=int(1e5)
X= gamma.rvs(a=3,scale = 1/lamda, size=n)
var_1=theta_1.var()        # var calculation
mse_1 = var_1+ Bias_1**2    # MSE calculation
print("The value of MSE for lamda= ",lamda ,"is", mse_1)
print("As it can be seen the value of MSE is close to zero, so estimator is consistent.")


# option 4
summation_2 = sum(X)
xbar= summation_2/n
theta_2 = (3)/ xbar
exp_2 = theta_2.mean()
Bias_2 = exp_2- lamda      # bias calculation for large n
var_2=theta_2.var()         # var calculation
mse_2 = var_2+ Bias_2**2     # MSE calculation
print("The value of MSE for lamda= ",lamda ,"is", mse_2)
print("As it can be seen the value of MSE is close to zero, so estimator is consistent.")
print()

# Checking for another value of lamda
print("For lamda  = 1 ")
lamda =1
# option 1
n=int(1e5)
X= gamma.rvs(a=3,scale = 1/lamda,size = n)
summation_1 = sum(1/X)
theta_1 = (2*summation_1)/ n  # estimator
exp_1 = theta_1.mean()      # expectattion of estimator
Bias_1 = exp_1- lamda       # bias calculation
print("The value of bias for lamda= ",lamda ,"is", Bias_1)
print("As it can be seen the value of bias is close to zero,so estimator is unbiased ")

# option 2
n=10   # for small value of n
X= gamma.rvs(a=3,scale = 1/lamda, size=n)

summation_2 = sum(X_2)
xbar= summation_2/n
theta_2 = (3)/ xbar       # estimator
exp_2 = theta_2.mean()     # expectattion of estimator
Bias_2 = exp_2- lamda       # bias calculation
print("The value of bias for lamda= ",lamda ,"is", Bias_2)
print("As it can be seen the value of bias is not much close to zero,so estimator is not unbiased ")

# As MSE limit n tends to infinity should be zero so big value of n is chosen
# option 3
n=int(1e5)
X= gamma.rvs(a=3,scale = 1/lamda, size=n)
var_1=theta_1.var()        # var calculation
mse_1 = var_1+ Bias_1**2    # MSE calculation
print("The value of MSE for lamda= ",lamda ,"is", mse_1)
print("As it can be seen the value of MSE is close to zero, so estimator is consistent.")


# option 4
summation_2 = sum(X)
xbar= summation_2/n
theta_2 = (3)/ xbar
exp_2 = theta_2.mean()
Bias_2 = exp_2- lamda      # bias calculation for large n
var_2=theta_2.var()         # var calculation
mse_2 = var_2+ Bias_2**2     # MSE calculation
print("The value of MSE for lamda= ",lamda ,"is", mse_2)
print("As it can be seen the value of MSE is close to zero, so estimator is consistent.")
print()

# Last check for lamda = 10
print("Final check for lamda =10 ")
lamda = 10
# option 1
n=int(1e5)
X= gamma.rvs(a=3,scale = 1/lamda,size = n)
summation_1 = sum(1/X)
theta_1 = (2*summation_1)/ n  # estimator
exp_1 = theta_1.mean()      # expectattion of estimator
Bias_1 = exp_1- lamda       # bias calculation
print("The value of bias for lamda= ",lamda ,"is", Bias_1)
print("As it can be seen the value of bias is close to zero,so estimator is unbiased ")

# option 2
n=10   # for small value of n
X= gamma.rvs(a=3,scale = 1/lamda, size=n)

summation_2 = sum(X_2)
xbar= summation_2/n
theta_2 = (3)/ xbar       # estimator
exp_2 = theta_2.mean()     # expectattion of estimator
Bias_2 = exp_2- lamda       # bias calculation
print("The value of bias for lamda= ",lamda ,"is", Bias_2)
print("As it can be seen the value of bias is not much close to zero,so estimator is not unbiased ")

# As MSE limit n tends to infinity should be zero so big value of n is chosen
# option 3
n=int(1e5)
X= gamma.rvs(a=3,scale = 1/lamda, size=n)
var_1=theta_1.var()        # var calculation
mse_1 = var_1+ Bias_1**2    # MSE calculation
print("The value of MSE for lamda= ",lamda ,"is", mse_1)
print("As it can be seen the value of MSE is close to zero, so estimator is consistent.")


# option 4
summation_2 = sum(X)
xbar= summation_2/n
theta_2 = (3)/ xbar
exp_2 = theta_2.mean()
Bias_2 = exp_2- lamda      # bias calculation for large n
var_2=theta_2.var()         # var calculation
mse_2 = var_2+ Bias_2**2     # MSE calculation
print("The value of MSE for lamda= ",lamda ,"is", mse_2)
print("As it can be seen the value of MSE is close to zero, so estimator is consistent.")
print()
print("Therefore option 1,3 and 4 are right by simulations.")






