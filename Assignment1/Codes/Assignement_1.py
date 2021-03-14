import numpy as np
from scipy.stats import bernoulli

#Number of members in the meeting is assumed 1000, X=1 denotes if the person is in favour and X=0 denotes if he is in oppose
simlen=int(1000)

#Probability of the event selected member is in favour, i.e., X=1 is 0.7
prob = 0.7

#Generating sample data using Bernoulli r.v.
data_bern = bernoulli.rvs(size=simlen,p=prob)
#Calculating the number of favourable outcomes
err_ind = np.nonzero(data_bern == 1)
#calculating the simulatedprobability
err_n = np.size(err_ind)/simlen

# Calculating E(X) using E(x)= n*p where p is simulated probability
exp_X= 1*err_n

# Calculating Var(X) using Var(x)= n*p*(1-p) where p is simulated probability
Var_X=1*err_n*(1-err_n)
#Theory vs simulation

print("The E(X) obtained after simulaion is",exp_X)
print("The E(X) obtained after simulaion is",Var_X)
print() 
print("The E(X) obtained theoretically is",1*prob)
print("The E(X) obtained theoretically is",1*prob*(1-prob))
print("The results obtained are approximately same.")



