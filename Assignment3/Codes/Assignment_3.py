# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1It8nMnr1gVRoVhOxveExQ8-HYHcErXpd
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli

# For plotting the CDF of Z
# Setting the corresponding x and y - coordinates
# Region 1 of CDF
x_1 = np.arange(-5, -3, 0.01)
y_1 = x_1*0
# Region 2 of CDF
x_2 =   np.arange(-3, -1, 0.01)
y_2=  (x_2**2 +6*x_2 + 9)/12
# Region 3 of CDF
x_3 =   np.arange(-1, 0, 0.01)
y_3=  (x_3+2)/3
# Region 4 of CDF
x_4 =   np.arange(0, 2, 0.01)
y_4=  (8 + 4* x_4 - x_4**2)/12
# Region 5 of CDF
x_5 =   np.arange(2, 5, 0.01)
y_5=  x_5**0

# Plotting the points
plt.plot(x_1, y_1,'b')
plt.plot(x_2, y_2,'b')
plt.plot(x_3, y_3,'b')
plt.plot(x_4, y_4,'b')
plt.plot(x_5, y_5,'b')
plt.xlabel('Random Variable , Z')
plt.ylabel('$F_{Z}(z)$')

# function to show the plot
plt.grid()
plt.show()

# For plotting the PDF of Z
# Setting the corresponding x and y - coordinates
# Region 1 of PDF
x_1 = np.arange(-5, -3, 0.01)
y_1 = x_1*0
# Region 2 of PDF
x_2 =   np.arange(-3, -1, 0.01)
y_2=  (x_2 + 3)/6
# Region 3 of PDF
x_3 =   np.arange(-1, 0, 0.01)
y_3=  (1)/3 *(x_3/x_3)
# Region 4 of PDF
x_4 =   np.arange(0, 2, 0.01)
y_4=  (2- x_4)/6
# Region 5 of PDF
x_5 =   np.arange(2, 5, 0.01)
y_5=  x_5*0

# Plotting the points
plt.plot(x_1, y_1,'b')
plt.plot(x_2, y_2,'b')
plt.plot(x_3, y_3,'b')
plt.plot(x_4, y_4,'b')
plt.plot(x_5, y_5,'b')
plt.xlabel('Random Variable , Z')
plt.ylabel('$Pr(z)$')

# function to show the plot
plt.grid()
plt.show()