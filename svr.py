# SUPPORT VECTOR REGRESSION (SVR)

# Importing liberaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# Importing dataset
dataset = pd.read_csv("Position_Salaries.csv")



# Seperating dependent and independent variables
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# We reshaped it so that we could apply feature scaling on it, we basically conveted into an array
y=np.reshape(y,(len(y),1))


 
# Splitting the dataset into Training set and Test set
'''
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
'''


# We need to apply Feature Scaling manually when using SVR class (SVM liberary)

# Feature Scaling -> Normalizing the range of data/vairiable values

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x = sc_x.fit_transform(x)
# We can also rescale y if we need
# Here scaling of both x and y is necessary as the SVM liberary does not do that itself
sc_y = StandardScaler()
y = sc_y.fit_transform(y)





# Fitting SVR to the dataset
from sklearn.svm import SVR
# We need to choose which kernel we need and pass it as agrument
regressor = SVR(kernel = "rbf")
regressor.fit(x,y)




# Predicting a new result with SVR
# As we have scaled(normalized) y so we will also get normalized outputs, so to get non-normalizes outputs we reverse the scaling
y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))))
print(y_pred)




# Visualising the Polynomial SVR results
plt.scatter(x, y, color = 'red')
plt.plot(x, regressor.predict(x), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


'''
# Visualising the Regression results (for higher resolution and smoother curve)
# If we want the graph to be more accurate by making inputs complicated.
# It gives a much smoother curve.
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
'''