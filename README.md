![shutterstock_1164353044](https://user-images.githubusercontent.com/52822987/120143771-5feaa800-c1fe-11eb-8a57-6f6c12fc80ae.jpg)


         # Machine-Learning-on-Salary-Insight

It has 2 columns — “Years of Experience” and “Salary” for 30 employees in a company. We have to predict the salary of an employee given how many years of experience they have.
Here is the code for your reference, although I have uploaded the jupyter notebook.

![pic](https://user-images.githubusercontent.com/52822987/120143985-c7a0f300-c1fe-11eb-984c-a5f85c7d9e02.JPG)


import pandas as pd
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,1].values

            # Step 2: Split data into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

            # Step 3: Fit Simple Linear Regression to Training Data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

          # Step 4: Make Prediction
y_pred = regressor.predict(X_test)

          # Step 5 - Visualize training set results
import matplotlib.pyplot as plt

          # Step 6 - Plot the actual data points of training set
plt.scatter(X_train, y_train, color = 'red')

         # Step 7 - Plot the regression line
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

       # Step 8 - Visualize test set results
import matplotlib.pyplot as plt


      # Step 9 - Plot the actual data points of test set
plt.scatter(X_test, y_test, color = 'red')

     # Step 10 - plot the regression line (same as above)
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

This project is quite helpful for beginners. 
Learnt from: Omair Aasim
