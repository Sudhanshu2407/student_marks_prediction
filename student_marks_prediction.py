#Importing the important library.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings

warnings.filterwarnings("ignore")

#----------------------------------------------
#Now,here we read the dataset.

student_df=pd.read_csv(r"C:\sudhanshu_projects\project-task-training-course\student_info.csv")

#----------------------------------------------
#Now here we check the number of null values.

student_df.isnull().sum()

#Conclusion: In study_hours there are 5 null values,but in student_marks there is no null value.

#-------------------------------------------------
#Now here we fill the null value with mean of column.

student_df["study_hours"]=student_df["study_hours"].fillna(student_df["study_hours"].mean())

#-------------------------------------------------
#Now again check the null values.

student_df.isnull().sum()

#Conclusion: There are no null value is present.

#-------------------------------------------------
#Now we have to check the info of dataset.

student_df.info()

#Conclusion: The datatype of both column is float.

#--------------------------------------------------
#Now here we check the description of dataset.

student_df.describe()

#Conclusion: Here we get mean,min and max.

#---------------------------------------------------
#Now here we define dependent and independent variable.

x=student_df.drop("student_marks",axis=1) #Independent Variable.

y=student_df.drop("study_hours",axis=1) #Dependent variable.

#---------------------------------------------------
#Now we split the dataset into train and test.

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#---------------------------------------------------
#Now here we create linear regression model.

#---------------------------------------------
#Here we import the libraries.
from sklearn.linear_model import LinearRegression

#---------------------------------------------
#Here we create linear regression model object.
lr=LinearRegression()

#---------------------------------------------
#Here we train lr model.
lr.fit(x_train,y_train)

#---------------------------------------------
#Here we predict the values using lr model.
y_pred_lr=lr.predict(x_test)

#---------------------------------------------
#Here we calculate the accuracy of lr model.
score_lr=lr.score(x_test,y_test)
print(f"The accuracy of lr model is {score_lr}.")

#Conclusion:The accuracy of lr model is 95.21%.

#---------------------------------------------------
#Here we compare train and test accuracy.

training_accuracy_lr=lr.score(x_train,y_train)
print(f"The training accuracy of lr model is {training_accuracy_lr}.")

testing_accuracy_lr=lr.score(x_test,y_test)
print(f"The testing accuracy of lr model is {testing_accuracy_lr}.")

#Conclusion: As the training and testing accuracy is almost same,so it is best fit model.

#----------------------------------------------------
#Now we compare the y_test and y_pred_lr.

compare_df=pd.DataFrame(np.c_[x_test,y_test,y_pred_lr],columns=["x_test","y_test","y_pred"])
compare_df

#---------------------------------------------------
#Now we can use one non-linear model.

#---------------------------------------------
#Here we import the libraries.
from sklearn.tree import DecisionTreeRegressor

#---------------------------------------------
#Here we create decision tree regression model object.
dtr=DecisionTreeRegressor()

#---------------------------------------------
#Here we train dtr model.
dtr.fit(x_train,y_train)

#---------------------------------------------
#Here we predict the values using dtr model.
y_pred_dtr=dtr.predict(x_test)

#---------------------------------------------
#Here we calculate the accuracy of dtr model.
score_dtr=dtr.score(x_test,y_test)
print(f"The accuracy of dtr model is {score_dtr}.")

#Conclusion: Here the accuracy is decreases.(i.e 88.77)

#--------------------------------------------------------
#Now here we can do visualization.

sns.scatterplot(data=student_df,x="study_hours",y="student_marks")
plt.title("Study Hours Vs Student Marks")
plt.xlabel("Study Hours")
plt.ylabel("Student Marks")
plt.show()

#Conclusion: Here we seen that the two features have strong positive correlation.

plt.scatter(x_test,y_test,color="blue")
plt.plot(x_test,lr.predict(x_test),color="red")
plt.title("Actual Vs Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

#Conclusion: Here we seen that most of the predicted value is same as actual.and there is very less residual.

#----------------------------------------------------
#Here we save the lr model.

pickle.dump(lr,open(r"C:\sudhanshu_projects\project-task-training-course\student_marks_prediction.pkl","wb"))

#----------------------------------------------------
#Now here we load the model.

model=pickle.load(open(r"C:\sudhanshu_projects\project-task-training-course\student_marks_prediction.pkl","rb"))

#-------------------------------
#Now here we test the model.

model.score(x_test,y_test)
