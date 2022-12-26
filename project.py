import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
df=pd.read_csv("gre_score.csv")
print(df) 
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
print(x)
print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)
model=LinearRegression()
model.fit(x_train,y_train)
pred=model.predict([[9.4]])
print("the prediction output of the model is: ",pred)