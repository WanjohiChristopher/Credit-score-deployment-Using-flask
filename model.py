

# load Pandas
import pandas as pd



#pickle
import pickle

df=pd.read_csv('train3.csv')

X=df.drop(columns=['default'])
y=df.default

# scaling our features

from sklearn.preprocessing import MinMaxScaler
X_sc = MinMaxScaler().fit_transform(X)

# splitting into training and test sets 80-20
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X_sc,y,test_size=0.2,random_state=5,stratify = y)

# creating a balanced dataset
from imblearn.over_sampling import SMOTE
smt=SMOTE()
X_train,y_train=smt.fit_sample(X_train,y_train)

# implementing logistic regression
#from sklearn.linear_model import LogisticRegression
#logistic_classifier=LogisticRegression(solver='saga')
#logistic_classifier.fit(X_train,y_train)

#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
print('y_predict is:',y_pred)
# evaluating the model
#y_predict=logistic_classifier.predict(X_test)
#print('y_predict is:',y_predict)
# evaluating the model
from sklearn.metrics import accuracy_score

print("accuracy_score:")
#print(accuracy_score(y_predict,y_test))
print(accuracy_score(y_pred,y_test))

# Generating our pickle file

pickle.dump(classifier, open("logistic_classifier.pkl", "wb")) 
