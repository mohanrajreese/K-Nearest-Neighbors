import pandas as pd                                      
from sklearn.model_selection import train_test_split    
from sklearn.neighbors import KNeighborsClassifier      
from sklearn.metrics import confusion_matrix            
df = pd.read_csv('heart.csv')

y = df['output']
X = df.drop('output', axis = 1)                     
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,shuffle = True)
model = KNeighborsClassifier(n_neighbors=7)
model.fit(X_train,y_train)
predicted= model.predict(X_test)

print("The Total Length of our dataset is  : ",len(df))
print("The Length of training dataset is   : ",len(y_train))
print("The Length of testing dataset is    : ",len(y_test))
cm= confusion_matrix(y_test, predicted)
accuracy=(cm[0][0]+cm[1][1])/len(y_test)
precision=cm[0][0]/(cm[0][0]+cm[0][1])
recall=cm[0][0]/(cm[0][0]+cm[1][0])
print("The Accuracy value is               : ",accuracy*100)
print("The Precision value is              : ",precision)
print("The recall value is                 : ",recall)