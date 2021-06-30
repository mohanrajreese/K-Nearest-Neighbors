import pandas as pd                                      # For importing data from neraya formats like CSV, JSON, SQL
from sklearn.model_selection import train_test_split     # Split arrays or matrices into random train and test subsets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix             # Compute confusion matrix to evaluate the accuracy 
                                                         #                              of a classification.
df = pd.read_csv('heart.csv')

y = df['output']
X = df.drop('output', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)
model = KNeighborsClassifier(n_neighbors=6)
model.fit(X_train,y_train)
predicted= model.predict(X_test)
print("The Length of training dataset is   : ",len(y_train))
print("The Length of testing dataset is    : ",len(y_test))
cm= confusion_matrix(y_test, predicted)
accuracy=(cm[0][0]+cm[1][1])/len(y_test)
precision=cm[0][0]/(cm[0][0]+cm[0][1])
recall=cm[0][0]/(cm[0][0]+cm[1][0])
print("The Accuracy value is               : ",accuracy*100)
print("The Precision value is              : ",precision)
print("The recall value is                 : ",recall)