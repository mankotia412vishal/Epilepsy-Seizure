import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
import warnings
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
warnings.filterwarnings('ignore')
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def Logistic(row_no,model_name):
   
    df = pd.read_csv('input.csv')
    row=df.iloc[row_no,1:-1]
    row=row.astype('int64')
    print(row)
    a=input("Enter the link of the video: ")
    if(model_name=="Logistic"):
        df = pd.read_csv('Epileptic_Seizure_Recognition.csv')
        X = df.iloc[:,1:-1]
        y=df.iloc[:,-1:]
        def toBinary(x):
            if x != 1: return 0;
            else: return 1
    
        y = y['y'].apply(toBinary)
        y = pd.DataFrame(data=y)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=42)

        model = LogisticRegression()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        # print the pridction
        # print(y_pred)
        # reshape the row to be able to predict it
        row = np.array(row).reshape(1, -1)
        # print the new shape
        print(row.shape)
        # predict the row
        y_pred = model.predict(row)
        # print the prediction
        print(y_pred)
        a=input("Enter the link of the video: ")

Logistic(3,"Logistic")