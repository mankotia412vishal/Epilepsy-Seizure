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
import pickle


def Logistic(row_no,model_name):
    print(type(row_no))
    print(row_no)
    # a=input("Enter the link of the video:1 ")

    # a=input("Enter the link of the video:1 ")
    df = pd.read_csv('input.csv')
    row=df.iloc[int(row_no)-1,1:-1]
    row=row.astype('int64')
    df = pd.read_csv('Epileptic_Seizure_Recognition.csv')
    X = df.iloc[:,1:-1]
    y=df.iloc[:,-1:]
    def toBinary(x):
        if x != 1: return 0;
        else: return 1

    y = y['y'].apply(toBinary)
    y = pd.DataFrame(data=y)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=42)

   
    # row

    # a=input("Enter the link of the video: ")
    if(model_name=="Logistic" or  model_name=="Svm" or model_name=="Knn" or model_name=="Lstm" or model_name=="Ann"):
         
        model = LogisticRegression()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        
        
        row = np.array(row).reshape(1, -1)
        
        print(row.shape)
        
        y_pred = model.predict(row)
        
        print(y_pred)
        print(type(y_pred))

         
        ans=int(y_pred[0])

        
        
        return ans
    elif(model_name=="Svm"):
        model = SVC()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        row = np.array(row).reshape(1, -1)
        
        print(row.shape)
        
        y_pred = model.predict(row)
        
        print(y_pred)
        print(type(y_pred))
         
        ans=int(y_pred[0])
        
        return ans
    elif model_name=="Knn":
        print(type(x_train))
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        model = KNeighborsClassifier()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(row)
        row = np.array(row).reshape(1, -1)
        
        print(row.shape)
       
        y_pred = model.predict(row)
        
        print(y_pred)
        print(type(y_pred))
         
        ans=int(y_pred[0])
        print(ans)
        # a=input("Enter the link of the video: ")


        return ans
    elif model_name=="Lstm":
        filename = 'lstm.sav'
        # now take predicts 
# load the model from disk
        loaded_model = pickle.load(open(filename, 'rb'))
        # take a single row as input and predict its ouput from input .csv file
        df = pd.read_csv('input.csv')
        # row=df.iloc[12,1:-1]
        # row=row.astype('int64')
        print(row)
        
        
        # Reshape the row to match the LSTM input shape
        row = np.array(row).reshape(1, 1, 178)
        # (1, 1, 178)
        # means
        # Print the new shape
        print("Reshaped Shape:", row.shape)

        # Predict the row
        y_pred = loaded_model.predict(row)

        #  Print the prediction
        # print("Prediction:", y_pred)
        # Print its class (0 or 1)
        print("Predicted class:", np.argmax(y_pred, axis=1))
        # a=input("Enter the link of the video: ")
        ans=int(np.argmax(y_pred, axis=1))
        return ans
    


        






    





        

         
        
         
        

