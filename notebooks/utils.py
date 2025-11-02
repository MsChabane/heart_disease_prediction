from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score,f1_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import time
import pandas as pd 

def get_train_test(df:pd.DataFrame,target:str,scale=None,test_size:float=0.2,random_state:int=42):
    X_train,X_test,y_train,y_test=train_test_split(df.drop(target,axis=1),df[target],test_size=test_size,random_state=random_state)
    if scale :
        scale_mappping={
            'minmax':MinMaxScaler,'standard':StandardScaler
        }
        scaler=scale_mappping[scale]().fit(X_train)
        X_train=pd.DataFrame(scaler.transform(X_train),columns=scaler.get_feature_names_out())
        X_test=pd.DataFrame(scaler.transform(X_test),columns=scaler.get_feature_names_out())
        
    return (X_train,y_train),(X_test,y_test)
    

def evaluate_model(model,params:dict,train,test):
    start_train=time.time()
    model.fit(train[0],train[1])
    end_train=time.time()
    start_test=time.time()
    pred = model.predict(test[0])
    proba = model.predict_proba(test[0])[:,1]
    end_test=time.time()
    result ={
        "acc":accuracy_score(test[1],pred),
        'auc':roc_auc_score(test[1],proba),
        'f1':f1_score(test[1],pred),
        'train_time':1000*(end_train-start_train),
        'test_time':1000*(end_test-start_test)
    }
    result.update(params)
    return result
