import numpy as np
import pandas as pd
import xgboost as xgb
import math
import pdb
import sklearn as sk
from cleandata import cleanData
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import OneHotEncoder
import time


start = time.time()

train = pd.read_csv('data/train.csv')
store = pd.read_csv('data/store.csv')
test = pd.read_csv('data/test.csv')

# Join train and test data with store data

train_data = train.join(store.set_index('Store'), on='Store')

# splitting train data into train and validation data

train, validation = train_test_split(train_data, test_size=0.2, random_state=42)

# Clean Data

train_clean = cleanData(train, 'train')

validation_clean = cleanData(validation,'test')


# Splitting features and label

def helper(row):
    if row['StateHoliday'] == 0.0:
        return '0'
    else:
        return row['StateHoliday']

X_train = train_clean.drop(['Sales'], axis=1)
y_train = train_clean['Sales']
X_train =X_train.reset_index().drop('index',axis=1)
X_train['StateHoliday'] = X_train.apply(helper, axis=1)

X_validation = validation_clean.drop(['Sales'], axis=1)
y_validation = validation_clean['Sales']
X_validation = X_validation.reset_index().drop('index',axis=1)
X_validation['StateHoliday'] = X_validation.apply(helper, axis=1)

# OnehotEncode train and validation data

def get_training_data(x_tr):
    # encode PromoInterval
    PromoInterval_enc = OneHotEncoder(handle_unknown='ignore')
    PromoInterval_encoding = PromoInterval_enc.fit_transform(x_tr['PromoInterval'].to_numpy().reshape(-1, 1)).toarray()
    ohe_PromoInterval = pd.DataFrame(PromoInterval_encoding, columns = PromoInterval_enc.get_feature_names())
    x_tr = pd.concat([x_tr, ohe_PromoInterval], axis=1).drop(['PromoInterval'], axis=1)
    
    # encode StateHoliday
    StateHoliday_enc = OneHotEncoder(handle_unknown='ignore')
    StateHoliday_encoding = StateHoliday_enc.fit_transform(x_tr['StateHoliday'].to_numpy().reshape(-1, 1)).toarray()
    ohe_StateHoliday = pd.DataFrame(StateHoliday_encoding, columns = StateHoliday_enc.get_feature_names())
    x_tr = pd.concat([x_tr, ohe_StateHoliday], axis=1).drop(['StateHoliday'], axis=1) 
    
    # encode StoreType
    StoreType_enc = OneHotEncoder(handle_unknown='ignore')
    StoreType_encoding = StoreType_enc.fit_transform(x_tr['StoreType'].to_numpy().reshape(-1, 1)).toarray()
    ohe_StoreType = pd.DataFrame(StoreType_encoding, columns = StoreType_enc.get_feature_names())
    x_tr = pd.concat([x_tr, ohe_StoreType], axis=1).drop(['StoreType'], axis=1)  
    
    # encode Assortment
    Assortment_enc = OneHotEncoder(handle_unknown='ignore')
    Assortment_encoding = Assortment_enc.fit_transform(x_tr['Assortment'].to_numpy().reshape(-1, 1)).toarray()
    ohe_Assortment = pd.DataFrame(Assortment_encoding, columns = Assortment_enc.get_feature_names())
    x_tr = pd.concat([x_tr, ohe_Assortment], axis=1).drop(['Assortment'], axis=1)         
    
    return x_tr, {'PromoInterval': PromoInterval_enc, 'StateHoliday': StateHoliday_enc, 'StoreType': StoreType_enc, 'Assortment': Assortment_enc}


def get_test_data(x_te, encoders):
    # encode PromoInterval
    PromoInterval_encoding = encoders['PromoInterval'].transform(x_te['PromoInterval'].to_numpy().reshape(-1, 1)).toarray()
    ohe_PromoInterval = pd.DataFrame(PromoInterval_encoding, columns = encoders['PromoInterval'].get_feature_names())
    x_te = pd.concat([x_te, ohe_PromoInterval], axis=1).drop(['PromoInterval'], axis=1)
    
    # encode StateHoliday
    StateHoliday_encoding = encoders['StateHoliday'].transform(x_te['StateHoliday'].to_numpy().reshape(-1, 1)).toarray()
    ohe_StateHoliday = pd.DataFrame(StateHoliday_encoding, columns = encoders['StateHoliday'].get_feature_names())
    x_te = pd.concat([x_te, ohe_StateHoliday], axis=1).drop(['StateHoliday'], axis=1)
    
    # encode StoreType
    StoreType_encoding = encoders['StoreType'].transform(x_te['StoreType'].to_numpy().reshape(-1, 1)).toarray()
    ohe_StoreType = pd.DataFrame(StoreType_encoding, columns = encoders['StoreType'].get_feature_names())
    x_te = pd.concat([x_te, ohe_StoreType], axis=1).drop(['StoreType'], axis=1)  
    
    # encode Assortment
    Assortment_encoding = encoders['Assortment'].transform(x_te['Assortment'].to_numpy().reshape(-1, 1)).toarray()
    ohe_Assortment = pd.DataFrame(Assortment_encoding, columns = encoders['Assortment'].get_feature_names())
    x_te = pd.concat([x_te, ohe_Assortment], axis=1).drop(['Assortment'], axis=1)      
    
    return x_te


x_tr, encoders = get_training_data(X_train)
x_te = get_test_data(X_validation, encoders)


# DataFrame to numpy array

X_train = x_tr.to_numpy()
y_train = y_train.to_numpy()
X_validation = x_te.to_numpy()
y_validation = y_validation.to_numpy()


# Metric as defined by the kaggle competition(given)

def metric(preds, actuals):
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])


# Model and train

def model(params, X_train, y_train, X_test, y_test):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    bst = xgb.train(params, dtrain)
    test = xgb.DMatrix(X_test)
    y_h_test = bst.predict(test)
    train = xgb.DMatrix(X_train)
    y_h_train = bst.predict(train)
    
    return (metric(y_h_test, y_test), metric(y_h_train, y_train), bst)

params = {
  'booster' : 'gbtree',
  'colsample_bynode': 0.8,
  'learning_rate': 0.4,
  'max_depth': 9,
  'min_child_weight': 1,
  'gamma': 0,  
  'num_parallel_tree': 25,
  'objective': 'reg:squarederror',
  'subsample': 0.8,
  'num_boost_round': 30,
   'eval_metric': 'rmse'}

eval_val, eval_train, model = model(params, X_train, y_train, X_validation, y_validation)
print('Validation result: ', eval_val)
print('Train result: ', eval_train)


# Save and load model

pickle.dump(model, open("pima.pickle.dat", "wb"))
model = pickle.load(open("pima.pickle.dat", "rb"))

# Predictions

def prediction(test, model):
    data = test.join(store.set_index('Store'), on='Store')
    test_clean = cleanData(data, 'test')
    test_clean['StateHoliday'] = test_clean.apply(helper, axis=1)
    test_clean = test_clean.reset_index().drop('index', axis=1)
    feature = test_clean.drop(['Sales'], axis=1)
    feature = get_test_data(feature, encoders)
    feature = feature.to_numpy()
    label = test_clean['Sales'].to_numpy()
    dpredict = xgb.DMatrix(feature)
    y_h_predict = model.predict(dpredict)
    
    
    return metric(y_h_predict, label)

pred = prediction(test, model)
print("Xgboost prediction on test data: {} in {} seconds".format(pred, time.time()-start))






