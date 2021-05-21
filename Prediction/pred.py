# import block
import numpy as np
import pandas as pd
from sklearn import linear_model

# K folds of cross validation
def kfold_CV(data, col_names, inputs, output, k):
    n_data = data.shape[0]
    test_errors=[]
    
    for i in range(k):
        #split data into train and test
        test_ind = list(np.arange(int(n_data*i/k),int(n_data*(i+1)/k)))
        test_data = data[test_ind,:]
        train_ind = list(set(range(n_data)).difference(set(test_ind)))
        train_data = data[train_ind,:]
        
        # map inputs and output col into integer
        input_index = []
        for elem in inputs:
            input_index.append(col_names.index(elem))
        output_index = col_names.index(output)
        
        #fit model to training dataset
        lm = linear_model.LinearRegression()
        mod = lm.fit(train_data[:,input_index],train_data[:,output_index])
        
        #compute the testing error and add it to the list of testing errors
        test_preds = mod.predict(test_data[:,input_index])
        test_error = compute_mse(test_preds,test_data[:,output_index])
        test_errors.append(test_error)
    
    # Compute tehe cross-val error
    cross_val_error = np.mean(test_errors)
    return cross_val_error.astype(float)

def compute_mse(truth_vec, predict_vec):
    return np.mean((truth_vec - predict_vec)**2)