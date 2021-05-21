import pytest
import numpy as np
import pandas as pd
from sklearn import linear_model
import pred

gaia = pd.read_csv('../data/confirmed_transit.csv', comment='#')
gaia_new = gaia.loc[:,['pl_orbper','pl_bmassj','pl_radj','pl_orbsmax']]
gaia_new_nonah = gaia_new.dropna()
gaia_new_nonah = gaia_new_nonah.to_numpy()

def test_kfold_CV_type():
	cols = ['pl_orbper','pl_bmassj','pl_radj','pl_orbsmax']
	CV = pred.kfold_CV(gaia_new_nonah, cols, ['pl_radj'], 'pl_orbsmax',12)
	assert isinstance(CV, float) 

def test_kfold_CV_shape():
	cols = ['pl_orbper','pl_bmassj','pl_radj','pl_orbsmax']
	CV = pred.kfold_CV(gaia_new_nonah, cols, ['pl_bmassj'], 'pl_orbsmax',12)

	expected = 1
	assert len([CV]) == expected

    
