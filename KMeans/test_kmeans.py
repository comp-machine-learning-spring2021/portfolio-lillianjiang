import pytest
import pandas as pd
import numpy as np
import kmeans

gaia = pd.read_csv('../data/confirmed_transit.csv', comment='#')
gaia_new = gaia.loc[:,['pl_orbper','pl_bmassj','pl_radj']]
gaia_np = gaia_new.to_numpy()

def test_my_kmeans_type():
	assert isinstance(kmeans.kMeans(gaia_np, 3, 200), tuple)

def test_my_kmeans_shape():
	expected = 2
	assert len(kmeans.kMeans(gaia_np, 3, 200)) == expected

def test_my_kmeans_center_num():
	expected = (2,3)
	centers_shape = kmeans.kMeans(gaia_np, 2, 200)[0].shape
	assert centers_shape == expected
	
def test_my_kmeans_labels():
	expected = 5
	label_max = np.max(kmeans.kMeans(gaia_np, 6, 100)[1])
	assert label_max == expected

def test_different_cols():
	expected = False
	centers=kmeans.kMeans(gaia_np, 6, 100)[0]
	comp_cols = sum(centers[:,0] == centers[:,1]) == 6
	assert comp_cols == expected

def test_looping_kmeans_type():
	assert isinstance(kmeans.looping_kmeans(gaia_np,
		list(range(1,5))), list)

def test_looping_kmeans_size():
	expected = 14
	assert len(kmeans.looping_kmeans(gaia_np,
		list(range(1,15)))) == expected

def test_looping_kmeans_goodness():
	out = kmeans.looping_kmeans(gaia_np,list(range(1,15)))
	assert (out[1:] <= out[:-1])

    
