# Linear Model: Make Predictions for Missing Orbit Semi-Major Axis
I use linear model to predict missing values in the dataset in these coding exercise.

## Motivation 
The dataset has many missing value for some physical parameters, such as eccentricity and orbit semi-major axis. Over 60% of the dataset have values for all entries. I chose to predict the missing values based on the model generated from the remaining of known dataset.

## Dataset
The data is from the NASA Exoplanet Archive: https://exoplanetarchive.ipac.caltech.edu/index.html. I extract the exoplanets that are found by Transit Method into `../Data/confirm_transit.csv`. 

## Method
1. First I separate the observations with no missing value group and have-missing-value group. We will use the former dataframe to generate a prediction model.
2. The next step is to determine the shape of the "true" relationship for Orbit Semi-Axis. The inputs are `pl_orbper`,`pl_bmassj`,and `pl_radj`. We consider all possible combinations.
3. Use K-Fold to find the smallest error. We found the combination of orbital period and radius of the planet gives us the best relationship, with error of 0.022166.
4. Run the full model using these two parameters, applying on the dataset that is missing orbit semi-major axis information. 

## Results:
We now have a prediction table for all the missing  `pl_orbsmax` values:

                    |    | pl_orbper | pl_bmassj | pl_radj | pl_orbsmax |
                    |---:|----------:|----------:|--------:|-----------:|
                    |  2 | 41.685500 |   0.07000 |   0.230 |   0.149298 |
                    |  7 |  1.508956 |   1.03000 |   1.490 |   0.044130 |
                    | 38 |  1.742997 |   3.31000 |   1.414 |   0.047238 |
                    | 66 |  4.256800 |  21.66000 |   1.010 |   0.065633 |
                    | 81 |  0.854000 |   0.01447 |   0.141 |   0.092180 |

## Reference
https://exoplanets.nasa.gov/what-is-an-exoplanet/overview/ \
https://solarsystem.nasa.gov/planet-compare/ \
https://stackoverflow.com/questions/64152213/fill-pandas-column-nans-with-numpy-array-values \
https://stackoverflow.com/questions/37313691/how-to-remove-a-pandas-dataframe-from-another-dataframe
