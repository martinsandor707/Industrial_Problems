import numpy as np
import pandas as pd
from helper import read_stdata, read_stdata2, my_train_test_split, fitmdl
from sklearn.metrics import mean_squared_error

########################################################################################
# read the t2m station data
datast1 = read_stdata("t2_station1.pickle")
datast2 = read_stdata("t2_station2.pickle")
datast3 = read_stdata("t2_station3.pickle")
datast4 = read_stdata("t2_station4.pickle")

# convert the t2m data (use Celsius instead of Kelvin)
datast1.iloc[:,2:] = datast1.iloc[:,2:].apply(lambda x: x-273.15)
datast2.iloc[:,2:] = datast2.iloc[:,2:].apply(lambda x: x-273.15)
datast3.iloc[:,2:] = datast3.iloc[:,2:].apply(lambda x: x-273.15)
datast4.iloc[:,2:] = datast4.iloc[:,2:].apply(lambda x: x-273.15)
#######################################################################################

train_1, test_1 = my_train_test_split(datast1, '2018-01-01')
train_2, test_2= my_train_test_split(datast2, '2018-01-01')
train_3, test_3= my_train_test_split(datast3, '2018-01-01')
train_4, test_4= my_train_test_split(datast4, '2018-01-01')

#######################################################################################
# read the wind and tcc hres terms.
# wind1 columns: day, lt, u10, v10 (last two are the 10m wind components, from these we can
# calculate the wind speed)
wind1 = read_stdata2("wind10_hres_station1.pickle")
wind1['wspeed'] = np.sqrt(np.square(wind1['u10'])+np.square(wind1['v10']))

tcc1 = read_stdata2("tcc_hres_station1.pickle")
tcc1.columns = ['day','lt','tcc']

# split into train/test set
windtrain_1, windtest_1 = my_train_test_split(wind1, '2018-01-01')
tcctrain_1, tcctest_1 = my_train_test_split(tcc1, '2018-01-01')
#######################################################################################

#######################################################################################
# dataframes with different features
f = [str(i) for i in range(51)]
train_m1 = pd.DataFrame({'day': train_1['day'], 'lt': train_1['lt'], 'hres': train_1['hres'],
                         'mean': np.mean(train_1[f], axis=1 ), 'sd': np.std(train_1[f],axis=1 ),
                         'tcc': tcctrain_1['tcc'], 'wind': windtrain_1['wspeed'],
                         'obs': train_1['obs']})
test_m1 = pd.DataFrame({'day': test_1['day'], 'lt': test_1['lt'],'hres': test_1['hres'],
                         'mean': np.mean(test_1[f], axis=1 ), 'sd': np.std(test_1[f],axis=1 ),
                        'tcc': tcctest_1['tcc'], 'wind': windtest_1['wspeed'],
                        'obs': test_1['obs']})
#######################################################################################
#####################################
# define and train a network
####################################
# the features we want to use (make experiments with different feature sets, with different
# networks)
features = ['lt', 'hres', 'mean', 'sd', 'wind', 'tcc']
model, m, s = fitmdl(train_m1,features,"weights.keras")
# scale the test data using the same scaling parameters as in case of the training set
testA = (test_m1[features] - m) / s
model.load_weights("weights.keras")
predY1 = model.predict(testA)
dfres1 = pd.DataFrame({'day': test_1['day'], 'lt': test_1['lt'], 'pred': predY1[:,0]})
dfres1.to_excel("output_point1.xlsx")
errnew1 = mean_squared_error(test_1['obs'], predY1)
print('the MSE of the predicted values on the test set: '+str(errnew1))

