import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Concatenate, Flatten, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn. metrics import roc_curve, accuracy_score, roc_auc_score, confusion_matrix
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.utils import resample


####################################################################################
# data
# it is the modified dataset (there are 0s instead of the missing values)
data = pd.read_excel('c2k_dataM.xlsx')
cols = data.columns
####################################################################################
# column names for the airports
airports_all_coln = ['i1_dep_1_place','i1_rcf_1_place','i1_dep_2_place', 'i1_rcf_2_place',
                     'i1_dep_3_place','i1_rcf_3_place',
                 'i2_dep_1_place','i2_rcf_1_place','i2_dep_2_place','i2_rcf_2_place',
                     'i2_dep_3_place','i2_rcf_3_place',
                 'i3_dep_1_place','i3_rcf_1_place','i3_dep_2_place','i3_rcf_2_place',
                     'i3_dep_3_place','i3_rcf_3_place',
                 'o_dep_1_place', 'o_rcf_1_place','o_dep_2_place','o_rcf_2_place',
                     'o_dep_3_place','o_rcf_3_place']
airports_all = data[airports_all_coln]

airports = data[['i1_dep_1_place','i1_rcf_1_place','i1_rcf_2_place','i1_rcf_3_place',
                 'i2_dep_1_place','i2_rcf_1_place','i2_rcf_2_place','i2_rcf_3_place',
                 'i3_dep_1_place','i3_rcf_1_place','i3_rcf_2_place','i3_rcf_3_place',
                 'o_rcf_1_place','o_rcf_2_place','o_rcf_3_place']]

# list of airport ids
ap_id = set(np.array(airports_all).flat)
print(ap_id)
print('The number of the airports: ', len(ap_id))
#############################################################################################


#count the airport frequencies
dtype=[('idx', int),('N', int)]

values = []
for ap in ap_id:
    no_ap = np.where(np.array(airports_all) == ap,1,0)
    values.append( (ap, np.sum(no_ap)))

# sort the airports based on the frequency
ap_arr = np.array(values, dtype=dtype)
ar2 = np.sort(ap_arr, order='N')
print(ar2)
csN = np.cumsum(ar2['N'])


# how many different fr.s. do we have?
nf = max(list(set([h[1] for h in ar2])))
# replace the airport id to the frequency
#data[airports_all_coln] = data[airports_all_coln].replace(to_replace=dict(ar2))
#print(data[airports_all_coln])
airp_to_int = dict((aidx,n) for n, aidx in enumerate(ar2['idx']))
print(airp_to_int)
##############################################################################################
# calculate the length of the legs
# type: #list of 'p' (planned) and 'e' (actual) chars
def cal_leg_length(data,type):
    legs = ['i1', 'i2', 'i3', 'o']
    df = pd.DataFrame()
    df['nr'] = data['nr']
    service = ['rcs','dep_1','rcf_1','dep_2','rcf_2','dep_3','rcf_3','dlv']
    for leg in legs:
        col=[]
        if leg == 'o':
            k = 8
        else:
            k = 0
        for s in service:
            col.append(leg+'_'+s+'_'+type[k])
            k = k + 1
        df[leg] = np.sum(data[col],axis=1)
    return df

##############################################################################################
# find the class labels
###############################################################################################
# planned length of the legs
df_legs_p = cal_leg_length(data,['p']*16)
# actual length of the legs
df_legs_e = cal_leg_length(data,['e']*16)

# the end-to-end process
delay = df_legs_e.iloc[:,1:4].max(axis=1) + df_legs_e['o'] - df_legs_p.iloc[:,1:4].max(axis=1) - df_legs_p['o']
obs = pd.DataFrame({'nr': data['nr'], 'delay': delay >0})
obs.delay = obs.delay.replace({True: 1, False: 0})
#print(obs)
print('number of delays: ', np.sum(obs == True))
#################################################################################################
# ordering of the inbound legs
#################################################################################################
# the function returns the ordered times for the inbound legs
def leg_order_t(data):
    d2 = (data.set_index('nr')[['i1', 'i2', 'i3']].apply(lambda x:
        x.sort_values(ascending=False).tolist(), axis=1, result_type='expand').reset_index())
    return d2

# the function returns the ordered names of the inbound legs
def leg_order(data):
    d2 = (data.set_index('nr')[['i1', 'i2', 'i3']].apply(lambda x:
        x.sort_values(ascending=False).index.tolist(), axis=1, result_type='expand').reset_index())
    return d2


# selects the columns of data1 corresponding to the longest inbound leg, where data2 consists of the
# ordered indices
def longest_legs(data1,data2,type):
    N = data1.shape[0]
    a = ['_rcs_'+type[0], '_dep_1_'+type[1], '_dep_1_place','_rcf_1_'+type[2], '_rcf_1_place',
       '_dep_2_'+type[3], '_rcf_2_'+type[4], '_rcf_2_place', '_dep_3_'+type[5],
        '_rcf_3_'+type[6], '_rcf_3_place','_dlv_'+type[7]]
    newd = np.zeros((N,12))
    for k in range(N):
        an = [data2.iloc[k,1] + s for s in a]
        newd[k,:12] =  data1[an].iloc[k,:].values
    return newd

#########################################################################################################
# data corresponding to the longest inbound leg + the outbound leg
def create_data(data,type):
    #calculate the length of legs
    df_legs = cal_leg_length(data,type)
    # order the indices according to the time
    legord = leg_order(df_legs)
    # get the data corresponding to the longest inbound leg
    newd = longest_legs(data,legord,type)
    # collect the data corresponding to the longest inbound leg and to the outbound leg
    a_in = ['_rcs', '_dep_1', '_dep_1_place','_rcf_1', '_rcf_1_place',
       '_dep_2', '_rcf_2', '_rcf_2_place', '_dep_3',
        '_rcf_3', '_rcf_3_place','_dlv']
    a_out = ['o_rcs_'+type[8], 'o_dep_1_'+type[9], 'o_dep_1_place','o_rcf_1_'+type[10], 'o_rcf_1_place',
       'o_dep_2_'+type[11], 'o_rcf_2_'+type[12], 'o_rcf_2_place', 'o_dep_3_'+type[13],
        'o_rcf_3_'+type[14], 'o_rcf_3_place','o_dlv_'+type[15]]
    new_df = pd.DataFrame(newd,columns=a_in)
    new_df.insert(0,'nr',data['nr'])
    new_df.set_index('nr',inplace=True)
    out_df = data[['nr']+a_out]
    out_df.set_index('nr',inplace=True)
    datasimp = pd.concat([new_df,out_df], axis=1)
    return datasimp
#####################################################################################################
# learning rate scheduler
def scheduler(epoch, lr):
    if epoch >5:
        lr = 0.001
        return lr
    else:
        return lr
#####################################################################################################

def fitMLPC(data1,data2,obs,filepath, bagging=False):
    # data1: time values
    # data2: airport id.s.
    scaler = StandardScaler()
    data1 = scaler.fit_transform(data1)

    # input for the airports
    airp1 = Input(shape=(8,))
    # input for the inbound leg times
    in1 = Input(shape=(8,))
    # input for the outbound leg times
    in2 = Input(shape=(8,))
    #emb1 = Embedding(62275,6)(airp1)
    emb1 = Embedding(238,5)(airp1)
    emb1 = Flatten()(emb1)
    x = Concatenate()([in1,in2,emb1])
    x = Dense(20,activation='elu')(x)
    out = Dense(1,activation='sigmoid')(x)
    model = Model(inputs=[in1,in2,airp1], outputs=out)
    model.compile(optimizer=Adam(0.001),loss='BinaryCrossentropy')

    if bagging == False:
        train1, val1, train2, val2, ytrain, yval = (
            train_test_split(data1,data2,obs,test_size=0.2))
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                     save_best_only=True, mode='min')
        es = EarlyStopping(monitor='val_loss', verbose=1, patience=10)
        #lrsch = LearningRateScheduler(scheduler)
        #cl={0:1.0,1:2.0}
        history = model.fit([train1[:,:8],train1[:,8:16],train2],
                            ytrain, validation_data=([val1[:,:8],val1[:,8:16],val2],
                                                     yval),epochs=1500,batch_size=512,
                             callbacks=[es,checkpoint])
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
    else:
        history = model.fit([data1[:,:8],data1[:,8:16],data2],
                            obs,epochs=200,batch_size=512)
        plt.plot(history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()
    return model, scaler.mean_, scaler.scale_
##################################################################################################################


def predict_the_delay(data,obs,type,bagging=False):
    print("################################In predict_the_delay function")
    print(data.columns)
    print(data)
    datasimp = create_data(data,type)

    return
    trainset = datasimp.iloc[:2627,:]
    testset = datasimp.iloc[2627:,:]
    obs2 = obs['delay']
    trainobs = obs2[:2627]
    testobs = obs2[2627:]

    features1 = ['_rcs', '_dep_1', '_rcf_1', '_dep_2', '_rcf_2',  '_dep_3',
            '_rcf_3', '_dlv', 'o_rcs_'+type[8], 'o_dep_1_'+type[9],'o_rcf_1_'+type[10],
           'o_dep_2_'+type[11], 'o_rcf_2_'+type[12], 'o_dep_3_'+type[13], 'o_rcf_3_'+type[14], 'o_dlv_'+type[15]]
    features2 = ['_dep_1_place','_rcf_1_place','_rcf_2_place','_rcf_3_place',
                 'o_dep_1_place','o_rcf_1_place', 'o_rcf_2_place','o_rcf_3_place']

    testemb = testset[features2].replace(airp_to_int)
    trainemb = trainset[features2].replace(airp_to_int)

    if bagging == True:
        accvals2=[]
        acc1vals2=[]
        pred_all = []
        for k in range(10):
            trainsetB, trainobsB = resample(trainset, trainobs, replace=True, n_samples=2200)
            trainemb = trainsetB[features2].replace(airp_to_int)
            model, m, s = fitMLPC(trainsetB[features1],trainemb,trainobsB, "airweights.keras",bagging=True)
            testA = (testset[features1] - m) / s
            predy = model.predict([testA.iloc[:,:8],testA.iloc[:,8:16],testemb])
            pred_all.append(predy)
            acc = keras.metrics.BinaryAccuracy()
            acc.update_state(testobs,predy)
            accvals2.append(tf.keras.backend.get_value(acc.result()))
            acc1 = keras.metrics.AUC()
            acc1.update_state(testobs,predy)
            acc1vals2.append(tf.keras.backend.get_value(acc1.result()))
            print(accvals2)
            print(acc1vals2)
        pred_all = np.array(pred_all)
        newpred = np.mean(pred_all,axis=0)
        acc = keras.metrics.BinaryAccuracy()
        acc.update_state(testobs,newpred)
        acc1 = keras.metrics.AUC()
        acc1.update_state(testobs,newpred)

    else:
        model, m, s = fitMLPC(trainset[features1],trainemb,trainobs, "airweights.keras",bagging=False)
        #model.load_weights("airweights.keras")
        testA = (testset[features1] - m) / s
        predy = model.predict([testA.iloc[:,:8],testA.iloc[:,8:16],testemb])
        acc = keras.metrics.BinaryAccuracy()
        acc.update_state(testobs,predy)
        print('accuracy: ', acc.result())
        acc1 = keras.metrics.AUC()
        acc1.update_state(testobs,predy)
        print(confusion_matrix(testobs,np.round(predy)))
        print('AUC: ',acc1.result())
    return acc.result(), acc1.result()

def update_the_pred():
    accvals=[]
    acc1vals=[]
    for k in range(16):
        acc,acc1 = predict_the_delay(data,obs,['e']*k+['p']*(16-k))
        accvals.append(tf.keras.backend.get_value(acc))
        acc1vals.append(tf.keras.backend.get_value(acc1))
        print('accuracy: ', tf.keras.backend.get_value(acc))
        print('AUC: ',tf.keras.backend.get_value(acc1))
    print(accvals)
    print(acc1vals)
    plt.plot(np.arange(0,16),accvals,c="b")
    plt.plot(np.arange(0,16),acc1vals,c="r")
    plt.show()


#update_the_pred()

def make_bagging():
    acc,acc1 = predict_the_delay(data,obs,['p']*16, bagging=True)
    print(tf.keras.backend.get_value(acc))
    print(tf.keras.backend.get_value(acc1))

data = pd.read_excel('c2k_dataM.xlsx')

make_bagging()


def draw_roc(ytrue,ypred):
    y = np.argmax(ytrue, axis=-1)
    fpr, tpr, thresholds = roc_curve(y, ypred[:,1])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.show()


#draw_roc(testobs,predy)