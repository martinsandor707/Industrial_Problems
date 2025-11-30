import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def calculate_total_hop_length(df, col_type:str = 'p'):
    """

    :param df:
    :param type: single character, either "e" for effective or "p" for planned duration.
    :return:
    """
    col_names = ['i1','i2','i3','o']
    new_rows = []
    for index, row in df.iterrows():

        new_row = {'nr': np.int64(row['nr'])}
        for col in col_names:
            time = 0
            col_names_to_add =['rcs', 'dep_1', 'rcf_1', 'dep_2', 'rcf_2', 'dep_3', 'rcf_3', 'dlv']
            for i in range(len(col_names_to_add)):
                time+=row[f"{col}_{col_names_to_add[i]}_{col_type}"] if row[f"{col}_{col_names_to_add[i]}_{col_type}"] != '?' else 0

            new_row[col] = time
        new_rows.append(new_row)

    output_df = pd.DataFrame(new_rows)
    output_df = output_df.set_index('nr')

    return output_df


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


#TODO
def predict_the_delay(data,obs,type):
    datasimp = create_data(data,type)
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

    trainemb = trainset[features2].replace(airp_to_int)
    testemb = testset[features2].replace(airp_to_int)

# def fitMLPC(data1, data2, obs, filepath):
#     # data1: time values
#     # data2: airport id.s.
#
#     scaler = StandardScaler()
#     data1  = scaler.fit_transform(data1)
#
#     train1, val1, train2, val2, ytrain, yval = (
#         train_test_split(data1, data2, obs, test_size=0.2)
#     )
#
#     airp1 = Input(shape=(8,))
#     in1 = Input(shape=(8,))
#     in2 = Input(shape=(8,))
#
#     emb1 = Embedding(input_dim=238, output_dim=5)(airp1)
#     emb1 = Flatten()(emb1)
#     x = Concatenate()([in1, in2, emb1])
#     x = Dense(25, activation='elu')(x)
#     out = Dense(1, activation='sigmoid')(x)
#
#     model = Model(inputs=[in1, in2, airp1], outputs=out)
#     model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
