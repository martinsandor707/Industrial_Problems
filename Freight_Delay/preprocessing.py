import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import helper

# Load an Excel file

##### itt mar a 0-k legyenek a ? helyett (a data forderben van egy ilyen fajl is)
df = pd.read_excel("c2k_dataM.xlsx")

#### ezt a valtozot atneveztem, ezek csak a column nevek lesznek
airports_all_col = ['i1_dep_1_place','i1_rcf_1_place','i1_dep_2_place', 'i1_rcf_2_place',
                     'i1_dep_3_place','i1_rcf_3_place',
                 'i2_dep_1_place','i2_rcf_1_place','i2_dep_2_place','i2_rcf_2_place',
                     'i2_dep_3_place','i2_rcf_3_place',
                 'i3_dep_1_place','i3_rcf_1_place','i3_dep_2_place','i3_rcf_2_place',
                     'i3_dep_3_place','i3_rcf_3_place',
                 'o_dep_1_place', 'o_rcf_1_place','o_dep_2_place','o_rcf_2_place',
                     'o_dep_3_place','o_rcf_3_place']

# ez a megfelelp dataframe
airports_all = df[airports_all_col]

# itt akkor eleg ezt a dataframet hasznalni, de meg egyszer: itt mar kellenek a 0-k!!!
ap_id = set(np.array(airports_all).flat)

values = []

for ap in ap_id:
    no_ap = np.where(np.array(airports_all) == ap, 1, 0)
    # print(f"{ap} Airport ID counts: {no_ap}")
    values.append((ap, np.sum(no_ap)))


ap_arr = np.array(values)
print(f"{ap_arr=}")

# itt az ar2=... sor nem mukodik, ha nem irja be ezt a 2 sort:
dtype=[('idx', int),('N', int)]
ap_arr = np.array(values, dtype=dtype)
ar2= np.sort(ap_arr, order="N")
airp_to_int = dict((aidx, n) for n, aidx in enumerate(ar2['idx']))

# ez mar nem kell
#df = df.replace("?", np.int64(0))
lengths = helper.cal_leg_length(df,['e']+['p']*15)
print(f"{airp_to_int=}")
print(helper.leg_order(lengths))
print(helper.leg_order_t(lengths))

print("#"*20)
print(helper.create_data(df, ['e']+['p']*15 ))