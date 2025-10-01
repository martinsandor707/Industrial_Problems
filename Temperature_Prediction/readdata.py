import numpy as np
import matplotlib.pyplot as plt
from helper import read_stdata

datast1 = read_stdata("t2_station1.pickle")
datast2 = read_stdata("t2_station2.pickle")
datast3 = read_stdata("t2_station3.pickle")
datast4 = read_stdata("t2_station4.pickle")


# plot the data for a given day
# we have forecasts/observations for the next 5 days with 6h time step (i.e. 20 steps)
# 4 subplots (4 stations)
fig, ax =plt.subplots(2,2)
# the Xticks are the lead times (6h, 12h, .... , 120h)
a=np.arange(6,121,step=6)
plt.setp(ax, xticks=np.arange(6,121,step=20))
# the stations
statnames = ['Bregenz','Feldkirch','Warth','Galtuer']
# the given day
day='2018-05-01'
i=0
for df in [datast1, datast2, datast3, datast4]:
    d_arr = df[df['day']==day]
    obs = d_arr['obs'] #observation
    h = d_arr['hres']  # high resolution
    d1 = d_arr['0']  # control term
    d2 = np.mean(d_arr.iloc[:,4:],axis=1) # mean of the 51-ensemble

    ax[i//2,i%2].plot(a,obs,ls='--',c='r')
    ax[i//2,i%2].plot(a,d2,c='b')
    ax[i//2,i%2].plot(a,d1,c='g')
    ax[i//2,i%2].plot(a,h,c='k')
    ax[i//2,i%2].set_title(statnames[i])
    i = i+1
fig.suptitle(day)
plt.show()

