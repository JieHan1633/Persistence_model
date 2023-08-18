import pandas as pd
import numpy as np
from metrics.probabilistic import sharpness, continuous_ranked_probability_score
import matplotlib.pyplot as plt
import csv
import os

def persistence_model(data, startT, endT, percentiles, past_days):
    '''
    generate probabilistic forecasts at percentiles during time [startT, endT] 
    according to past "past_days" 

    Parameters
    ----------
    data : dataframe
        includes "NL" and "TimeStamp" columns 
    startT : pd.Timestamp
        starting time of the forecasting.
    endT : pd.Timestamp
        ending time of the forecasting
    percentiles : (n, p). 
        a list with n samples and p percentiles. n is the length of timestamps 
        between startT and endT. percentiles should between 0-100.
    past_days : int.
        Prob distribution based on past_days same time NL values.

    Returns
    -------
    forecasts: (n, p) 
        np.array n is the number of forecasting time steps. p is the number of percentiles.
    observations: (n,) 
        np.array n is the number of observation time steps.

    '''
    forecasts = []
    observations = []
    data['TimeStamp'] = pd.to_datetime(data['TimeStamp'])
    idst = data[data['TimeStamp']==startT].index
    ided = data[data['TimeStamp']==endT].index
    assert (len(idst)>0 and len(ided)>0), "start time or end time does not exist"
    idst = idst[0]
    ided = ided[0]
    for i in range(idst,ided+1):
        # create 30 days before the t
        t = data['TimeStamp'][i] 
        print('observation at time: ',t)
        observations.append(data['NL'][i])
        index = []
        for day in range(1,past_days+1):
            tmp = t-pd.Timedelta(day,'d')
            idx = data[data['TimeStamp']==tmp].index
            if len(idx)>0:
                index.append(idx[0])
        time_of_days = data['NL'][index]
        forecasts.append(np.nanpercentile(time_of_days,percentiles))
    return np.array(forecasts), np.array(observations)

def pinball_loss(observations,forecasts):
    ''' 
    Parameters
    ----------
    observations : (n,) 
        np.array n is the number of observation time steps.
    forecasts : (n, p) 
        np.array n is the number of forecasting time steps. p is the number of percentiles.

    Returns
    -------
    pinball_mean : float 
    pinball_sum : float  
    '''
    I = np.where(observations[:,None]<forecasts, 1, 0) 
    tau = np.reshape(np.arange(0,1.1,0.1),(1,11))
    pinball_sum = np.sum(tau * (1 - I) * (observations[:,None]-forecasts) + 
                         (tau - 1) * I * (observations[:, None] - forecasts))
    pinball_mean =  np.mean(tau * (1 - I) * (observations[:, None]-forecasts) + 
                            (tau - 1) * I * (observations[:, None] - forecasts)) 
    return pinball_mean, pinball_sum

def meassurements(forecasts, observations, fx_prob): 
    ''' 
    Parameters
    ----------
    forecasts : (n,p)
        np.array, n forecasts at p different percentiles. p is the number of percentiles.
    observations : (n,)
        np.array, n observations.
    fx_prob : (n,p)
        n is the number of samples. p are the percentiles [0,10,20,...100]

    Returns
    -------
    crps : float
        crps value.
    rel : list with shape (p) 
        reliability.
    sh : list with shape (p-1)
        sharpness, with p-1 intervals.

    '''
    crps = continuous_ranked_probability_score(observations, forecasts, fx_prob)
    rel=[]
    sh = []
    p = fx_prob.shape[1]
    n_intervals = int(np.floor(p/2))
    n = len(observations)
    for i in range(n_intervals):
        upper = forecasts[:,n_intervals+i]
        lower = forecasts[:,n_intervals-i]
        sh.append(sharpness(lower,upper))
        rel.append(np.sum(np.where((observations<upper)& (observations>=lower), 1, 0))/n)
    pinball,_ = pinball_loss(observations, forecasts)
    return crps, rel, sh, pinball

if __name__=='__main__': 
    time_zone = {'city':['Amity','Donalsonville','San_Antonio','Waianae'],
                 'start_time':[8,5,6,11],'latitude':[45.114559,31.044241,29.424122,21.446911],
                           'longtitude':[-123.204903,-84.879128,-98.493629,-158.188736]}
    files = os.listdir(os.getcwd()+'/data')
    for i in range(4):
        city = time_zone['city'][i]
        fn = [file for file in files if city in file] 
        data1 = pd.read_csv('data/'+fn[0], skiprows=2)
        data2 = pd.read_csv('data/'+fn[1], skiprows=2) 
        data = pd.concat([data1,data2],ignore_index=True)
        data = data.rename(columns = {'timestamp':'TimeStamp','value':'NL'})
        data['TimeStamp'] = pd.to_datetime(data['TimeStamp'])
        data = data.sort_values(by='TimeStamp')
        data = data.reset_index(drop=True)
        startT = pd.Timestamp(year=2023,month=5,day=1,hour=time_zone['start_time'][i],tz='UTC')
        if city=='San_Antonio':
            endT = pd.Timestamp(year=2023,month=5,day=30,hour=time_zone['start_time'][i]-1,tz='UTC')
        else:
            endT = pd.Timestamp(year=2023,month=5,day=31,hour=time_zone['start_time'][i]-1,tz='UTC')
        percentiles = list(range(0,110,10)) # percentiles [0,10,20,...100]
        past_days = 30
        
        forecasts, observations = persistence_model(data, startT, endT, percentiles, past_days)
        # fx_prob = np.array([percentiles]*len(observations))
        # crps,rel,sh,pinball = meassurements(forecasts, observations, fx_prob)
        
        df = pd.DataFrame(forecasts,columns = ['P0','P1','P2','P3','P4','P5','P6','P7',\
                                                'P8','P9','P10'])
        df['targets'] = observations
        time_col = (data[(data['TimeStamp']>=startT)&(data['TimeStamp']<=endT)])['TimeStamp']
        df = pd.concat([time_col.reset_index(drop=True),df],axis=1)
        df.to_csv(f"data/{time_zone['city'][i]}_persistence_model_results.csv",index=False)
        print(f"finished predictions on {time_zone['city'][i]}")
        
    # data1 = pd.read_excel("data/HECOData_Clean_2017_with_zenith.xlsx")
    # data2 = pd.read_excel("data/HECOData_Clean_2018_with_zenith.xlsx")
    # NL_max = data1['NL'].max()
    # data2['NL'] /= NL_max
    # data1['NL'] /= NL_max
    # data = pd.concat([data1,data2],ignore_index=True) 
    # startT = pd.Timestamp(year=2018,month=6,day=18,hour=11)
    # endT = pd.Timestamp(year=2018,month=7,day=19,hour=10)
    # percentiles = list(range(0,110,10)) # percentiles [0,10,20,...100]
    # past_days = 30 
    
    # forecasts, observations = persistence_model(data, startT, endT, percentiles, past_days)
    
    # fx_prob = np.array([percentiles]*len(observations))
    # # fx_prob = np.array([percentiles])
    # crps,rel,sh,pinball = meassurements(forecasts, observations, fx_prob)
    
    # # # save predictions
    # df = pd.DataFrame(forecasts,columns = ['P0','P1','P2','P3','P4','P5','P6','P7',\
    #                                         'P8','P9','P10'])
    # df['targets'] = observations
    # time_col = (data[data['TimeStamp']>=startT])['TimeStamp']
    # df = pd.concat([time_col.reset_index(drop=True),df],axis=1)
    # df.to_csv('data/normalized_persistence_0618_0719.csv',index=False)
    # df1 = pd.DataFrame({'CRPS':crps,'reliability':rel,'sharpness':sh,'pinball':pinball})
    # df1.to_csv('data/measurements_of_persistence_0618_0719.csv')
    # df[['P0','P1','P2','P3','P4','P5','P6','P7','P8','P9','P10','targets']] *=NL_max  
    # df.to_csv('data/persistence_results.csv',index=False)
    
    # # plot sharpness
    # fig, ax = plt.subplots()
    # ax.plot(np.linspace(0.2,1,5),sh,'-*',linewidth=2.0,label = 'Persistence model')
    # plt.xlabel("Prediction intervals")
    # plt.ylabel("Interval width (KW)")
    # plt.legend()
    # plt.show()
    
    # # plot reliability
    # fig, ax = plt.subplots()
    # ax.plot(np.linspace(0.2,1,5),rel,'-*',linewidth=2.0,label='Persistence model')
    # ax.plot(np.linspace(0,1,10),np.linspace(0,1,10),'black',linewidth=2.0,label = 'Ideal')
    # plt.xlabel("Norminal probability")
    # plt.ylabel("Observaed probability")
    # plt.legend()
    # plt.show()
    
    # # plot predictions
    # fig, ax = plt.subplots()  
    # ax.plot(np.linspace(1, 49,48),observations[0:48], 'b-*',linewidth=2.0, label='targets')
    # for i in range(11):
    #     ax.plot(np.linspace(1, 49,48),forecasts[0:48,i], linewidth=1.0,label = 'P'+str(i))
    # ax.legend()
    # plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    # plt.xlabel("Hours")
    # plt.ylabel("Normalized Net Load")
    # plt.show()
