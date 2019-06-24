import pywt
from statsmodels.robust import mad
import numpy as np

def waveletSmooth( x, wavelet="haar", level=1, DecLvl=2, title=None):
    # calculate the wavelet coefficients
    coeff = pywt.wavedec( x, wavelet, mode="symmetric", level=DecLvl )
    # calculate a threshold
    #sigma = mad( coeff[-level] )
    #uthresh = sigma * np.sqrt( 2*np.log( len( x ) ) )
    #coeff[1:] = ( pywt.threshold( i, value=uthresh, mode="soft" ) for i in coeff[1:] )
    # reconstruct the signal using the thresholded coefficients
    #y = pywt.waverec( coeff, wavelet, mode="per" )
    #return y
    #train = []
    log_ret = []
    log_train_data = []
    
    (ca, cd) = pywt.dwt(x, "haar")
    ca_sd = np.std(ca)
    cd_sd = np.std(cd)
    for m in np.arange(0.0, 1.0, 0.1):
        for k in np.arange(0.0, 1.0, 0.1):
            ca_thresh = m*n*ca_sd * (np.sqrt( 2*np.log( len( x ) ) ))
            cd_thresh = m*n*cd_sd * (np.sqrt( 2*np.log( len( x ) ) ))

    #cat = pywt.threshold(ca, np.std(ca), mode="hard" , substitute = -(np.std(ca)))
    #cdt = pywt.threshold(cd, np.std(cd), mode="hard" , substitute = -(np.std(cd)))
    #cat = pywt.threshold(ca, uthresh, mode="hard" , substitute = -(np.std(ca)))
    #cdt = pywt.threshold(cd, uthresh, mode="hard" , substitute = -(np.std(cd)))
    #cat = pywt.threshold(ca, ca_thresh, mode="soft" , substitute = (-ca))
    #cdt = pywt.threshold(cd, ca_thresh, mode="soft" , substitute = (-cd))
    cat = pywt.threshold(ca, ca_thresh, mode="soft" )
    cdt = pywt.threshold(cd, ca_thresh, mode="soft" )
    tx = pywt.idwt(cat, cdt, "haar")
    #print(tx)
    log_tx = np.log(tx)*100
    #print('log return is ' , log)
    #macd = np.mean(x[5:]) - np.mean(x)
    # ma = np.mean(x)
    #sd = np.std(x)
    #log_ret = np.append(log_ret, log)
    #x_tech = np.append(macd*10, sd)
    #train = np.append(train, x_tech)
    #train_data.append(train)
    #log_train_data.append(log_ret)
    #trained = pd.DataFrame(train_data)
    #trained.to_csv("preprocessing/indicators.csv")
    #log_train = pd.DataFrame(log_train_data, index=None)
    return tx
