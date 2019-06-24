import pywt
from statsmodels.robust import mad
import numpy as np

def waveletSmooth( x, wavelet="haar", title=None):
    
    (ca, cd) = pywt.dwt(x, "haar")
    ca_sd = np.std(ca)
    cd_sd = np.std(cd)
    ca_thresh = ca_sd * (np.sqrt( 2*np.log( len( x ) ) ))
    cd_thresh = cd_sd * (np.sqrt( 2*np.log( len( x ) ) ))

    cat = pywt.threshold(ca, ca_thresh, mode="hard" , substitute = -(np.std(ca)))
    
    tx = pywt.idwt(cat, cdt, "haar")
    #print(tx)
    #log_tx = np.log(tx)*100
    
    return tx
    
