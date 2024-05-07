import logging


logger=logging.getLogger(__name__)
import numpy as np
try:
    import cupy as cp
except:
    logger.warning("cupy not installed")

import joblib
from tqdm.auto import tqdm
from zeitgeber.rle import encode

def one_pass_filter_all(data, n_jobs=1):
    if isinstance(data, np.ndarray):
        data=[data[:, i].tolist() for i in range(data.shape[1])]
        nx=np
    elif isinstance(data, cp.ndarray):
        data=cp.asnumpy(data)
        data=[data[:, i].tolist() for i in range(data.shape[1])]
        nx=cp
    else:
        nx=np
    
    if n_jobs>1:
        rounds=int(np.ceil(len(data)/n_jobs))
        n_jobs=len(data)//rounds

    assert len(data)>1

    out = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(
            one_pass_filter_1d
        )(
            data[i]
        )
        for i in range(len(data))
    )

    logger.debug("Stacking %s features along axis=1", len(out))
    data=np.stack(out, axis=1)
    logger.debug("Done")
    if nx is cp:
        logger.debug("Sending to GPU")
        data=cp.array(data)
        logger.debug("Done")
    return data



def one_pass_filter_1d(data):
    """
    Overwrite stationary data which is surrounded by two bouts of stationary data where the coordinate is the same
    """
    if isinstance(data, np.ndarray):
        nx=np
        data=data.tolist()
    elif isinstance(data, cp.ndarray):
        nx=cp
        data=cp.asnumpy(data).tolist()
    else:
        nx=np

    data=np.round(data).tolist()

    encoding=encode(data)
    filtered_encoding=[encoding[0]]
    
    for pos in tqdm(range(1, len(encoding))):
        length=encoding[pos][1]
        val=encoding[pos][0]
    
        if (pos==len(encoding)-1):
            filtered_encoding.append((val, length))
            # end of time series
            break
            
        if filtered_encoding[pos-1][0]==encoding[pos+1][0]:
            val=encoding[pos-1][0]
        
        filtered_encoding.append((val, length))

    filtered_data=[]
    for val, length in filtered_encoding:
        filtered_data.extend([val,]*length)
        
    filtered_data=nx.array(filtered_data)
    return filtered_data



def one_pass_filter_1d_2x(data):
    """
    Overwrite stationary data which is surrounded by two bouts of stationary data where the coordinate is the same
    """
    encoding=encode(data)
    filtered_encoding=[encoding[0]]
    
    for pos in tqdm(range(1, len(encoding))):
        length=encoding[pos][1]
        val=encoding[pos][0]
    
        if (pos==len(encoding)-1):
            filtered_encoding.append((val, length))
            # end of time series
            break
            
        if filtered_encoding[pos-1][0]==encoding[pos+1][0]:
            val=encoding[pos-1][0]
        elif (pos<len(encoding)-2) and filtered_encoding[pos-1][0]==encoding[pos+2][0]:
            val=encoding[pos-1][0]
        
        filtered_encoding.append((val, length))

    filtered_data=[]
    for val, length in filtered_encoding:
        filtered_data.extend([val,]*length)
        
    filtered_data=np.array(filtered_data)
    return filtered_data
