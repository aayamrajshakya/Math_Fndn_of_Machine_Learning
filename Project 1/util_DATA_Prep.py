import numpy as np
import pandas as pd

def load_data(excelfile):
    df = pd.read_excel(excelfile)
    df.fillna(0,inplace=True) #replace nan(=empty spot) by 0
    DATA = df.values;
    header = df.columns.tolist()
    print('@@',excelfile)
    print('   DATA.dtype,DATA.shape =',DATA.dtype,DATA.shape)
    print('   header =',header)

    return DATA,header

