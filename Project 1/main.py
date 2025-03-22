import pandas as pd
import numpy as np

file_TS  = './dataFiles/data-Total-Sale.xlsx'
file_ELP = './dataFiles/data-ECommerce-Labor_Prod.xlsx'

# Right now, the Total-Sale sheet is messy, with the 'year' as column headers.
# Function to convert the column headers into row data and group the new year rows with corresponding 'Total' values
def fix_file_TS():
    ts = pd.read_excel(file_TS)
    ts.fillna(0, inplace=True)
    ts = ts.melt(id_vars=["NAICS"], var_name='year', value_name="Total")
    # Had to assert the year values as integer, otherwise merging won't work
    ts['year']=ts['year'].astype(int)
    ts.sort_values(by=['NAICS', 'year'], ascending=[True, True], inplace=True, ignore_index=True)
    return ts

# Function to sort the existing ELP sheet based on NAICS code and year
def sort_file_ELP():
    elp = pd.read_excel(file_ELP)
    elp.fillna(0, inplace=True)
    elp.sort_values(by=['NAICS', 'year'], ascending=[True, True], inplace=True, ignore_index=True)
    return elp

# Running the custom functions
ts = fix_file_TS()
elp = sort_file_ELP()

f3 = pd.merge(ts, elp, how="right", on=["NAICS", "year"])
f3.index = np.arange(1, len(f3) + 1)
f3.to_excel("Trimmed-DATA.xlsx", index=True)


# REFERENCES:
# https://stackoverflow.com/questions/28654047/convert-columns-into-rows-with-pandas
# https://stackoverflow.com/questions/50649853/trying-to-merge-2-dataframes-but-get-valueerror/
