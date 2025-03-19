import numpy as np
import os,sys
from util_DATA_Prep import *

file_ELP = './dataFiles/data-ECommerce-Labor_Prod.xlsx'
file_TS  = './dataFiles/data-Total-Sale.xlsx'

#--------------------------------------------------
# Read Excel files
#--------------------------------------------------
DATA_ELP, header_ELP = load_data(file_ELP)
DATA_TS,  header_TS  = load_data(file_TS)

#--------------------------------------------------
# Combine and Sort
#  Combine the above for <DATA> and <header>
#  in the order ['NAICS', 'year', 'Total', 'E-commerce', 'Labor-Prod']
#  Sort: First, with 'NAICS code' and then with 'year'
#--------------------------------------------------

# Implement a function or two into "util_DATA_Prep.py" to complete

#--------------------------------------------------
# You can save the trimmed "DATA" to an Excel file:
# First, you should get combined <DATA> and <header>
#--------------------------------------------------

