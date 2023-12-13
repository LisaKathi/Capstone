import streamlit as st 

#from functions import netz_heatmap, netz_heatmap_weekly


import datetime as dt
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


from functions import netz_heatmap, netz_heatmap_weekly


# Streamlit app
st.title('Charge Station Heatmap')

# Default charge station ID
default_charge_station_id = '4940A04D-62A0-ADF5-FE81-B67DD755CECC'

# Get charge station ID from user input
charge_station_id = st.text_input('Enter Charge Station ID:', value=default_charge_station_id)


# Display the heatmap
netz_heatmap(charge_station_id)
netz_heatmap_weekly(charge_station_id)
