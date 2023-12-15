import streamlit as st 

#from functions import netz_heatmap, netz_heatmap_weekly


from datetime import datetime as dt
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import traceback


from functions import netz_heatmap, netz_heatmap_weekly, netz_heatmap_prediction, electricity_prod, pv_heatmap, pv_heatmap_weekly

# Streamlit app

st.set_page_config(page_title='Smart Charging Advisor', layout = "wide")

st.title('Smart Charging Advisor')

st.sidebar.image("./logo.jpg")

# user inputs --> mehrere PV-Anlagen auf unterschiedliche Direktionen erlauben? (2. Taste 'andere PV-Anlage hinzufügen')
    # auch möglich, default values zu übernehmen
flaeche = st.sidebar.number_input('Panelfläche in Quadratmeter', value=56)
panelleistung = st.sidebar.number_input('Panelleistung der PV-Anlage in kWp', value=7, help='Spezifisch zur PV-Anlage (Typischerweise: 1kWp pro 7 m²)') #button 'About' to add
ausrichtung = st.sidebar.slider('Ausrichtung des Gebäudes in Grad', min_value=-90, max_value=90, step=10, value= -20, help='Ost:-90, Süd:0, West:90')
dachneigung = st.sidebar.slider('Dachneigung in Grad', min_value=0, max_value=90, step=10, value=30, help='(0° - horizontal, 90° - vertikal)')
temp_coeff = st.sidebar.number_input('Temperaturkoeffizient der PV-Anlage in %', value=0.37, help='Spezifisch zur PV-Anlage (typischerweise: zw. 0.30 und 0.45)') #button 'About' to add
noct = st.sidebar.number_input('NOCT in °C', value=25, help='Spezifisch zur PV-Anlage (typischerweise: 25°C)') #button 'About' to add
charge_station_id = st.sidebar.text_input('Enter Charge Station ID', value='4940A04D-62A0-ADF5-FE81-B67DD755CECC')

# Default charge station ID
#default_charge_station_id = '4940A04D-62A0-ADF5-FE81-B67DD755CECC'



df_solarertrag = pd.read_csv("./df_solarertrag.csv")
df_strahlung_pro_stunde = pd.read_csv("./df_strahlung_pro_stunde.csv")

#PV-Anlage
sun_coeff = df_solarertrag.loc[(df_solarertrag['Ausrichtung']==ausrichtung) & (df_solarertrag['Dachneigung']==dachneigung)].reset_index(drop=True)
sun_coeff = float(sun_coeff["Solarertrag"])
globalstrahlung = df_strahlung_pro_stunde['Strahlung Rieden [kW/m²]']
temperatur = df_strahlung_pro_stunde['Temp. Rieden [°C]']

#stromproduktion = object_pv.electricity_prod(sun_coeff, globalstrahlung, temperatur)

stromproduktion =  electricity_prod(flaeche, panelleistung, temp_coeff, noct, sun_coeff, globalstrahlung, temperatur)


final_df_pv = df_strahlung_pro_stunde.copy()
final_df_pv['Stromproduktion'] = pd.Series(stromproduktion)
final_df_pv["Datum"] = pd.to_datetime(final_df_pv['Datum'])
start_date = dt.strptime('2023-11-15', '%Y-%m-%d').date()
end_date = dt.strptime('2023-11-28', '%Y-%m-%d').date()
heatmap_data_pv = final_df_pv.pivot_table(index='Stunde', columns='Datum', values='Stromproduktion').loc[:, start_date:end_date]

## weekly

heatmap_data_average = pd.read_csv("./heatmap_data_average.csv")
stromproduktion =  electricity_prod(flaeche, panelleistung, temp_coeff, noct, sun_coeff, globalstrahlung, temperatur)
heatmap_data_average['Stromproduktion'] = pd.Series(stromproduktion)
heatmap_data_average = heatmap_data_average

                
tab1, tab2, tab3 = st.tabs(["Aktueller Vergleich", "Wöchentlicher Vergleich", "Zukünftiger Vergleich"])

with tab1:
    col1, col2 = st.columns(2)
    try:
        with col1:
            st.header("Ihre PV-Anlage")
            st.markdown("### Stromproduktion der letzten Tage")
            pv_heatmap(heatmap_data_pv)

        with col2:
            st.header("Ihre Ladestation")
            st.markdown("### Stromnutzung der letzten Tage")
            netz_heatmap(charge_station_id)
    except Exception as e:
        st.error(f"An error occured {e}")
        st.error(traceback.format_exc())

with tab2:
    col1, col2 = st.columns(2)
    try:
        month = st.text_input('Monat (English)', value='November')
        with col1:
            st.header("Ihre PV-Anlage")
            st.markdown("### Durchschnittliche Stromproduktion")
            pv_heatmap_weekly(heatmap_data_average, month)

        with col2:
            st.header("Ihre Ladestation")
            st.markdown("### Durchschnittliche Stromnutzung")
            netz_heatmap_weekly(charge_station_id)

    except Exception as e:
        st.error(f"An error occured {e}")
        st.error(traceback.format_exc())

final_df_pv = df_strahlung_pro_stunde.copy()
final_df_pv['Stromproduktion'] = pd.Series(stromproduktion)
final_df_pv["Datum"] = pd.to_datetime(final_df_pv['Datum'])
start_date = dt.strptime('2023-11-15', '%Y-%m-%d').date()
end_date = dt.strptime('2023-11-29', '%Y-%m-%d').date()
heatmap_data_pv = final_df_pv.pivot_table(index='Stunde', columns='Datum', values='Stromproduktion').loc[:, start_date:end_date]

pred_pv_united = pd.read_csv("./pred_pv_united.csv")
stromproduktion =  electricity_prod(flaeche, panelleistung, temp_coeff, noct, sun_coeff, pred_pv_united['Strahlung Rieden [kW/m²]'], pred_pv_united['Temp. Rieden [°C]'])
final_df_pv = pred_pv_united.copy()
final_df_pv['Stromproduktion'] = pd.Series(stromproduktion)
final_df_pv["Datum"] = pd.to_datetime(final_df_pv['Datum'])
start_date = dt.strptime('2023-11-15', '%Y-%m-%d').date()
end_date = dt.strptime('2023-11-29', '%Y-%m-%d').date()
heatmap_data_pv = final_df_pv.pivot_table(index='Stunde', columns='Datum', values='Stromproduktion').loc[:, start_date:end_date]

with tab3:
    col1, col2 = st.columns(2)
    try:
        with col1:
            st.header("Ihre PV-Anlage")
            st.markdown("### Stromproduktion in der Zukunft")
            pv_heatmap(heatmap_data_pv)

        with col2:
            st.header("Ihre Ladestation")
            st.markdown("### Stromnutzung in der Zukunft")
            netz_heatmap_prediction(charge_station_id)
    except Exception as e:
        st.error(f"An error occured {e}")
        st.error(traceback.format_exc())


#netz_heatmap_prediction(charge_station_id)
