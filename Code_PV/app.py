# Smart Charger Advisor: User interface

############################## Remarks ##############################
# st.download_button at the end for user to download result matrix
# Beispiel Ladestation_id: 4940a04d-62a0-adf5-fe81-b67dd755cecc

#####################################################################



# imports
import streamlit as st
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime

from classes import Photovoltaik, Netz
from utils import create_dataframes_pv, create_dataframes_netz





# Main function
def main():
    # theme and title config
    st.set_page_config(page_title='Smart Charger Advisor: Illwerke und Ihre PV-Anlage')
    st.title('Smart Charger Advisor: Illwerke und Ihre PV-Anlage')

    # user inputs --> mehrere PV-Anlagen auf unterschiedliche Direktionen erlauben? (2. Taste 'andere PV-Anlage hinzufügen')
    # auch möglich, default values zu übernehmen
    flaeche = st.sidebar.number_input('Panelfläche in Quadratmeter')
    panelleistung = st.sidebar.number_input('Panelleistung der PV-Anlage in kWp', help='Spezifisch zur PV-Anlage (Typischerweise: 1kWp pro 7 m²)') #button 'About' to add
    ausrichtung = st.sidebar.slider('Ausrichtung des Gebäudes in Grad', min_value=-90, max_value=90, step=10, help='Ost:-90, Süd:0, West:90')
    dachneigung = st.sidebar.slider('Dachneigung in Grad', min_value=0, max_value=90, step=10, help='(0° - horizontal, 90° - vertikal)')
    temp_coeff = st.sidebar.number_input('Temperaturkoeffizient der PV-Anlage in %', help='Spezifisch zur PV-Anlage (typischerweise: zw. 0.30 und 0.45)') #button 'About' to add
    noct = st.sidebar.number_input('NOCT in °C', help='Spezifisch zur PV-Anlage (typischerweise: 25°C)') #button 'About' to add
    ladestation_id = st.sidebar.text_input('ID Ihrer Ladestation') 

    # dataframes:
    df_pv, df_solarertrag, df_strahlung_pro_stunde = create_dataframes_pv()
    df_netz = create_dataframes_netz()


    # button to get smart charger advice
    col1, col2, col3 = st.columns(3)
    if st.button('Wie stimmt die Illwerke mit meiner PV-Anlage?'):
        if not flaeche or not panelleistung or not ausrichtung or not dachneigung or not temp_coeff or not noct or not ladestation_id:
            st.error('Bitte alle Parameter eingeben')
        else:
            try:
                object_pv = Photovoltaik(ausrichtung, dachneigung, flaeche, panelleistung, noct, temp_coeff)
                object_netz = Netz(ladestation_id)

                #PV-Anlage
                sun_coeff = df_solarertrag.loc[(df_solarertrag['Ausrichtung']==ausrichtung) & (df_solarertrag['Dachneigung']==dachneigung)].reset_index(drop=True)
                sun_coeff = float(sun_coeff['Solarertrag'])
                globalstrahlung = df_strahlung_pro_stunde['Strahlung Rieden [kW/m²]']
                temperatur = df_strahlung_pro_stunde['Temp. Rieden [°C]']
                stromproduktion = object_pv.electricity_prod(sun_coeff, globalstrahlung, temperatur)
                final_df_pv = df_strahlung_pro_stunde.copy()
                final_df_pv['Stromproduktion'] = pd.Series(stromproduktion)
                start_date = datetime.strptime('2023-11-15', '%Y-%m-%d').date()
                end_date = datetime.strptime('2023-11-28', '%Y-%m-%d').date()
                heatmap_data_pv = final_df_pv.pivot_table(index='Stunde', columns='Datum', values='Stromproduktion').loc[:, start_date:end_date]
                
                with col2:
                    st.header('Ihre PV-Anlage')
                    object_pv.heat_map(heatmap_data_pv, 'Heatmap der Stromproduktion nach Zeit und Datum', 'Datum', 'Zeit', 'Stromproduktion [kW]')

                #Netzbelastung
                station = df_netz.loc[df_netz["ChargeStationID"] == ladestation_id].copy()
                station = station.groupby(["date", "hour"]).agg({"value": "mean"}).reset_index()

                heatmap_data_netz = station.pivot_table(index='hour', columns='date', values='value', aggfunc='sum')

                #weekly heatmap
                #df = pd.read_csv('cleaned.csv')
                df_netz["timestamp_message_UTC"] = pd.to_datetime(df_netz["timestamp_message_UTC"], format="mixed")
                df_netz["hour"] = df_netz["timestamp_message_UTC"].dt.hour
                df_netz["day_name"] = df_netz["timestamp_message_UTC"].dt.day_name()

                days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                df_netz["day_name"] = pd.Categorical(df_netz["day_name"], categories=days_order, ordered=True)
                
                station = df_netz.loc[df_netz["ChargeStationID"] == ladestation_id].copy()
                station = station.groupby(["day_name", "hour"], observed=True).agg({"value": "mean"}).reset_index()
            
                if station.empty:
                    st.warning(f"No data available for the specified Charge Station ID: {ladestation_id}")
                    #return

                heatmap_data_netz_average = station.pivot_table(index='hour', columns='day_name', values='value', aggfunc='sum')

                with col3:
                    st.header('Ihre Ladestation')
                    #object_netz.heat_map(heatmap_data_netz, ladestation_id, 'Heatmap der Energienutzung nach Zeit und Datum', 'Datum', 'Zeit', 'Energienutzung [kW]')
                
                
            except Exception as e:
                st.error(f'An error occured: {e}')
                st.error(traceback.format_exc())
    if st.button('Wie stimmt die Illwerke mit meiner PV-Anlage im Durchschnitt?'):
        if not flaeche or not panelleistung or not ausrichtung or not dachneigung or not temp_coeff or not noct or not ladestation_id:
            st.error('Bitte alle Parameter eingeben')

        else: 
            try:
                # PV-Anlage heatmap
                with col2:
                    st.header('Ihre PV-Anlage')

                #Netzbelastung
                with col3:
                    st.header('Ihre Ladestation')
                    #object_netz.heat_map(heatmap_data_netz_average, ladestation_id, 'Heatmap der durchschnittlichen Energienutzung pro Tag', 'Tag', 'Zeit', 'Energienutzung [kW]')
            except Exception as e:
                st.error(f'An error occured: {e}')
                st.error(traceback.format_exc())


if __name__ == "__main__":
    main()





# Display Average heatmaps


