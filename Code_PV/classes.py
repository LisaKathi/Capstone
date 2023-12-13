#Define classes for the project
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


class Photovoltaik:
    def __init__(self, direction, dachneigung, panelflaeche, panelleistung, noct, temp_coeff):
        '''
        Photovoltaik constructor
        :param direction: Ausrichtung des Gebäudes (-90° - 90°) von Ost (-90°) zu West (90°) durch Süd (0°)
        :param dachneigung: Winkel (0-90°)
        :param panelflaeche: Anzahl Quadrameter für die gesamte Fläche der PV-Anlage
        :param panelleistung: Leistung der PV-Anlage in kWp (spzeifisch zur PV-Anlage)
        :param noct: NOCT (Nominal Operating Cell Temperature) in °C (spezifisch zur PV-Anlage)
        :param temp_coeff: Temperaturkoeffizient in %, um den Verlust an Paneleffizienz zu charakterisieren (spezifisch zur PV-Anlage)
        '''

        #if (isinstance(direction,float)):
        self.direction = direction
        self.dachneigung = dachneigung
        self.panelflaeche = panelflaeche
        self.panelleistung = panelleistung
        self.noct = noct 
        self.temp_coeff = temp_coeff
        #else:
        #    raise AttributeError 
        
    def electricity_prod(self, sun_coeff, globalstrahlung, temperatur): 
        '''
        sun_coeff: float (% an Solarertrag), Kombination von direction und dachneigung)
        globalstrahlung: pd.Series (kW/m²)
        temperatur: pd.Series (°C)
        '''
        weather_adjustment = np.where(temperatur > self.noct, self.temp_coeff * (temperatur - self.noct), 0)
        weather_adjustment = pd.Series(weather_adjustment, index=temperatur.index)
        paneleffizienz = (self.panelleistung / self.panelflaeche) * (1-weather_adjustment)
        return globalstrahlung * paneleffizienz * self.panelflaeche * sun_coeff
    
    def heat_map(self, df_heatmap, title, labelx, labely, labelbar):
        '''
        df_heatmap: pd.DataFrame mit 5 Spalten: Datum (datetime.date), Stunde (datetime.hour), Globalstrahlung (kW/m²), Temperaturen (°C), Stromproduktion (kW)
        '''
        # Create a custom colormap that transitions from white to purple
        cmap = LinearSegmentedColormap.from_list('custom', ['#FFFFFF', '#800080'])
        
        fig, ax = plt.subplots(figsize=(12,8))
        heatmap_plot = sns.heatmap(df_heatmap, cmap=cmap, annot=False, fmt=".1f", linewidths=0.5)

        # Customise the plot
        plt.title(title)
        plt.xlabel(labelx)
        plt.ylabel(labely)

        # Display annotations only in the sidebar (color bar)
        cbar = heatmap_plot.collections[0].colorbar
        cbar.set_label(labelbar)

        # Rotate y-axis labels by 90 degrees
        date_labels = [f"{date.strftime('%d.%m.%Y')} {date.strftime('%a')}" for date in df_heatmap.columns]
        plt.xticks(ticks=[i for i in range(len(date_labels))], labels=date_labels, rotation=90, ha='right', rotation_mode='anchor')
        plt.yticks(rotation=0)

        # Show the plot
        st.pyplot(fig)
        


class Netz:
    def __init__(self, id):
        '''
        :param id: ID der Ladestation (str)
        '''
        #if isinstance(id, str):
        self.id = id
        #else:
        #    raise AttributeError
        
    def heat_map(self, df_heatmap, id, title, labelx, labely, labelbar):
        # Create a custom colormap that transitions from white to purple
        cmap = LinearSegmentedColormap.from_list('custom', ['#FFFFFF', '#800080'])
        
        fig, ax = plt.subplots(figsize=(12,8))
        heatmap_plot = sns.heatmap(df_heatmap, cmap=cmap, annot=False, fmt=".1f", linewidths=0.5)

        # Customise the plot
        plt.title(title)
        plt.xlabel(labelx)
        plt.ylabel(labely)

        # Display annotations only in the sidebar (color bar)
        cbar = heatmap_plot.collections[0].colorbar
        cbar.set_label(labelbar)

        # Rotate y-axis labels by 90 degrees
        date_labels = [f"{date.strftime('%d.%m.%Y')} {date.strftime('%a')}" for date in df_heatmap.columns]
        plt.xticks(ticks=[i for i in range(len(date_labels))], labels=date_labels, rotation=90, ha='right', rotation_mode='anchor')
        plt.yticks(rotation=0)

        # Show the plot
        st.pyplot(fig)



