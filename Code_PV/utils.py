#Define classes for the project
import streamlit as st
import numpy as np
import pandas as pd
import datetime as dt


# PV-Anlage - dataframes
def create_dataframes_pv():
    df_strahlung = pd.read_excel('vlotte_HSG_GlobalstrahlungBregenz_20231129.xlsx')
    df_strahlungtemp = pd.read_excel('vlotte_HSG_GlobalstrahlungTempBregenz_20231206.xlsx')
    df = pd.merge(df_strahlung, df_strahlungtemp, how='outer', left_on=['ZEIT', 'Strahlung Rieden [W/m²]'], right_on=['ZEIT', 'Strahlung Rieden [W/m²]'])
    df['Datum'] = df['ZEIT'].dt.date
    df['Zeit'] = df['ZEIT'].dt.time
    df['Stunde'] = df['ZEIT'].dt.hour
    df['Strahlung Rieden [W/m²]'] /= 1000
    for i in df['Datum']:
        i=str(i)
    #df = df[['Datum', 'Zeit', 'Stunde', 'Strahlung Rieden [W/m²]', 'Temp. Rieden [°C]']]
    df = df[['Datum', 'Stunde', 'Strahlung Rieden [W/m²]', 'Temp. Rieden [°C]']]

    df.columns = ['Datum', 'Stunde', 'Strahlung Rieden [kW/m²]', 'Temp. Rieden [°C]']
    df.dropna()

    # PV-Anlage - Grafik Solarertrag
    d = list(range(-90,100,10)) # Himmelsrichtung (Ausrichtung des Gebäudes) {'Ost':-90, 'Süd': 0,  'West':90} 
    dn = list(range(0,100,10)) # Neigungswinkel in ° (Dachneigung)
    # Kombinationen
    combs = []
    for i in d:
        for j in dn:
            if -10<=i<=10 and 25<=j<=35:
                s=1
                color='red'
            elif -58<=i<=55 and 6<=j<=52:
                s=0.95
                color='black'
            elif j<=62:
                s=0.9
                color='red'
            elif j<=69:
                s=0.85
                color='black'
            elif j<=76:
                s=0.8
                color='red'
            elif j<=81:
                s=0.75
                color='black'
            elif j<=87:
                s=0.7
                color='red'
            else:
                s=0.65
                color='black'
            
            #Verbesserungen
            if (i==-50 and j in (10,40,50)) or (i in (-40,-30,30,40) and j==50) or (i==50 and j in (10,40)):
                s=0.9
                color='red'
            if (i in (-90,90) and j==20) or (i in (-90,-80,80,90) and j==30) or (i in (-80,-70,70,80) and j==40) or (i in (-70,-60,50,60) and j==50) or (i in (-50,-40,-30,30,40) and j==60):
                s=0.85
                color='black'
            if (i in (-90,90) and j==40) or (i in (-80,70,80) and j==50) or (i in (-70,-60,50,60) and j==60):
                s=0.8
                color='red'
            if (i in (-90,90) and j==50) or (i in (-80,70,80) and j==60) or (i in (-60,-50,50,60) and j==70): 
                s=0.75
                color='black'
            if (i in (-90,90) and j==60) or (i in (-80,-70,70) and j==70) or (i in (-60,-50,-40,40,50) and j==80):
                s=0.7
                color='red'
            if (i in (-90,80-90) and j==70) or (i in (-80,-70,60,70) and j==80):
                s=0.65
                color='black'
            if (i in (-90,80-90) and j==80) or (i in (-70,-60,-50,50,60,70) and j==90): 
                s=0.6
                color='red'
            if (i in (-90,-80,80,90) and j==90) or (i==-80 and j==90):
                s=0.55
                color='black'
            combs.append((i,j,s,color)) # i=Gebäudesausrichtung °, j=Dachneigung °, s=Solarertrag (zw. 0-1), color for plot

    # df Solarertrag & Strahlung_pro_stunde
    d = [i for (i,j,k,l) in combs]
    dn = [j for (i,j,k,l) in combs]
    s = [k for (i,j,k,l) in combs]
    colors = [l for (i,j,k,l) in combs]
    df_solarertrag = pd.DataFrame(data={'Ausrichtung':d, 'Dachneigung':dn, 'Solarertrag':s, 'Color':colors})
    df_strahlung_pro_stunde = df.groupby(['Datum', 'Stunde']).mean().reset_index()

    return df, df_solarertrag, df_strahlung_pro_stunde


def create_dataframes_netz():
    df = pd.read_csv("./hsg_export_28.csv", sep = ";")
    df = df.loc[(df["meter_point"] == "Power.Active.Import ") & (df["value"] != "0")]
    df["value"] = df["value"].str.replace(',', '.').astype(float) / 1000
    stamm = pd.read_excel("vlotte_HSG_StammdatenLadepunkte_20231127.xlsx")

    stamm_relevant = stamm[["ChargeStationID", "AccessLevel" ]].copy()
    stamm_relevant = stamm_relevant.dropna()
    stamm_relevant["ChargeStationID"] = stamm_relevant["ChargeStationID"].str.replace("{","").str.replace("}","")
    stamm_relevant.set_index("ChargeStationID", inplace=True)

    stamm_dict = stamm_relevant.to_dict()
    stamm_dict = stamm_dict["AccessLevel"]

    df["ChargeStationID"] = df["ChargeStationID"].str.upper()
    df["AccessLevel"] = df["ChargeStationID"].map(stamm_dict)
    df = df.loc[df["AccessLevel"] == "PRIVATE"].copy()

    df = df[["ChargeStationID", "timestamp_message_UTC", "value"]]


    df["timestamp_message_UTC"] = pd.to_datetime(df["timestamp_message_UTC"], utc=True)
    df["hour"] = df["timestamp_message_UTC"].dt.hour
    df["date"] = df["timestamp_message_UTC"].dt.date
    df.to_csv("cleaned.csv", index=False)

    df["timestamp_message_UTC"] = pd.to_datetime(df["timestamp_message_UTC"], format="mixed")
    df["hour"] = df["timestamp_message_UTC"].dt.hour
    df["date"] = df["timestamp_message_UTC"].dt.date

    



    return df#, heatmap_data, heatmap_data_average


