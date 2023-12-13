import datetime as dt
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def netz_heatmap(id):
    df = pd.read_csv('cleaned.csv')
    df["timestamp_message_UTC"] = pd.to_datetime(df["timestamp_message_UTC"], format="mixed")
    df["hour"] = df["timestamp_message_UTC"].dt.hour
    df["date"] = df["timestamp_message_UTC"].dt.date
    
    station = df.loc[df["ChargeStationID"] == id].copy()
    station = station.groupby(["date", "hour"]).agg({"value": "mean"}).reset_index()

    if station.empty:
        st.warning(f"No data available for the specified Charge Station ID: {id}")
        return

    cmap = LinearSegmentedColormap.from_list('custom', ['#f5e9e9', '#fc0505'])

    heatmap_data = station.pivot_table(index='hour', columns='date', values='value', aggfunc='sum')

    fig, ax = plt.subplots(figsize=(12, 8))

    heatmap_plot = sns.heatmap(heatmap_data, cmap=cmap, annot=False, fmt=".1f", linewidths=.5)

    plt.title('Heatmap of the Energy Usage by Hour and Date')
    plt.xlabel('Date')
    plt.ylabel('Hour')

    cbar = heatmap_plot.collections[0].colorbar
    cbar.set_label('kW')


    date_labels = [f"{date.strftime('%d.%m.%Y')} {date.strftime('%a')}" for date in heatmap_data.columns]
    plt.xticks(ticks=[i + 0.5 for i in range(len(date_labels))], labels=date_labels, rotation=90, ha='right', rotation_mode='anchor')
    plt.yticks(rotation=0)
   
    st.pyplot(fig)


def netz_heatmap_weekly(id):
    df = pd.read_csv('cleaned.csv')
    df["timestamp_message_UTC"] = pd.to_datetime(df["timestamp_message_UTC"], format="mixed")
    df["hour"] = df["timestamp_message_UTC"].dt.hour
    df["day_name"] = df["timestamp_message_UTC"].dt.day_name()

    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df["day_name"] = pd.Categorical(df["day_name"], categories=days_order, ordered=True)
    
    station = df.loc[df["ChargeStationID"] == id].copy()
    station = station.groupby(["day_name", "hour"], observed=True).agg({"value": "mean"}).reset_index()

    if station.empty:
        st.warning(f"No data available for the specified Charge Station ID: {id}")
        return

    cmap = LinearSegmentedColormap.from_list('custom', ['#f5e9e9', '#fc0505'])

    heatmap_data = station.pivot_table(index='hour', columns='day_name', values='value', aggfunc='sum')

    fig, ax = plt.subplots(figsize=(12, 8))

    heatmap_plot = sns.heatmap(heatmap_data, cmap=cmap, annot=False, fmt=".1f", linewidths=.5)

    plt.title('Heatmap of the Average Energy Usage per Weekday')
    plt.xlabel('Weekday')
    plt.ylabel('Hour')

    cbar = heatmap_plot.collections[0].colorbar
    cbar.set_label('kW')

    plt.yticks(rotation=0)
   
    st.pyplot(fig)

