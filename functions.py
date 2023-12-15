import datetime as dt
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


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

    #plt.title('Heatmap of the Energy Usage by Hour and Date')
    plt.xlabel('Datum')
    plt.ylabel('Stunde')

    cbar = heatmap_plot.collections[0].colorbar
    cbar.set_label('Stromnutzung [kW]')


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

    #plt.title('Heatmap of the Average Energy Usage per Weekday')
    plt.xlabel('Wochentag')
    plt.ylabel('Stunde')

    cbar = heatmap_plot.collections[0].colorbar
    cbar.set_label('Stromnutzung [kW]')

    plt.yticks(rotation=0)
   
    st.pyplot(fig)

def netz_heatmap_prediction(id):
    df = pd.read_csv('cleaned.csv')
    df["timestamp_message_UTC"] = pd.to_datetime(df["timestamp_message_UTC"], format="mixed")
    df["prediction_timestamp"] = df["timestamp_message_UTC"].dt.strftime('%Y-%m-%d %H:00')

    station = df.loc[df["ChargeStationID"] == id].copy()

    hourly =  station.groupby(["prediction_timestamp"], observed=True).agg({"value": "mean"})
    # Convert the index to datetime
    hourly.index = pd.to_datetime(hourly.index)

    # Create a new dataframe with the expected timestamp range
    start_time = hourly.index.min()
    end_time = hourly.index.max()
    expected_index = pd.date_range(start=start_time, end=end_time, freq='1H')
    expected_df = pd.DataFrame(index=expected_index, columns=['value'])
    merged_df = pd.merge(expected_df, hourly, how='left', left_index=True, right_index=True)
    merged_df['value'] = merged_df['value_y'].fillna(0)
    merged_df = merged_df.drop(columns=["value_x", "value_y"])
    hourly = merged_df  

    hourly['one_prior'] = hourly['value'].shift(1)
    hourly = hourly.dropna().copy()
    
    hourly['hour'] = hourly.index.hour
    hourly['hourly_mean'] = hourly.groupby(['hour'])['value'].transform('mean')

    from sklearn.metrics import r2_score
    # Split the data into training and test sets based on time
    train_size = int(len(hourly) * 0.8)
    train, test = hourly.iloc[:train_size], hourly.iloc[train_size:]

    X_train, y_train = train[['one_prior', "hourly_mean"]], train['value']
    X_test, y_test = test[['one_prior', "hourly_mean"]], test['value']


    from sklearn.ensemble import RandomForestRegressor

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Make predictions on the test set using Linear Regression
    rf_predictions = rf_model.predict(X_test)

    # Calculate R-squared for Linear Regression predictions
    rf_r_squared = r2_score(y_test, rf_predictions)
    #print(f'Random Forest R-squared: {rf_r_squared:.4f}')

    # Get the last 'value' from your hourly dataset
    last_value = hourly['value'].iloc[-1]

    # Initialize a list to store the predictions for the next 24 hours
    predictions = []

    # Make predictions for the next 24 hours
    for _ in range(24):
        # Get the current hour from the last timestamp index
        current_hour = (hourly.index[-1] + pd.Timedelta(hours=1)).hour

        # Fetch the hourly mean for the current hour
        last_hourly_mean = hourly[hourly['hour'] == current_hour]['hourly_mean'].iloc[-1]

        # Predict the next hour using both 'last_value' and 'last_hourly_mean'
        prediction_input = pd.DataFrame([[last_value, last_hourly_mean]], columns=['one_prior', 'hourly_mean'])
        next_hour_prediction = rf_model.predict(prediction_input)

        # Store the prediction
        predictions.append(next_hour_prediction[0])

        # Update last_value for the next iteration
        last_value = next_hour_prediction[0]

    # Print or use the predictions as needed
    #print("Predictions for the next 24 hours:", predictions)

    hourly['date'] = hourly.index.date

   # Create a new dataframe for predictions with correct date and hour
    prediction_df = pd.DataFrame({
        'date': pd.date_range(start=hourly.index[-1] + pd.Timedelta(hours=1), periods=24, freq='H'),
        'hour': pd.date_range(start=hourly.index[-1] + pd.Timedelta(hours=1), periods=24, freq='H').hour,
        'value': predictions
    })

    # Concatenate the 'date' and 'hour' columns to create a single datetime index
    prediction_df['datetime'] = pd.to_datetime(prediction_df['date'].astype(str) + ' ' + prediction_df['hour'].astype(str) + ':00:00', format='mixed')
    prediction_df.set_index('datetime', inplace=True)

    # Extract only the date part for the 'date' column
    prediction_df['date'] = prediction_df.index.date

    # Concatenate the new dataframe with the original hourly dataframe
    hourly_with_predictions = pd.concat([hourly, prediction_df[['date', 'hour', 'value']]], axis=0)

    # Create a consistent hourly index in the 'hourly_with_predictions' dataframe
    hourly_with_predictions.index = pd.date_range(start=hourly.index[0], periods=len(hourly_with_predictions), freq='H')


    station = hourly_with_predictions.copy()   
    
  
    heatmap_data = station.pivot_table(index='hour', columns='date', values='value', aggfunc='sum')

    cmap = LinearSegmentedColormap.from_list('custom', ['#f5e9e9', '#fc0505'])
    fig, ax = plt.subplots(figsize=(12, 8))

    heatmap_plot = sns.heatmap(heatmap_data, cmap=cmap, annot=False, fmt=".1f", linewidths=.5)

    #plt.title('Heatmap of the Energy Usage by Hour and Date + 1 Day Prediction')
    plt.xlabel('Datum')
    plt.ylabel('Stunde')

    cbar = heatmap_plot.collections[0].colorbar
    cbar.set_label('Stromnutzung [kW]')


    date_labels = [f"{date.strftime('%d.%m.%Y')} {date.strftime('%a')}" for date in heatmap_data.columns]
    plt.xticks(ticks=[i + 0.5 for i in range(len(date_labels))], labels=date_labels, rotation=90, ha='right', rotation_mode='anchor')
    plt.yticks(rotation=0)

    st.pyplot(fig)


def electricity_prod(panelflaeche, panelleistung, temp_coeff, noct, sun_coeff, globalstrahlung, temperatur): 
    '''
    sun_coeff: float (% an Solarertrag), Kombination von direction und dachneigung)
    globalstrahlung: pd.Series (kW/m²)
    temperatur: pd.Series (°C)
    '''
    weather_adjustment = np.where(temperatur > noct, temp_coeff * (temperatur - noct), 0)
    weather_adjustment = pd.Series(weather_adjustment, index=temperatur.index)
    paneleffizienz = (panelleistung / panelflaeche) * (1-weather_adjustment)
    return globalstrahlung * paneleffizienz * panelflaeche * sun_coeff


def pv_heatmap(df_heatmap):
        '''
        df_heatmap: pd.DataFrame mit 5 Spalten: Datum (datetime.date), Stunde (datetime.hour), Globalstrahlung (kW/m²), Temperaturen (°C), Stromproduktion (kW)
        '''
        # Create a custom colormap that transitions from white to purple
        cmap = LinearSegmentedColormap.from_list('custom', ['#FFFFFF', '#800080'])
        
        fig, ax = plt.subplots(figsize=(12,8))
        heatmap_plot = sns.heatmap(df_heatmap, cmap=cmap, annot=False, fmt=".1f", linewidths=0.5)

        # Customise the plot
        #plt.title("Heatmap der Stromproduktion nach Stunde und Datum")
        plt.xlabel("Datum")
        plt.ylabel("Stunde")

        # Display annotations only in the sidebar (color bar)
        cbar = heatmap_plot.collections[0].colorbar
        cbar.set_label("Stromproduktion [kW]")

        # Rotate y-axis labels by 90 degrees
        date_labels = [f"{date.strftime('%d.%m.%Y')} {date.strftime('%a')}" for date in df_heatmap.columns]
        plt.xticks(ticks=[i + 0.5 for i in range(len(date_labels))], labels=date_labels, rotation=90, ha='right', rotation_mode='anchor')
        plt.yticks(rotation=0)

        # Show the plot
        st.pyplot(fig)


def pv_heatmap_weekly(heatmap_data_average, month):
# Create a custom colormap that transitions from white to purple
    cmap = LinearSegmentedColormap.from_list('custom', ['#FFFFFF', '#800080'])
    heatmap_data_average = heatmap_data_average.loc[heatmap_data_average['Monat']==month]
    heatmap_data_average = heatmap_data_average.groupby(['Tag', 'Stunde']).agg({"Stromproduktion":"mean"}).reset_index()
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data_average['Tag'] = pd.Categorical(heatmap_data_average['Tag'], categories=days_order, ordered=True)
    heatmap_data_average.sort_values('Tag', inplace=True)
    heatmap_data_average = heatmap_data_average.pivot_table(index='Stunde', columns='Tag', values='Stromproduktion', aggfunc='sum')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    heatmap_plot = sns.heatmap(heatmap_data_average, cmap=cmap, annot=False, fmt=".1f", linewidths=.5)
    
    # Customize the plot
    #plt.title('Heatmap der durchschnittlichen Stromproduktion nach Tag und Stunde')
    plt.xlabel('Wochentag')
    plt.ylabel('Stunde')

    # Display annotations only in the sidebar (color bar)
    cbar = heatmap_plot.collections[0].colorbar
    cbar.set_label('Stromproduktion [kW]')

    # Rotate y-axis labels by 90 degrees
    plt.yticks(rotation=0)

    # Show the plot
    st.pyplot(fig)
    



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
        