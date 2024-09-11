import investpy.data
import streamlit as st
import streamlit.components.v1 as com
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pages
import time
from st_pages import show_pages,hide_pages,get_pages
from datetime import date,datetime
import datetime
from psx import stocks,data_reader,tickers
import requests
import plotly.graph_objects as go
import warnings
from sklearn.preprocessing import MinMaxScaler
st.set_page_config(page_title="N SPOT",layout="wide",page_icon="icon.jpg")
st.image("icon.jpg")
st.header('Namal SPOT',divider='blue',)
with st.sidebar:
   st.image('icon.jpg')
   selected=option_menu(
   menu_title="Pages",
   options=["home","Economic Indicators","Details of Stocks"],
   icons=["house-door-fill","bi bi-bar-chart-line","info-circle-fill"],
   menu_icon="receipt-cutoff",
   default_index=0,
   
   )
if selected == "Economic Indicators":
   st.switch_page("pages/page2.py")
if selected == "Details of Stocks":
   st.switch_page("pages/page3.py")

background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://img.freepik.com/free-vector/dynamic-lines-background-paper-style_23-2149021103.jpg?w=996&t=st=1716214317~exp=1716214917~hmac=a50af545a0001b23df503188dde11e6a82b69b1afb780e6aa7503084cfa28993");
    background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
    background-position: center;  
    background-repeat: no-repeat;

    }
</style>
"""
st.markdown(background_image, unsafe_allow_html=True)
sidebar_background = """
<style>
[data-testid="stSidebar"] {
        background-image: url("https://img.freepik.com/free-photo/background-with-white-round-lines_23-2148811508.jpg?t=st=1716228467~exp=1716232067~hmac=5b5df8fc05be80e7b18a10744fd7aa959ca8d9e541bf02893edbe3ac28536bed&w=900");
       background-attachment: fixed;
}
</style>
"""
st.markdown(
    """
    <style>
    .stDeployButton {
            visibility: hidden;
        }
        [data-testid="stStatusWidget"] {
    visibility: hidden;
}
    </style>
    """, unsafe_allow_html=True
)
st.markdown(sidebar_background, unsafe_allow_html=True)
com.html("""
<H1> Namal Stock Prediction Online Tool</H1>
         <div class="khansb">
Welcome to  N SPOT,N SPOT is a stock price prediction website that uses advanced machine learning techniques to analyze and forecast stock prices. It provides users with accurate predictions and visualizations to help them make informed investment decisions. The website offers a user-friendly interface and a variety of tools for data analysis.</div>
<style>
         
      html
         {
         margin: auto;
         overflow: auto;
        background: transparent;
         background-attachment: fixed;
         background-size: 400% 400%;
         color:black;
         font-size: large;
               }
H1{
         background-image: url('icon.jpg');

}
         </style>
""")
com.html(
   """
<div class="basis">The Prediction is made on the basis of:
<ol>
<li>Historical Data</li>
<li>Market News</li>
<li>Economic Indicators</li>
</ol>
</div>
<style>
         
      html
         {
         font-size: large;
               }
     li{
     font-weight: bold; 
     }
"""
)
ticker_list= ["ABL","ABOT","AGP","AICL","AIRLINK","AKBL","APL","ARPL","ATLH","ATRL","AVN","BAFL","BAHL","BNWM","BOP","CEPB","CNERGY","COLG","DAWH","DCR","DGKC","EFERT","EFUG","ENGRO","EPCL","FABL","FATIMA","FCCL","FCEPL","FFBL","FFC","FHAM","GADT","GATM","GHGL","GLAXO","HBL","HCAR","HGFA","HINOON","HMB","HUBC","IBFL","ILP","INDU","INIL","ISL","JDWS","JVDC","KAPCO","KEL","KOHC","KTML","LCI","LOTCHEM","LUCK","MARI","MCB","MEBL","MTL","MUGHAL","MUREB","NATF","NBP","NESTLE","NML","NRL","OGDC","PABC","PAEL","PAKT","PGLC","PIBTL","PIOC","POL","POML","PPL","PSEL","PSMC","PSO","PSX","PTC","RMPL","SCBPL","SEARL","SHEL","SHFA","SNGP","SRVI","SYS","TGL","THALL","TRG","UBL","UNITY","UPFL","YOUW"]
ticker=st.selectbox('Select the Company',ticker_list)
with st.sidebar:
   selected=option_menu(
   menu_title="selected company",
   options=[ticker],
   icons=["check2-circle"],
   menu_icon="building-fill")
   st.markdown(
    """
<style>
.sidebar .sidebar-content {
   background: transparent;
    color: white;
}
.menu .container-xxl[data-v-5af006b8] {
    background: transparent;
}
</style>
""",
    unsafe_allow_html=True,
)
with open('style.css') as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
if  ticker == "ABL":
   st.sidebar.image("Images\ABL.png")
if  ticker == "ABOT":
   st.sidebar.image("Images\ABOT.png")
if  ticker == "AGP":
   st.sidebar.image('Images/AGP.png')
if  ticker == "AICL":
   st.sidebar.image('Images/AICL.png')
if  ticker == "AIRLINK":
   st.sidebar.image('Images/AIRLINK.png')
if  ticker == "AKBL":
   st.sidebar.image('Images/AKBL.png')
if  ticker == "APL":
   st.sidebar.image('Images/APL.png')
if  ticker == "ARPL":
   st.sidebar.image('Images/ARPL.jpeg')
if  ticker == "ATLH":
   st.sidebar.image('Images/ATLH.jpeg')
if  ticker == "ATRL":
   st.sidebar.image('Images/ATRL.jpeg')
if  ticker == "AVN":
   st.sidebar.image('Images/AVN.jpeg')
if  ticker == "BAFL":
   st.sidebar.image('Images/BAFL.png')
if  ticker == "BAHL":
   st.sidebar.image('Images/BAHL.jpeg')
if  ticker == "BNWM":
   st.sidebar.image('Images/BNWM.png')
if  ticker == "BOP":
   st.sidebar.image('Images/BOP.png')
if  ticker == "CEPB":
   st.sidebar.image('Images/CEPB.png')
if  ticker == "CNERGY":
   st.sidebar.image('Images/CNERGY.png')
if  ticker == "COLG":
   st.sidebar.image('Images/COLG.png')
if  ticker == "DAWH":
   st.sidebar.image('Images/DAWH.jpeg')
if  ticker == "DCR":
   st.sidebar.image('Images/DCR.jpeg')
if  ticker == "DGKC":
   st.sidebar.image('Images/DGKC.png')
if  ticker == "EFERT":
   st.sidebar.image('Images/EFERT.jpeg')
if  ticker == "EFUG":
   st.sidebar.image('Images/EFUG.png')
if  ticker == "ENGRO":
   st.sidebar.image('Images/ENGRO.png')
if  ticker == "EPCL":
   st.sidebar.image('Images/EPCL.jpeg')
if  ticker == "FABL":
   st.sidebar.image('Images/FABL.jpeg')
if  ticker == "FATIMA":
   st.sidebar.image('Images/Fatima.jpeg')
if  ticker == "FCCL":
   st.sidebar.image('Images/FCCL.jpeg')
if  ticker == "FCEPL":
   st.sidebar.image('Images/FCEPL.png')
if  ticker == "FFBL":
   st.sidebar.image('Images/FFBL.png')
if  ticker == "FFC":
   st.sidebar.image('Images/FFC.png')
if  ticker == "FHAM":
   st.sidebar.image('Images/FHAM.jpeg')
if  ticker == "GADT":
   st.sidebar.image('Images/GADT.png')
if  ticker == "GATM":
   st.sidebar.image('Images/GATM.jpeg')
if  ticker == "GHGL":
   st.sidebar.image('Images/GHGL.png')
if  ticker == "GLAXO":
   st.sidebar.image('Images/GSK.png')
if  ticker == "HBL":
   st.sidebar.image('Images/HBL.png')
if  ticker == "HCAR":
   st.sidebar.image('Images/HCAR.png')
if  ticker == "HGFA":
   st.sidebar.image('Images/HGFA.jpeg')
if  ticker == "HINOON":
   data=pd.read_excel("DATA SET/HINOON_merged_final.xlsx")
if  ticker == "HMB":
   st.sidebar.image('Images/HMB.jpeg')
if  ticker == "HUBC":
   st.sidebar.image('Images/HUBC.png')
if  ticker == "IBFL":
   st.sidebar.image('Images/IBFL.jpeg')
if  ticker == "ILP":
   st.sidebar.image('Images/ILP.jpeg')
if  ticker == "INDU":
   st.sidebar.image('Images/INDU.png')
if  ticker == "INIL":
   st.sidebar.image('Images/INIL.jpeg')
if  ticker == "ISL":
   st.sidebar.image('Images/ISL.jpeg')
if  ticker == "JDWS":
   st.sidebar.image('Images/JDWS.jpeg')
if  ticker == "JVDC":
   st.sidebar.image('Images/JVDC.png')
if  ticker == "KAPCO":
   st.sidebar.image('Images/KAPCO.png')
if  ticker == "KEL":
   st.sidebar.image('Images/KEL.png')
if  ticker == "KOHC":
   st.sidebar.image('Images/KOHC.jpeg')
if  ticker == "KTML":
   st.sidebar.image('Images/KTML.jpeg')
if  ticker == "LCI":
   st.sidebar.image('Images/LCI.png')
if  ticker == "LOTCHEM":
   st.sidebar.image('Images/LOTCHEM.jpeg')
if  ticker == "LUCK":
   st.sidebar.image('Images/LUCK.jpeg')
if  ticker == "MARI":
   st.sidebar.image('Images/MARI.png')
if  ticker == "MCB":
   st.sidebar.image('Images/MCB.png')
if  ticker == "MEBL":
   st.sidebar.image('Images/MEBL.jpeg')
if  ticker == "MTL":
   st.sidebar.image('Images/MTL.png')
if  ticker == "MUGHAL":
   st.sidebar.image('Images/MUGHAL.png')
if  ticker == "MUREB":
   st.sidebar.image('Images/MUREB.jpeg')
if  ticker == "NATF":
   st.sidebar.image('Images/NATF.jpeg')
if  ticker == "NBP":
   st.sidebar.image('Images/NBP.jpeg')
if  ticker == "NESTLE":
   st.sidebar.image('Images/NESTLE.jpeg')
if  ticker == "NML":
   st.sidebar.image('Images/NML.png')
if  ticker == "NRL":
   st.sidebar.image('Images/NRL.png')
if  ticker == "OGDC":
   st.sidebar.image('Images/OGDC.png')
if  ticker == "PABC":
   st.sidebar.image('Images/PABC.jpeg')
if  ticker == "PAEL":
   st.sidebar.image('Images/PAEL.jpeg')
if  ticker == "PAKT":
    st.sidebar.image('Images/PAKT.jpeg')
if  ticker == "PGLC":
   st.sidebar.image('Images/PGLC.jpeg')
if  ticker == "PIBTL":
   st.sidebar.image('Images/PIBTL.png')
if  ticker == "PIOC":
   st.sidebar.image('Images/PIOC.jpeg')
if  ticker == "POL":
   st.sidebar.image('Images/POL.jpeg')
if  ticker == "POML":
   st.sidebar.image('Images/POML.jpeg')
if  ticker == "PPL":
   st.sidebar.image('Images/PPL.png')
if  ticker == "PSEL":
   st.sidebar.image('Images/PSEL.png')
if  ticker == "PSMC":
   st.sidebar.image('Images/PSMC.jpeg')
if  ticker == "PSO":
   st.sidebar.image('Images/PSO.png')
if  ticker == "PSX":
   st.sidebar.image('Images/PSX.jpeg')
if  ticker == "PTC":
   st.sidebar.image('Images/PTCL.png')
if  ticker == "RMPL":
   st.sidebar.image('Images/RMPL.jpeg')
if  ticker == "SCBPL":
   st.sidebar.image('Images/SCBPL.png')
if  ticker == "SEARL":
   st.sidebar.image("Images/SEARL.png")
if  ticker == "SHEL":
   st.sidebar.image('Images/shel.png')
if  ticker == "SHFA":
   st.sidebar.image('Images/SHFA.jpeg')
if  ticker == "SNGP":
   st.sidebar.image('Images/SNGP.png')
if  ticker == "SRVI":
   st.sidebar.image('Images/SRVI.jpeg')
if  ticker == "SYS":
   st.sidebar.image('Images/SYS.jpeg')
if  ticker == "TGL":
   st.sidebar.image('Images/TGL.jpeg')
if  ticker == "THALL":
   st.sidebar.image('Images/THAL.png')
if  ticker == "TRG":
   st.sidebar.image('Images/TRG.png')
if  ticker == "UBL":
   st.sidebar.image('Images/UBL.png')
if  ticker == "UNITY":
   st.sidebar.image('Images/UNITY.jpeg')
if  ticker == "UPFL":
   st.sidebar.image('Images/UPFL.jpeg')
if  ticker == "YOUW":
   st.sidebar.image('Images/YOUW.png')
st.sidebar.header('Choose Date from below')
start_date=st.sidebar.date_input('Start date',date (2024,1,1))
data=stocks(ticker, start=start_date,end=datetime.date.today())
data.insert(0,"Date",data.index,True)
data.reset_index(drop=True,inplace=True)
button_clicked=st.button('Show Data')
if button_clicked:
   st.table(data)
model_button=st.button("Show Graphs")
if model_button:
    # Convert the 'Date' column to datetime format
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Filter data from 1/1/2020 to present
    filtered_data = data[data['Date'] >= '2020-01-01']
    
    # Set the 'Date' column as the index
    filtered_data.set_index('Date', inplace=True)
    
    # Create a candlestick chart using Plotly
    fig = go.Figure(data=[go.Candlestick(x=filtered_data.index,
                                         open=filtered_data['Open'],
                                         high=filtered_data['High'],
                                         low=filtered_data['Low'],
                                         close=filtered_data['Close'])])
    
    fig.update_layout(title='Candlestick Chart from 1/1/2020 to Present',
                      yaxis_title='Price')
    
    # Select the 'Close' prices for training
    close_prices = filtered_data['Close'].values.reshape(-1, 1)
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    
    # Create sequences of 60 days for training
    def create_sequences(data, seq_length):
        sequences = []
        labels = []
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i + seq_length])
            labels.append(data[i + seq_length])
        return np.array(sequences), np.array(labels)
    
    seq_length = 60
    X, y = create_sequences(scaled_data, seq_length)
    
    # Split the data into training and test sets
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(seq_length, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(50))
    model.add(Dense(1))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    with st.spinner("Processing Graphs... please wait..."): 
    # Train the model
        history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_split=0.2)
    def plot_prices():
      # Predict the prices
      predicted_prices = model.predict(X_test)
      predicted_prices = scaler.inverse_transform(predicted_prices)
      
      # Visualize the results
      last_60_days = scaled_data[-60:]
      future_input = last_60_days.reshape((1, seq_length, 1))
      future_predictions = []
      
      for _ in range(7):
          future_pred = model.predict(future_input)
          future_predictions.append(future_pred[0])
          future_input = np.append(future_input[:, 1:, :], [future_pred], axis=1)
      
      future_predictions = scaler.inverse_transform(future_predictions).flatten()
      
      # Get the last 14 days of historical data
      last_14_days = filtered_data[-14:]
      
      # Correcting the plotting issue
      plt.figure(figsize=(14, 5))
      plt.plot(last_14_days.index, last_14_days['Close'], 'bo-', label='Historical Prices')  # Blue line with markers
      future_dates = pd.date_range(last_14_days.index[-1] + pd.Timedelta(days=1), periods=7)  # Ensure we have 7 future dates
      plt.plot(future_dates, future_predictions, 'yo--', label='Predicted Prices')  # Yellow dashed line with markers
      plt.xlabel('Date')
      plt.ylabel('Price')
      plt.title('Historical and Predicted Prices')
      plt.legend()
      plt.grid(True)
      plt.show()
    st.pyplot(plot_prices())
    st.plotly_chart(fig)