import streamlit as st
import streamlit.components.v1 as com
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import date,datetime
import datetime
from psx import stocks,data_reader,tickers
import plotly.graph_objects as go
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
   st.switch_page("pages/Page3.py")

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
ticker_list= ["ABL","ABOT","AGP","AICL","AIRLINK","AKBL","APL","ARPL","ATLH","ATRL","AVN","BAFL","BAHL","BNWM","BOP","CEPB","CNERGY","COLG","DAWH","DCR","DGKC","EFERT","EFUG","ENGRO","EPCL","FABL","FATIMA","FCCL","FCEPL","FFBL","FFC","FHAM","GADT","GATM","GHGL","GLAXO","HBL","HCAR","HGFA","HINOON","HMB","HUBC","IBFL","ILP","INDU","INIL","ISL","JDWS","JVDC","KAPCO","KEL","KOHC","KTML","LCI","LOTCHEM","LUCK","MARI","MCB","MEBL","MTL","MUGHAL","MUREB","NATF","NBP","NESTLE","NML","NRL","OGDC","PABC","PAEL","PAKT","PGLC","PIBTL","PIOC","POL","POML","PPL","PSEL","PSMC","PSO","PSX","PTCL","RMPL","SCBPL","SEARL","SHEL","SHFA","SNGP","SRVI","SYS","TGL","THALL","TRG","UBL","UNITY","UPFL","YOUW"]
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
st.sidebar.image('Images/'+ticker+'.png')
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
