import streamlit as st
import streamlit.components.v1 as com
import pandas as pd
import investpy
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import investpy
from tqdm import tqdm
import pages
import time
from datetime import date
import datetime
import requests
from sklearn.preprocessing import MinMaxScaler
st.set_page_config(page_title="Details",page_icon="icon.jpg",initial_sidebar_state="collapsed",layout="wide")
with open('style.css') as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://images.pexels.com/photos/4593876/pexels-photo-4593876.jpeg?auto=compress&cs=tinysrgb&w=600");
    background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
    background-position: center;  
    background-repeat: no-repeat;

    }
</style>
"""
st.markdown(background_image, unsafe_allow_html=True)
st.markdown("""
<style>
    #MainMenu, header, footer {visibility: hidden;}

    /* This code gets the first element on the sidebar,
    and overrides its default styling */
    section[data-testid="stSidebar"] div:first-child {
        top: 0;
        height: 100vh;
    }
</style>
""",unsafe_allow_html=True)
st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
</style>
""",
    unsafe_allow_html=True,
)
st.image('icon.jpg')
st.header("Economic Indicators")
st.subheader('Kibor Rates')
st.write('KIBOR, or the Karachi Interbank Offered Rate, is a benchmark interest rate used in Pakistan. It is similar to other interbank rates like LIBOR (London Interbank Offered Rate) and EURIBOR (Euro Interbank Offered Rate).KIBOR is the average interest rate at which major banks in Pakistan are willing to lend to one another. It serves as a benchmark for various financial products, including loans, mortgages, and savings accounts. The rate is calculated daily and reflects the cost of borrowing funds in the interbank market.KIBOR is determined by a panel of selected banks, which report the rates at which they are willing to lend unsecured funds to other banks. These rates are collected and averaged to determine the daily KIBOR. The calculation and publication of KIBOR are overseen by the Financial Markets Association of Pakistan (FMAP).Here is the Recent Kibor Rates Data')
import requests
import pandas as pd
url = "https://www.brecorder.com/markets/kibor-rates"
response = requests.get(url)
kibor_data = pd.read_html(response.content)[0]
button=st.button("Show Kibor Data")
if button:
    st.write(kibor_data)
graph_Kibor=st.button("Show Graph")
if graph_Kibor:
  st.image("output.png")
st.subheader("EPS Values of Selected Stock")
st.write("Earnings Per Share (EPS) is a financial metric that measures a company profitability on a per-share basis, providing insight into how much profit each share of stock generates. To calculate basic EPS, divide the company net income by the average number of outstanding shares. For example, if a company has a net income of $10 million and 2 million shares outstanding, the EPS would be $5.There are variations such as diluted EPS, which considers the potential dilution from convertible securities like stock options and warrants. For instance, if the company has convertible securities that could add 500,000 shares to the total, the diluted EPS would be $4.")
graph_eps=st.button("Show EPS Graph")
if graph_eps:
     st.image("output2.png")
st.subheader("Dividend of Selected Stock")
st.write('Dividends are payments made by companies to their shareholders from their profits. These payments can come in various forms, such as cash, additional shares of stock, or other property. Cash dividends are the most common type and involve direct cash payments to shareholders. Stock dividends distribute additional shares of the company stock. Special dividends are one-time payments made when a company has substantial profits, and preferred dividends are regular payments to holders of preferred shares with a fixed dividend rate.Select the company from selectbox below and click on the button to show Dividend Data')
graph_dividend=st.button("Show Dividend Graph")
if graph_dividend:
   st.image("output3.png")