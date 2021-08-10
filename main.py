import yfinance as yf
import streamlit as st
import numpy as np
import pandas as pd
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

### FORMATTING
st.set_page_config(page_title="TC's Opt Screen", layout="wide")
#st.header("Welcome to TC's Option Screener")
st.write('Hello, *World!* :sunglasses:')

# askdfjhka


snsgreen, snsorange, snsred, snsblue, snsgrey = ['#55a868','#ff7f0e','#c44e52','#4c72b0','#8c8c8c']
irange  = range(5)
width   = [1.25,1.25,1.25,1.25,1.25]
opacity = [1,1,0.5,0.75,1]
ticker_list = ['SB', 'DSX', 'AMC', 'AAPL', 'MSFT', 'AMZN', 'FB', 'GOOGL', 'GOOG', 'BRK-B', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V', 'UNH', 'HD', 'PG', 'PYPL', 'MA', 'DIS', 'BAC', 'ADBE', 'CMCSA', 'PFE', 'XOM', 'CRM', 'CSCO', 'NFLX', 'VZ', 'NKE', 'KO', 'INTC', 'ABT', 'PEP', 'TMO', 'LLY', 'ACN', 'ABBV', 'WFC', 'T', 'WMT', 'AVGO', 'CVX', 'DHR', 'COST', 'MRK', 'TXN', 'MCD', 'MDT', 'QCOM', 'ORCL', 'HON', 'LIN', 'NEE', 'PM', 'BMY', 'MS', 'C', 'UNP', 'INTU', 'SBUX', 'UPS', 'GS', 'LOW', 'AMD', 'RTX', 'AMGN', 'AMAT', 'IBM', 'TGT', 'AMT', 'BA', 'BLK', 'MRNA', 'ISRG', 'NOW', 'GE', 'MMM', 'DE', 'CAT', 'AXP', 'SCHW', 'CVS', 'SPGI', 'CHTR', 'PLD', 'ZTS', 'ANTM', 'LRCX', 'MU', 'ADP']

### FUNCTIONS
@st.cache
def get_hist(ticker):
    df_hist = yf.download(
        tickers=ticker,
        # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        period="1y",
        # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        interval="1d",
        # group by ticker (to access via stk_px['SPY']), default is 'column'
        group_by='column',
        # adjust all OHLC automatically
        auto_adjust=True,
        # download pre/post regular market hours stk_px
        prepost=False,
        # use threads for mass downloading? (True/False/Integer)
        threads=True,
        # proxy URL scheme use use when downloading?
        proxy=None)

    df_hist = df_hist['Close'].to_frame()
    df_hist = df_hist.rename(columns={'Close':ticker})
    df_hist = df_hist.sort_index(axis=0, ascending=False)
    return df_hist


@st.cache
def get_exp_dates(ticker):
    weekly = False
    D2Emin = 20
    D2Emax = 90
    min_day, max_day = (1, 31) if weekly else (15, 21)
    filt_dates = lambda x: (D2Emin < ((x - datetime.datetime.now()).days) < D2Emax) \
                           and (min_day <= x.day <= max_day)
    exp_dates = list(map(lambda x: str(x.date()), filter(filt_dates, pd.to_datetime(yf.Ticker(ticker).options))))
    return exp_dates


def get_chains(ticker):
    call_chain, put_chain = yf.Ticker(ticker).option_chain(str(exp_date))
    call_chain['pcf'] = 1
    call_chain = call_chain[call_chain['inTheMoney'] == False]
    call_chain = call_chain.sort_index(ascending=True).reset_index(drop=True)
    put_chain['pcf'] = -1
    put_chain = put_chain[put_chain['inTheMoney'] == False]
    put_chain = put_chain.sort_index(ascending=False).reset_index(drop=True)
    return call_chain, put_chain


def plot_payoff():
    fig = go.Figure()

    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, shared_yaxes=True,
                        vertical_spacing=0.015, horizontal_spacing=0.01,
                        row_heights=[2, 1], column_widths=[8, 2],
                        column_titles=[f'{long_short} {lots} lots of <br>{exp_date} {ticker} {strike} strike {strategy}s @ {str(lastPrice)}', f'Premium Paid = {lots * mult * lastPrice:,.0f}'],
                        row_titles=['Profit & Loss', 'Rolling Ret & Backtest'])

    fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist[ticker],
                             name=ticker, connectgaps=True, line={'color': snsblue, 'width': 2.5}, opacity=0.8), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_hist[::-1].index, y=df_hist[ticker][::-1].rolling(td2e).mean(),
        name=str(td2e)+' td SMA',line={'color':snsgrey,'width':1}),row=1,col=1)

    for i, w, o in zip(irange, width, opacity):

        if (i - 1) * lots == 0:
            color = snsgrey
        elif (i - 1) * lots > 0:
            color = snsgreen
        else:
            color = snsred

        fig.add_trace(go.Scatter(
            x=[today_date, exp_date],
            y=[strike + i * pcf * lastPrice, strike + i * pcf * lastPrice],
            name=f'{i-1}x Premium', mode='lines', opacity=o, line={'color': color, 'width': w, 'dash': 'dashdot'}),
            row=1, col=1)

        fig.add_trace(go.Scatter(
            x=np.multiply([-1, 3], lastPrice) * lots * mult,
            y=[strike + i * pcf * lastPrice, strike + i * pcf * lastPrice],
            mode='lines+text',
            text=[f'{(i - 1) * lastPrice * lots * mult:,.0f}'], textposition='bottom center', textfont=dict(color=color),
            showlegend=False,
            name=f'{i}x Premium', opacity=o, line={'color': color, 'width': w, 'dash': 'dashdot'}), row=1, col=2)

        fig.add_trace(go.Scatter(
            x=[df_hist.index.min(), exp_date],
            y=[(strike + i * pcf * lastPrice) / df_hist[ticker][today_date] - 1,
               (strike + i * pcf * lastPrice) / df_hist[ticker][today_date] - 1],
            showlegend=False, mode='lines', opacity=o, line={'color': color, 'width': w, 'dash': 'dashdot'}), row=2, col=1)

    fig.add_trace(go.Scatter(x=np.multiply([-1, -1, 0, 1, 2, 3], lastPrice) * lots * mult,
                             y=strike + np.multiply([-4, 0, 1, 2, 3, 4], pcf) * lastPrice,
                             name='', line={'color': snsblue, 'width': 2}, opacity=0.7, showlegend=False), row=1, col=2)

    fig.add_trace(go.Scatter(x=np.multiply([-1.5, 3.5], lastPrice) * lots * mult,
                             y=[strike, strike],
                             name='', opacity=0.0, showlegend=False), row=1, col=2)

    fig.add_trace(go.Scatter(x=df_hist.index[::-1],
                             y=df_hist[ticker][::-1] / df_hist[ticker][::-1].shift(td2e) - 1,
                             name=str(td2e) + ' td Rol Ret', connectgaps=True, line={'color': snsgrey, 'width': 2}), row=2, col=1)

    fig.update_xaxes(zerolinecolor='grey', zerolinewidth=1.25, col=2, row=1)
    fig.update_yaxes(zerolinecolor='grey', zerolinewidth=1.25, col=1, row=2)
    fig.update_layout(margin=dict(l=0, r=0, t=60, b=0), template='seaborn', plot_bgcolor='#F0F2F6')
    fig.update_layout(height=700, width=1400)
    return fig


### BODY

col1, col2 = st.columns([2,10])

with col1:
    ticker = st.selectbox('Pick stock',(ticker_list))

df_hist = get_hist(ticker)

with col2:
    st.write('')
    st.write('')
    st.write(f'{ticker} *ref px* = {df_hist.iloc[0][0]:.2f}')


exp_dates = get_exp_dates(ticker)
exp_date = st.sidebar.selectbox('Pick exp date', exp_dates)
exp_date = datetime.datetime.strptime(exp_date, '%Y-%m-%d').date()
today_date = df_hist.index.max().date()
cd2e = (exp_date - today_date).days
td2e = cd2e//7*5 + cd2e%7
#st.write(today_date, exp_date, cd2e, td2e)

call_chain, put_chain = get_chains(ticker)
strategy = st.sidebar.radio('Strategy', ('Call', 'Put'))

if strategy == 'Call':
    strike = st.sidebar.selectbox('select strike', (call_chain['strike'].tolist()))
    i = call_chain[call_chain['strike']==strike].index[0]
    strike, lastPrice, impliedVolatility, pcf = call_chain.loc[i][['strike','lastPrice', 'impliedVolatility', 'pcf']].tolist()
else:
    strike = st.sidebar.selectbox('Select strike', (put_chain['strike'].tolist()))
    i = put_chain[put_chain['strike']==strike].index[0]
    strike, lastPrice, impliedVolatility, pcf = put_chain.loc[i][['strike','lastPrice', 'impliedVolatility', 'pcf']].tolist()

long_short = st.sidebar.radio('Long or Short', ('Long', 'Short'))
if long_short == 'Long':
    ls = 1
else:
    ls = -1

lots = st.sidebar.select_slider('N of lots', [1, 5, 10, 20, 50, 100]) * ls
mult = 100

fig = plot_payoff()
st.plotly_chart(fig)