import yfinance as yf
import streamlit as st
import numpy as np
import pandas as pd
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from scipy.stats import norm

# LAYOUT
st.set_page_config(page_title="TC's Opt Screen", layout="wide")
st.sidebar.header("TC's Option Screener :sunglasses:")

# CONSTANTS
snsgreen, snsorange, snsred, snsblue, snsgrey = ['#55a868','#ff7f0e','#c44e52','#4c72b0','#8c8c8c']
irange  = range(5)
width   = [1.25,1.25,1.25,1.25,1.25]
opacity = [1,1,0.5,0.75,1]
riskfree = 0.005

# FUNCTIONS

@st.cache
def get_stock_hist(ticker):
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
    if df_hist.empty:
        raise NameError("You did not input a correct stock ticker! Try again.")

    df_hist = df_hist['Close'].to_frame()
    df_hist = df_hist.rename(columns={'Close':ticker})
    df_hist = df_hist.sort_index(axis=0, ascending=False)
    return df_hist


@st.cache
def get_exp_dates(ticker):
    weekly = False
    D2Emin = 0
    D2Emax = 120
    min_day, max_day = (1, 31) if weekly else (15, 21)
    filt_dates = lambda x: (D2Emin < ((x - datetime.datetime.now()).days) < D2Emax) \
                           and (min_day <= x.day <= max_day)
    exp_dates = list(map(lambda x: str(x.date()), filter(filt_dates, pd.to_datetime(yf.Ticker(ticker).options))))
    return exp_dates


def get_chains(ticker):
    call_chain, put_chain = yf.Ticker(ticker).option_chain(str(exp_date))
    call_chain['pcf'] = 1
    if hide_itm:
        call_chain = call_chain[call_chain['inTheMoney'] == False]
    call_chain = call_chain.sort_index(ascending=True).reset_index(drop=True)
    put_chain['pcf'] = -1
    if hide_itm:
        put_chain = put_chain[put_chain['inTheMoney'] == False]
    put_chain = put_chain.sort_index(ascending=False).reset_index(drop=True)
    return call_chain, put_chain


def BlackSholes(CallPutFlag,S,X,T,r,v):
    d1 = (np.log(S/X)+(r+v*v/2)*T)/(v*np.sqrt(T))
    d2 = d1-v*np.sqrt(T)
    if CallPutFlag=='Call':
        return S*norm.cdf(d1)-X*np.exp(-r*T)*norm.cdf(d2)
    else:
        return X*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)


def calc_vol_from_px(sigma):
    return np.sqrt((BlackSholes(strategy, ref_price, strike, (exp_date - ref_date).days / 365, riskfree, sigma) - lastPrice) ** 2)


def calc_opt_hist():
    opt_hist = stock_hist.copy()
    opt_hist['cd2e'] =  (exp_date - stock_hist.index.date)
    opt_hist['cd2e'] = opt_hist['cd2e'].dt.days
    opt_hist['bs'] = opt_hist.apply(lambda row: BlackSholes(strategy, row[0], strike, row.cd2e/365, riskfree, solver_vol), axis=1)
    return opt_hist


def plot_payoff():
    fig = go.Figure()

    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, shared_yaxes=True,
                        vertical_spacing=0.05, horizontal_spacing=0.01,
                        row_heights=[2, 1], column_widths=[8, 2],
                        column_titles=[f'<b>{long_short} {lots} lots of <br>{exp_date}  {ticker}  {strike}  strike  {strategy}s  @  {str(tx_price)}  [now {lastPrice}]',
                                       f'<b>Curr P&L = {lots * mult * (lastPrice - tx_price):,.0f}'],
                        # Premium Paid = {lots * mult * tx_price:,.0f}<br>
                        # row_titles=['<b>Profit & Loss', '<b>Rolling Ret & Backtest'],
                        subplot_titles = ('', '', '<b>Rolling Return & Backtest'))

    fig.add_trace(go.Scatter(x=stock_hist.index, y=stock_hist[ticker],
                             name=ticker, connectgaps=True, line={'color': snsblue, 'width': 2.5}, opacity=0.8), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=stock_hist[::-1].index, y=stock_hist[ticker][::-1].rolling(td2e).mean(),
        name=str(td2e)+' td SMA',line={'color':snsgrey,'width':1}),row=1,col=1)

    fig.add_trace(go.Scatter(x=opt_hist.index, y=strike + opt_hist['bs']*pcf,
                             name='BS approx', connectgaps=True, line={'color': snsorange, 'width': 2.5}, opacity=0.4), row=1, col=1)

    for i, w, o in zip(irange, width, opacity):

        if (i - 1) * lots == 0:
            color = snsgrey
        elif (i - 1) * lots > 0:
            color = snsgreen
        else:
            color = snsred

        fig.add_trace(go.Scatter(
            x=[tx_date, exp_date],
            y=[strike + i * pcf * tx_price, strike + i * pcf * tx_price],
            name=f'{i-1}x Premium', mode='lines', opacity=o, line={'color': color, 'width': w, 'dash': 'dashdot'}),
            row=1, col=1)

        fig.add_trace(go.Scatter(
            x=np.multiply([-1, 3], tx_price) * lots * mult,
            y=[strike + i * pcf * tx_price, strike + i * pcf * tx_price],
            mode='lines+text',
            text=[f'{(i - 1) * tx_price * lots * mult:,.0f}'], textposition='bottom center', textfont=dict(color=color),
            showlegend=False, name=f'{i}x Premium', opacity=o, line={'color': color, 'width': w, 'dash': 'dashdot'}), row=1, col=2)

        fig.add_trace(go.Scatter(
            x=[stock_hist.index.min(), exp_date],
            y=[(strike + i * pcf * tx_price) / stock_hist[ticker].iloc[0] - 1,
               (strike + i * pcf * tx_price) / stock_hist[ticker].iloc[0] - 1],
            showlegend=False, mode='lines', opacity=o, line={'color': color, 'width': w, 'dash': 'dashdot'}), row=2, col=1)

    fig.add_trace(go.Scatter(x=np.multiply([-1, -1, 0, 1, 2, 3], tx_price) * lots * mult,
                             y=strike + np.multiply([-4, 0, 1, 2, 3, 4], pcf) * tx_price,
                             name='', line={'color': snsblue, 'width': 2}, opacity=0.7, showlegend=False), row=1, col=2)

    fig.add_trace(go.Scatter(x=[0, (lastPrice-tx_price)*lots*mult],
                             y=[strike + tx_price * pcf, strike + lastPrice * pcf],
                             name='', line={'color': snsorange, 'width': 2}, opacity=1, showlegend=False), row=1, col=2)

    fig.add_trace(go.Scatter(x=[tx_date, ref_date],
                             y=[strike+tx_price*pcf, strike+lastPrice*pcf], mode='markers',
                             name='', line={'color': snsorange, 'width': 2}, opacity=1, showlegend=False), row=1, col=1)

    fig.add_trace(go.Scatter(x=np.multiply([-1.5, 3.5], tx_price) * lots * mult,
                             y=[strike, strike],
                             name='', opacity=0.0, showlegend=False), row=1, col=2)

    fig.add_trace(go.Scatter(x=stock_hist.index[::-1],
                             y=stock_hist[ticker][::-1] / stock_hist[ticker][::-1].shift(td2e) - 1,
                             name=str(td2e) + ' td Rol Ret', connectgaps=True, line={'color': snsgrey, 'width': 2}), row=2, col=1)

    fig.update_xaxes(zerolinecolor='grey', zerolinewidth=1.25, col=2, row=1)
    fig.update_yaxes(zerolinecolor='grey', zerolinewidth=1.25, tickformat='%', col=1, row=2)
    fig.update_layout(margin=dict(l=0, r=0, t=60, b=0), template='seaborn', plot_bgcolor='#F0F2F6')
    fig.update_layout(height=700, width=1400)
    return fig


# BODY

ticker = st.sidebar.text_input('Enter US stock ticker','AAPL')
stock_hist = get_stock_hist(ticker)
ref_date = stock_hist.index.max().date()
ref_price = stock_hist.iloc[0][0]

exp_dates = get_exp_dates(ticker)
exp_date = st.sidebar.selectbox('Pick exp date', exp_dates, index=len(exp_dates)-1)
exp_date = datetime.datetime.strptime(exp_date, '%Y-%m-%d').date()

strategy = st.sidebar.radio('Strategy', ('Call', 'Put'))
hide_itm = st.sidebar.checkbox('Hide ITM strikes', value=True)
call_chain, put_chain = get_chains(ticker)
if strategy == 'Call':
    strike = st.sidebar.selectbox(f'Select strike (ref price = {ref_price:.2f})', (call_chain['strike'].tolist()),
                                  format_func = lambda x: f'{x}  ({x / ref_price:.1%})')
    i = call_chain[call_chain['strike']==strike].index[0]
    strike, lastPrice, impliedVolatility, pcf = call_chain.loc[i][['strike','lastPrice', 'impliedVolatility', 'pcf']].tolist()
else:
    strike = st.sidebar.selectbox(f'Select strike (ref price = {ref_price:.2f})', (put_chain['strike'].tolist()),
                                  format_func = lambda x: f'{x}  ({x / ref_price:.1%})')
    i = put_chain[put_chain['strike']==strike].index[0]
    strike, lastPrice, impliedVolatility, pcf = put_chain.loc[i][['strike','lastPrice', 'impliedVolatility', 'pcf']].tolist()

long_short = st.sidebar.radio('Long or Short', ('Long', 'Short'))
if long_short == 'Long':
    ls = 1
else:
    ls = -1

lots = st.sidebar.select_slider('N of lots', [1, 5, 10, 20, 50, 100], 10) * ls
mult = 100

tx_date = st.sidebar.date_input('Trans date override', ref_date)
if tx_date > ref_date:
    tx_date = ref_date
cd2e = (exp_date - tx_date).days
td2e = cd2e//7*5 + cd2e%7

min_result = minimize(calc_vol_from_px, 0.15, method ='Nelder-Mead')
if min_result.success:
    solver_vol = min_result.x[0]
    #st.write(f'solver vol = {solver_vol:.1%}', BlackSholes(strategy, ref_price, strike, (exp_date - ref_date).days / 365, riskfree, solver_vol))
else:
    st.warning('Solver could not determine impl vol!')

opt_hist = calc_opt_hist()

i = opt_hist.index.get_loc(tx_date.isoformat(), method='nearest')
st.write(opt_hist.index)
st.write(tx_date)
st.write(opt_hist.index.get_loc(tx_date, method='nearest'))
#tx_price_suggest = round(opt_hist.iloc[i][2], 2)
tx_price_suggest = 1.0
tx_price = st.sidebar.number_input('Trans price override', min_value=0.01, max_value=None, value=max(tx_price_suggest,0.01))

# fig = plot_payoff()
# st.plotly_chart(fig)

st.write(f'Stock summary on [yahoo!] (https://finance.yahoo.com/quote/{ticker})')
