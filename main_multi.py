import yfinance as yf
import streamlit as st
import numpy as np
import pandas as pd
import datetime
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from scipy.stats import norm

# LAYOUT
st.set_page_config(page_title="TC's Opt Screen", layout="wide")
st.sidebar.header("TC's Option Screener :sunglasses:")

# CONSTANTS
snsgreen, snsorange, snsred, snsblue, snsgrey = ['#55a868','#ff7f0e','#c44e52','#4c72b0','#8c8c8c']
riskfree = 0.005

# FUNCTIONS

# @st.cache
def get_stock_hist(ticker):
    df_hist = yf.download(
        tickers=ticker,
        # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        period="2y",
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
    df_hist = df_hist.sort_index(axis=0, ascending=True)
    return df_hist


# @st.cache
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
    put_chain['pcf'] = -1
    return call_chain, put_chain


def BlackSholes(CallPutFlag, S, X, T, r, v):
    d1 = (np.log(S/X)+(r+v*v/2)*T)/(v*np.sqrt(T))
    d2 = d1-v*np.sqrt(T)
    if CallPutFlag=='Call':
        return S*norm.cdf(d1)-X*np.exp(-r*T)*norm.cdf(d2)
    else:
        return X*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)


def calc_opt_hist(strategy, strike):
    minimize_me = lambda x: np.sqrt((BlackSholes(strategy, ref_price, strike, (exp_date - ref_date).days / 365, riskfree, x) - lastPrice) ** 2)
    min_result = minimize(minimize_me, 0.15, method='Nelder-Mead')
    if min_result.success:
        solver_vol = min_result.x[0]
    else:
        st.warning('Solver could not determine impl vol!')
        st.stop()
    opt_hist = stock_hist.copy()
    opt_hist['cd2e'] = (exp_date - stock_hist.index.date)
    opt_hist['cd2e'] = opt_hist['cd2e'].dt.days
    opt_hist['bs'] = opt_hist.apply(lambda row: BlackSholes(strategy, row[0], strike, row.cd2e / 365, riskfree, solver_vol), axis=1)
    return opt_hist, solver_vol


def get_fig():
    fig = go.Figure()

    col_title_2 = f'<b>Current P&L = {pnl_last:,.0f}'
    fig = make_subplots(rows=2, cols=3, shared_xaxes=True, shared_yaxes=True,
                        vertical_spacing=0.05, horizontal_spacing=0.01,
                        row_heights=[3, 1], column_widths=[10, 2, 1],
                        column_titles=[col_title_1, col_title_2, '<b>Prob'],
                        subplot_titles = ('', '', '', '<b>Rol Ret / Rol Vol / Backtest'))

    fig.add_trace(go.Scatter(x=stock_hist.index, y=stock_hist[ticker],
                             name=ticker+' close price', connectgaps=True, line={'color': snsblue, 'width': 2.5}, opacity=0.8),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=[tx_date, ref_date], y=[ref_price_tx_date, ref_price], mode='markers',
                             showlegend=False, line={'color': snsblue, 'width': 2.5}, opacity=1),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=stock_hist.index, y=stock_hist[ticker].rolling(td2e).mean(),
                             name=str(td2e)+' td SMA', visible='legendonly', line={'color':snsgrey,'width':1}),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=opt_hist[opt_hist['cd2e'] <= (cd2e + 20)].index,
                             y=opt_hist[opt_hist['cd2e'] <= (cd2e + 20)]['breakeven'],
                             name='BS approx', connectgaps=True, mode='lines', line={'color': snsorange, 'width': 2.5},
                             opacity=0.4),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=bell, y=range, name='norm dist', mode='lines',
                             line={'color': snsblue, 'width': 0.75}, fill='tozerox'),
                  row=1, col=3)

    fig.add_trace(go.Scatter(x=[max(bell)], y=[ref_price_tx_date], mode='markers',
                             showlegend=False, line={'color': snsblue, 'width': 2.5}, opacity=1),
                  row=1, col=3)

    for l, p in zip(levels_short, pnl_short):

        width = 1.25

        if p == 0:
            color = snsgrey
            o = 1
        elif p > 0:
            color = snsgreen
            o = (p/max(pnl))*0.66 + 0.34
        else:
            color = snsred
            o = (p/min(pnl))*0.66 + 0.34

        fig.add_trace(go.Scatter(x=[tx_date, exp_date], y=[l, l], mode='lines+text', opacity=o,
                                 text=['', f'<b>{l:,.2f}  {(l/ref_price_tx_date-1):+,.1%}'],
                                 textposition='bottom center', textfont=dict(color=color), showlegend=False,
                                 line={'color': color, 'width': width, 'dash': 'dashdot'}),
                      row=1, col=1)

        fig.add_trace(go.Scatter(x=[min(pnl), max(pnl)], y=[l, l], mode='lines+text', name='',
                                 text=['', f'<b>${p:,.0f}  ({p/(tx_price*lots*mult)*ls:.1f}x)'], textposition='bottom center', textfont=dict(color=color),
                                 showlegend=False, opacity=o, line={'color': color, 'width': width, 'dash': 'dashdot'}),
                      row=1, col=2)

        fig.add_trace(go.Scatter(x=[stock_hist.index.min(), exp_date],
                                 y=[l / stock_hist[ticker].iloc[-1] - 1, l / stock_hist[ticker].iloc[-1] - 1],
                                 showlegend=False, mode='lines', opacity=o,
                                 line={'color': color, 'width': width, 'dash': 'dashdot'}),
                      row=2, col=1)

        fig.add_trace(go.Scatter(x=[0, max(bell)*1.25], y=[l, l],
                                 text=['', f'<b>p{1 - norm.cdf((abs(l / ref_price_tx_date - 1)) / (solver_vol * np.sqrt(td2e / 252))):.0%}'],
                                 textfont=dict(color=color), textposition='bottom left',
                                 showlegend=False, mode='lines+text', opacity=o, name='',
                                 line={'color': color, 'width': width, 'dash': 'dashdot'}),
                      row=1, col=3)

    fig.add_trace(go.Scatter(x=pnl, y=levels, name='payoff diagram',
                             line={'color': snsblue, 'width': 2}, opacity=0.7),
                  row=1, col=2)

    padding = (max(pnl)-min(pnl))/5
    fig.add_trace(go.Scatter(x=[min(pnl), max(pnl)+padding],
                             y=[levels[0], levels[0]],
                             name='', opacity=0.0, showlegend=False),
                  row=1, col=2)

    fig.add_trace(go.Scatter(x=[tx_date, ref_date], y=[level_tx, level_last],
                             showlegend=False, mode='markers', marker=dict(color=snsorange), opacity=1),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=[0, pnl_last], y=[level_tx, level_last],
                             showlegend=False, mode='lines+markers', marker=dict(color=snsorange), opacity=1),
                  row=1, col=2)

    if strategy in ['Straddle', 'Strangle']:
        fig.add_trace(go.Scatter(x=[tx_date, ref_date], y=[put_strikes[0]-tx_price, put_strikes[0]-lastPrice],
                                 showlegend=False, mode='markers', marker=dict(color=snsorange), opacity=1),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=[0, pnl_last], y=[put_strikes[0]-tx_price, put_strikes[0]-lastPrice],
                                 showlegend=False, mode='lines+markers', marker=dict(color=snsorange), opacity=1),
                      row=1, col=2)
        fig.add_trace(go.Scatter(x=opt_hist[opt_hist['cd2e'] <= (cd2e + 20)].index,
                                 y=opt_hist[opt_hist['cd2e'] <= (cd2e + 20)]['breakeven lower'],
                                 name='BS approx', connectgaps=True, mode='lines',
                                 line={'color': snsorange, 'width': 2.5}, opacity=0.4),
                      row=1, col=1)

    rol_ret = stock_hist[ticker] / stock_hist[ticker].shift(td2e) - 1
    fig.add_trace(
        go.Scatter(x=rol_ret.index, y=rol_ret,
                   name=str(td2e) + ' td Rol Ret', connectgaps=True, line={'color': snsgrey, 'width': 1},
                   fill='tozeroy'),
        row=2, col=1)

    rol_vol = np.log(stock_hist[ticker]/stock_hist[ticker].shift(1)).rolling(td2e).std()*np.sqrt(td2e)
    fig.add_trace(go.Scatter(x=rol_vol.index, y=rol_vol,
                             name=str(td2e) + ' td Rol Vol', connectgaps=True, line={'color': snsorange, 'width': 1.25}),
                  row=2, col=1)

    fig.add_trace(go.Scatter(x=rol_vol.index, y=-rol_vol,
                             name='', connectgaps=True, showlegend=False,
                             line={'color': snsorange, 'width': 1.25}),
                  row=2, col=1)

    fig.add_trace(go.Scatter(x=[tx_date,tx_date], y=[solver_vol*np.sqrt(td2e/252), -solver_vol*np.sqrt(td2e/252)],
                             name='ivol solver', mode='markers',
                             line={'color': snsorange, 'width': 1.25}),
                  row=2, col=1)


    fig.update_xaxes(row=1, col=2, zerolinecolor='grey', zerolinewidth=1.25)
    fig.update_yaxes(row=2, col=1, zerolinecolor='grey', zerolinewidth=1.25, tickformat='.0%')

    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), template='seaborn', plot_bgcolor='#F0F2F6')
    fig.update_layout(height=210*3.7, width=297*3.7)  #, paper_bgcolor='yellow')
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))

    return fig


# BODY

from_file = st.sidebar.checkbox('Positions from file', value=False)

if from_file:
    placeholder = st.empty()
    with placeholder.container():
        pos_file = st.file_uploader("Drag/drop pos file here:", type=['csv'], help='Must be a CSV file please!')
        if not pos_file:
            st.warning('Waiting for your file...')
            st.stop()
        st.success('Great well done')

    df = pd.read_csv(pos_file, parse_dates=[1, 9], na_values='<NA>')
    df['exp_date'] = df['exp_date'].dt.date
    df['tx_date'] = df['tx_date'].dt.date
    placeholder.empty()
    df.dropna(how='all', inplace=True)
    df['description'] = df['ticker'] + ' - ' + df['long_short'] + ' - ' + df['strategy']

    pos_chosen = st.sidebar.selectbox('Pick one option trade:', df['description'])
    ticker, exp_date, long_short, lots, strategy, call_0, call_1, put_0, put_1, tx_date, tx_price, xxx = df[df['description'] == pos_chosen].values.tolist()[0]

    if long_short == 'Long':
        ls = 1
    else:
        ls = -1

    if not pd.isnull(call_0):
        if not pd.isnull(call_1):
            call_strikes = [call_0, call_1]
        else:
            call_strikes = [call_0]
    else:
        call_strikes = []
        
    if not pd.isnull(put_0):
        if not pd.isnull(put_1):
            put_strikes = [put_0, put_1]
        else:
            put_strikes = [put_0]
    else:
        put_strikes = []

    stock_hist = get_stock_hist(ticker)
    ref_date = stock_hist.index.max().date()
    ref_price = stock_hist.iloc[-1][0]
    call_chain, put_chain = get_chains(ticker)
    mult = 100

else:
    ticker = st.sidebar.text_input('Enter US stock ticker','AAPL')
    stock_hist = get_stock_hist(ticker)
    ref_date = stock_hist.index.max().date()
    ref_price = stock_hist.iloc[-1][0]

    exp_dates = get_exp_dates(ticker)
    exp_date = st.sidebar.selectbox('Pick exp date', exp_dates, index=len(exp_dates)-1)
    exp_date = datetime.datetime.strptime(exp_date, '%Y-%m-%d').date()

    call_chain, put_chain = get_chains(ticker)
    hide_itm = st.sidebar.checkbox('Hide ITM strikes', value=True)
    if hide_itm:
        call_chain = call_chain[call_chain['inTheMoney'] == False]
        put_chain = put_chain[put_chain['inTheMoney'] == False]

    call_strikes = st.sidebar.multiselect('Call strikes (max 2)', call_chain['strike'], call_chain['strike'].iloc[0])
    call_strikes = sorted(call_strikes)
    # format_func = lambda x: f'{x} ({x / ref_price:.0%})'
    put_strikes = st.sidebar.multiselect('Put strikes (max 2)', put_chain[::-1]['strike'])
    put_strikes = sorted(put_strikes, reverse=True)

    long_short = st.sidebar.radio('Long or Short', ('Long', 'Short'))
    if long_short == 'Long':
        ls = 1
    else:
        ls = -1

    lots = st.sidebar.select_slider('N of lots', [1, 5, 10, 20, 50, 100], 10) * ls
    mult = 100

    tx_date = st.sidebar.date_input('Transaction date override', ref_date)
    if tx_date > ref_date:
        tx_date = ref_date

cd2e = (exp_date - tx_date).days
td2e = cd2e//7*5 + cd2e%7

if len(call_strikes) == 1 and len(put_strikes) == 0:
    strategy = 'Call'
    lastPrice, impliedVolatility, pcf = \
        call_chain[call_chain['strike'] == call_strikes[0]][['lastPrice', 'impliedVolatility', 'pcf']].values.tolist()[0]
    opt_hist, solver_vol = calc_opt_hist(strategy='Call', strike=call_strikes[0])
    opt_hist['breakeven'] = call_strikes[0] + opt_hist['bs']
    i = opt_hist.index.get_loc(pd.to_datetime(tx_date), method='nearest')
    tx_price_suggest = round(opt_hist.iloc[i]['bs'], 2)
    if not from_file:
        tx_price = st.sidebar.number_input('Transaction  price override', min_value=0.01, max_value=None,
                                           value=max(tx_price_suggest, 0.01))
    # tx_price = 9
    levels = call_strikes[0] + np.multiply([-4, 0, 1, 2, 3, 4], tx_price)
    levels_short = levels[1:]
    level_tx = call_strikes[0] + tx_price
    level_last = call_strikes[0] + lastPrice
    pnl = np.multiply([-1, -1, 0, 1, 2, 3], tx_price) * lots * mult
    pnl_short = pnl[1:]
    pnl_last = (lastPrice - tx_price) * lots * mult
    col_title_1 = f'<b>{long_short} {lots} lots of the {exp_date:%d-%b-%y}<br>{ticker}  {call_strikes[0]}  strike  {strategy}s  @  {tx_price:.2f}  [now {lastPrice:.2f}]'
    pos_strikes = {'call_0': call_strikes[0], 'call_1': np.NaN, 'put_0': np.NaN, 'put_1': np.NaN}
elif len(call_strikes) == 2 and len(put_strikes) == 0:
    strategy = 'Call'
    #lower strike
    lastPrice, impliedVolatility, pcf = call_chain[call_chain['strike'] == call_strikes[0]][['lastPrice', 'impliedVolatility', 'pcf']].values.tolist()[0]
    opt_hist_lower, solver_vol = calc_opt_hist(strategy='Call', strike=call_strikes[0])

    # higher strike
    lastPrice, impliedVolatility, pcf = call_chain[call_chain['strike'] == call_strikes[1]][['lastPrice', 'impliedVolatility', 'pcf']].values.tolist()[0]
    opt_hist_higher, solver_vol = calc_opt_hist(strategy='Call', strike=call_strikes[1])

    # combine
    opt_hist = opt_hist_lower
    opt_hist['bs'] = opt_hist['bs'] -  opt_hist_higher['bs']
    opt_hist['breakeven'] = call_strikes[0] + opt_hist['bs']

    i = opt_hist.index.get_loc(pd.to_datetime(tx_date), method='nearest')
    tx_price_suggest = round(opt_hist.iloc[i]['bs'], 2)
    if not from_file:
        tx_price = st.sidebar.number_input('Transaction  price override', min_value=0.01, max_value=None,
                                           value=max(tx_price_suggest, 0.01))

    strategy = 'Call Spread'
    spread = call_strikes[1] - call_strikes[0]
    lastPrice = call_chain[call_chain['strike'] == call_strikes[0]]['lastPrice'].item() - call_chain[call_chain['strike'] == call_strikes[1]]['lastPrice'].item()

    levels = [call_strikes[0] - spread, call_strikes[0], call_strikes[0] + tx_price, call_strikes[1], call_strikes[1] + spread]
    levels_short = levels[1:4]
    level_tx = call_strikes[0] + tx_price
    level_last = call_strikes[0] + lastPrice
    pnl = np.multiply([-tx_price, -tx_price, 0, spread - tx_price, spread - tx_price], lots*mult)
    pnl_short = pnl[1:4]
    pnl_last = (lastPrice - tx_price) * lots * mult
    col_title_1 = f'<b>{long_short} {lots} lots of the {exp_date:%d-%b-%y}<br>{ticker}  {call_strikes[0]}/{call_strikes[1]}  strike  {strategy}  @  {tx_price:.2f}  [now {lastPrice:.2f}]'
    pos_strikes = {'call_0': call_strikes[0], 'call_1': call_strikes[1], 'put_0': np.NaN, 'put_1': np.NaN}
elif len(call_strikes) == 0 and len(put_strikes) == 1:
    strategy = 'Put'
    lastPrice, impliedVolatility, pcf = put_chain[put_chain['strike']==put_strikes[0]][['lastPrice', 'impliedVolatility', 'pcf']].values.tolist()[0]

    opt_hist, solver_vol = calc_opt_hist(strategy='Put', strike=put_strikes[0])
    opt_hist['breakeven'] = put_strikes[0] - opt_hist['bs']
    i = opt_hist.index.get_loc(pd.to_datetime(tx_date), method='nearest')
    tx_price_suggest = round(opt_hist.iloc[i]['bs'], 2)
    if not from_file:
        tx_price = st.sidebar.number_input('Transaction  price override', min_value=0.01, max_value=None,
                                           value=max(tx_price_suggest, 0.01))
    levels = put_strikes[0] - np.multiply([-4, 0, 1, 2, 3, 4], tx_price)
    pnl = np.multiply([-1, -1, 0, 1, 2, 3], tx_price) * lots * mult
    levels_short = levels[1:]
    level_tx = put_strikes[0] - tx_price
    level_last = put_strikes[0] - lastPrice
    pnl_short = pnl[1:]
    pnl_last = (lastPrice-tx_price) * lots * mult
    col_title_1 = f'<b>{long_short} {lots} lots of the {exp_date:%d-%b-%y}<br>{ticker}  {put_strikes[0]}  strike  {strategy}s  @  {tx_price:.2f}  [now {lastPrice:.2f}]'
    pos_strikes = {'call_0': np.NaN, 'call_1': np.NaN, 'put_0': put_strikes[0], 'put_1': np.NaN}
elif len(call_strikes) == 0 and len(put_strikes) == 2:
    strategy = 'Put'
    # higher strike
    lastPrice, impliedVolatility, pcf = \
        put_chain[put_chain['strike'] == put_strikes[0]][['lastPrice', 'impliedVolatility', 'pcf']].values.tolist()[0]
    opt_hist_higher, solver_vol = calc_opt_hist(strategy='Put', strike=put_strikes[0])
    # lower strike
    lastPrice, impliedVolatility, pcf = \
        put_chain[put_chain['strike'] == put_strikes[1]][['lastPrice', 'impliedVolatility', 'pcf']].values.tolist()[0]
    opt_hist_lower, solver_vol = calc_opt_hist(strategy='Put', strike=put_strikes[1])
    # combine
    opt_hist = opt_hist_higher
    opt_hist['bs'] = opt_hist['bs'] - opt_hist_lower['bs']
    opt_hist['breakeven'] = put_strikes[0] - opt_hist['bs']
    i = opt_hist.index.get_loc(pd.to_datetime(tx_date), method='nearest')
    tx_price_suggest = round(opt_hist.iloc[i]['bs'], 2)
    if not from_file:
        tx_price = st.sidebar.number_input('Transaction  price override', min_value=0.01, max_value=None,
                                           value=max(tx_price_suggest, 0.01))
    strategy = 'Put Spread'
    spread = put_strikes[0] - put_strikes[1]
    lastPrice = put_chain[put_chain['strike'] == put_strikes[0]]['lastPrice'].item() - \
                put_chain[put_chain['strike'] == put_strikes[1]]['lastPrice'].item()
    levels = [put_strikes[0] + spread, put_strikes[0], put_strikes[0] - tx_price, put_strikes[1],
              put_strikes[1] - spread]
    levels_short = levels[1:4]
    level_tx = put_strikes[0] - tx_price
    level_last = put_strikes[0] - lastPrice
    pnl = np.multiply([-tx_price, -tx_price, 0, spread - tx_price, spread - tx_price], lots*mult)
    pnl_short = pnl[1:4]
    pnl_last = (lastPrice - tx_price) * lots * mult
    col_title_1 = f'<b>{long_short} {lots} lots of the {exp_date:%d-%b-%y}<br>{ticker}  {put_strikes[0]}/{put_strikes[1]}  strike  {strategy}  @  {tx_price:.2f}  [now {lastPrice:.2f}]'
    pos_strikes = {'call_0': np.NaN, 'call_1': np.NaN, 'put_0': put_strikes[0], 'put_1': put_strikes[1]}
elif len(call_strikes) == 1 and len(put_strikes) == 1:
    # higher strike
    strategy = 'Call'
    lastPrice, impliedVolatility, pcf = \
        call_chain[call_chain['strike'] == call_strikes[0]][['lastPrice', 'impliedVolatility', 'pcf']].values.tolist()[0]
    opt_hist_higher, solver_vol = calc_opt_hist(strategy='Call', strike=call_strikes[0])
    # lower strike
    lastPrice, impliedVolatility, pcf = \
        put_chain[put_chain['strike'] == put_strikes[0]][['lastPrice', 'impliedVolatility', 'pcf']].values.tolist()[0]
    opt_hist_lower, solver_vol = calc_opt_hist(strategy='Put', strike=put_strikes[0])
    # combine
    opt_hist = opt_hist_higher
    opt_hist['bs'] = opt_hist['bs'] + opt_hist_lower['bs']
    opt_hist['breakeven'] = call_strikes[0] + opt_hist['bs']
    opt_hist['breakeven lower'] = put_strikes[0] - opt_hist['bs']
    i = opt_hist.index.get_loc(pd.to_datetime(tx_date), method='nearest')
    tx_price_suggest = round(opt_hist.iloc[i]['bs'], 2)
    if not from_file:
        tx_price = st.sidebar.number_input('Transaction  price override', min_value=0.01, max_value=None,
                                           value=max(tx_price_suggest, 0.01))
    if call_strikes[0] == put_strikes[0]:
        strategy = 'Straddle'
    elif call_strikes[0] > put_strikes[0]:
        strategy = 'Strangle'

    lastPrice = call_chain[call_chain['strike'] == call_strikes[0]]['lastPrice'].item() + \
                put_chain[put_chain['strike'] == put_strikes[0]]['lastPrice'].item()
    levels = [call_strikes[0] + 2*tx_price,
              call_strikes[0] + tx_price,
              call_strikes[0],
              put_strikes[0],
              put_strikes[0] - tx_price,
              put_strikes[0] - 2*tx_price]

    level_tx = call_strikes[0] + tx_price
    level_last = call_strikes[0] + lastPrice
    levels_short = levels
    pnl = np.multiply([tx_price, 0, -tx_price, -tx_price, 0, tx_price], lots*mult)
    pnl_last = (lastPrice - tx_price) * lots * mult
    pnl_short = pnl
    if strategy=="Straddle":
        col_title_1 = f'<b>{long_short} {lots} lots of the {exp_date:%d-%b-%y}<br>{ticker}  {call_strikes[0]}  strike  {strategy}s  @  {tx_price:.2f}  [now {lastPrice:.2f}]'
    else:
        col_title_1 = f'<b>{long_short} {lots} lots of the {exp_date:%d-%b-%y}<br>{ticker}  {put_strikes[0]}/{call_strikes[0]}  strike  {strategy}  @  {tx_price:.2f}  [now {lastPrice:.2f}]'
    pos_strikes = {'call_0': call_strikes[0], 'call_1': np.NaN, 'put_0': put_strikes[0], 'put_1': np.NaN}
else:
    st.error('Holy :poop: I can only handle calls, puts, vertical spreads and straddles for now :man-shrugging:')
    st.stop()

range_from = min(min(stock_hist[ticker]), min(levels))
range_to = max(max(stock_hist[ticker]), max(levels))
range = np.linspace(range_from, range_to, num=100)
ref_price_tx_date = opt_hist.iloc[i][0]
bell = norm.pdf((range/ref_price_tx_date-1)/(solver_vol*np.sqrt(cd2e/365)))

col1, col2 = st.columns((9, 2))

with col1:
    fig = get_fig()
    st.plotly_chart(fig)

with col2:
    st.write("")
    st.write("")
    st.write("")
    st.metric(ticker+' last', f'${ref_price:.2f}')
    st.metric('option last', f'{lastPrice:.2f}')
    # st.metric('ivol yfinance', f'{impliedVolatility:.0%}')
    st.metric('ivol solver', f'{solver_vol:.0%}')
    st.metric('P&L', f'${pnl_last:,.0f}')

st.write(f'Stock summary on [yahoo!] (https://finance.yahoo.com/quote/{ticker})')
buffer = io.BytesIO()
fig.write_image(file=buffer, format="pdf")
st.download_button("donwnload chart", data=buffer, file_name='prova.pdf') #mime=None, key=None, help=None, on_click=None, args=None, kwargs=None, *, disabled=False)

if not from_file:
    pos_details = {'ticker': ticker, 'exp_date': exp_date.isoformat(), 'long_short': long_short, 'lots': lots,
                   'strategy': strategy, 'tx_date': tx_date.isoformat(), 'tx_price': tx_price}
    df_pos = pd.DataFrame(data={**pos_details, **pos_strikes}, index=[0])
    df_pos_cols = ['ticker', 'exp_date', 'long_short', 'lots', 'strategy', 'call_0', 'call_1', 'put_0', 'put_1', 'tx_date', 'tx_price']
    df_pos = df_pos[df_pos_cols]
    st.download_button('download position csv', data=df_pos.to_csv(index=False).encode('utf-8'), file_name='pos.csv')
    st.markdown('<br><br><br>', unsafe_allow_html=True)
    st.markdown(df_pos.to_html(), unsafe_allow_html=True)