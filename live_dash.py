import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
import plotly.express as px
import pandas as pd
import sqlite3
import datetime
from config import stop_words
from collections import Counter
import re
import string
import base64
import json
from plotly.subplots import make_subplots
import numpy as np


# it's ok to use one shared sqlite connection
# as we are making selects only, no need for any kind of serialization as well
# conn = sqlite3.connect('db/twitter_data.db', check_same_thread=False)
# conn.text_factory = str


def connect_to_db():
    return sqlite3.connect('db/twitter_data.db', check_same_thread=False)


style = {'color': '#657786', 'text-align': 'center', 'font-family': '-apple-system, BlinkMacSystemFont, "Segoe UI", '
                                                                    'Roboto, Oxygen, Ubuntu, Cantarell, "Open Sans", '
                                                                    '"Helvetica Neue", sans-serif',
         'font-weight': '600'}

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    [html.H1("Nemat's Twitter Sentiment", style=style),

     # Filters
     html.Div(className='filters', children=[
         dcc.Input(id='keyword', value='', type='text', placeholder='Put keyword here',
                   style={'width': '20%', 'height': '36px', 'margin-left': '10px'}),
         dcc.Dropdown(id='user-dropdown',
                      options=[{'label': i, 'value': i} for i in
                               pd.read_sql('SELECT DISTINCT user FROM tweets_table', connect_to_db()).user],
                      multi=True, placeholder='Select a username',
                      style={'width': '65%', 'height': '36px'}),
         html.Div(id='total-tweets',
                  style={'width': '15%', 'height': '36px', 'color': '#657786',  # 657786 grey  # 1DA1F2 blue
                         'text-align': 'center',
                         'font-family': style.get('font-family'),
                         'font-weight': '600'}),
         dcc.Interval(id='total-tweets-update', interval=5 * 1000, n_intervals=0)
     ], style={'display': 'flex', 'width': '80%'}),

     # First Row ###
     html.Div(className='graphs', children=[
         # First Graph ### Live number of tweets and sentiment ###
         # First graph with data refreshing for last 100 rows
         dcc.Graph(id='live-graph', animate=False,
                   style={'height': 450, 'width': '70%', 'display': 'block', '`overflowY`': 'scroll'}),
         # First graph update interval
         dcc.Interval(id='graph-update', interval=3 * 1000, n_intervals=0),
         # dcc.Input(id='live-graph-input', value='300'),

         # Second Graph ### Pie Chart for sentiment ###
         # dcc.Dropdown(id='pie-calc', options=[{'label': 'Avg', 'value': 'avg'},
         #                             {'label': 'Sum', 'value': 'sum'}], value='avg'),
         dcc.Graph(id='pie-chart', animate=False, style={'width': '30%', 'display': 'block'})],
              style={'display': 'flex', 'flex-direction': 'row'}),
     # Second graph update interval - Pie Chart Update Interval
     dcc.Interval(id='pie-update', interval=5 * 1000, n_intervals=0),

     # Second Row ###
     html.Div(className='graphs', children=[
         # First Graph ### Live number of tweets and sentiment ###
         # First graph with data refreshing for last 100 rows
         dcc.Graph(id='hist-graph', animate=False,
                   style={'height': 450, 'width': '50%', 'display': 'block', '`overflowY`': 'scroll'}),
         # First graph update interval
         dcc.Interval(id='hist-update', interval=20 * 1000, n_intervals=0),
         # dcc.Input(),

         # Second Graph ### Word Frequency for sentiment ###
         dcc.Graph(id='word-freq', animate=False, style={'width': '25%', 'display': 'block'}),
         dcc.Interval(id='word_freq_int', interval=5 * 1000, n_intervals=0),

         # Third Graph --- Most tweeting people.
         dcc.Graph(id='user-freq', animate=False, style={'width': '25%', 'display': 'block'}),
         dcc.Interval(id='user_freq_int', interval=5 * 1000, n_intervals=0),

     ], style={'display': 'flex', 'flex-direction': 'row'}),

     # Third Row ###
     html.Div(className='graphs', children=[html.Div(id="recent-tweets"),
                                            dcc.Interval(id='text_tweet_update', interval=3 * 1000, n_intervals=0)])

     ])
app.title = 'Sample Title'


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


# Total number of tweets
@app.callback(Output('total-tweets', 'children'), [Input('total-tweets-update', 'n_intervals')])
def total_tweet_count(n_intervals):
    return 'Total Tweets: ' + human_format(
        pd.read_sql('SELECT COUNT(user) FROM tweets_table', connect_to_db()).values[0][0])


# Updating names from keyword searched
@app.callback(Output('user-dropdown', 'options'), [Input('keyword', 'value')])
def set_options(keyword):
    if keyword:
        a = [{'label': i, 'value': i} for i in
             sorted(pd.read_sql('SELECT tweets_table.* FROM tweets_fts fts LEFT JOIN tweets_table '
                                'ON fts.rowid = tweets_table.id WHERE fts.tweets_fts MATCH ? ORDER BY fts.rowid DESC;',
                                connect_to_db(), params=(keyword + '*',)).user.unique())]
    else:
        a = [{'label': i, 'value': i} for i in
             sorted(pd.read_sql('SELECT DISTINCT user FROM tweets_table', connect_to_db()).user)]
    return a


# No need since I don't want to update the values but only options
# @app.callback(Output('user-dropdown', 'value'), [Input('user-dropdown', 'options')])
# def set_value(available_options):
#     return [x['value'] for x in available_options]


def df_resample_sizes(df, maxlen=100):
    df['sentiment'] = [x for x in
                       [i['compound'] for i in df['sentiment'].apply(lambda x: json.loads(x.replace('\'', '"')))
                        if isinstance(i, dict)]]
    # vol_df = df[df['timestamp'].apply(lambda x: str(x).isdigit())]
    # df = df[df['timestamp'].apply(lambda x: str(x).isdigit())]
    if len(df) == 1:
        df['volume'] = 1
    else:
        vol_df = df.copy()
        vol_df['volume'] = 1
        ms_span = (df.index[-1] - df.index[0]).seconds * 1000
        rs = int(ms_span / maxlen)
        df = df.resample('{}ms'.format(int(rs))).mean()
        df.dropna(inplace=True)
        vol_df = vol_df.resample('{}ms'.format(int(rs))).sum()
        vol_df.dropna(inplace=True)
        df = df.join(vol_df['volume'])
    return df


def pulling_data(keyword, user, rows=1000):
    if keyword:
        if user:
            df = pd.read_sql(
                'SELECT * FROM tweets_table WHERE user IN ({}) ORDER BY id DESC, timestamp DESC LIMIT {};'.format(
                    str(user)[1:-1], rows), connect_to_db())
            df = df.loc[df.tweet.str.lower().str.find(keyword.lower()) > 0]
        else:
            df = pd.read_sql(
                'SELECT tweets_table.* FROM tweets_fts fts LEFT JOIN tweets_table ON fts.rowid = tweets_table.id '
                'WHERE fts.tweets_fts MATCH ? ORDER BY fts.rowid DESC LIMIT {};'.format(rows),
                connect_to_db(), params=(keyword + '*',))
    else:
        if user:
            try:
                df = pd.read_sql(
                    'SELECT * FROM tweets_table WHERE user IN ({}) ORDER BY id DESC, timestamp DESC LIMIT {};'.format(
                        str(user)[1:-1], rows), connect_to_db())
            except Exception as e:
                with open('errors.txt', 'a') as f:
                    f.write(str(e) + ' - def live_graph')
                    f.write('\n')
        else:
            df = pd.read_sql('SELECT * FROM tweets_table ORDER BY id DESC, timestamp DESC LIMIT {};'.format(rows),
                             connect_to_db())
    return df


# Live Chart --- First Graph
@app.callback(Output('live-graph', 'figure'),
              [Input('graph-update', 'n_intervals'),
               Input('keyword', 'value'),
               Input('user-dropdown', 'value')])
def live_graph(n_intervals, keyword, user):
    try:
        df = pulling_data(keyword, user)
        # Adjusting DataFrame for use df['source'] = df['source'].apply(lambda x: x.split('>')[1].split('<')[0])  #
        # replace('Twitter for', '') if len(df) == 0: df = pd.read_sql('SELECT * FROM tweets_table ORDER BY id DESC,
        # timestamp DESC LIMIT 1000;', connect_to_db())
        df = df[df['timestamp'].apply(lambda x: str(x).isdigit())]
        df.sort_values('timestamp', inplace=True)
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms').add(datetime.timedelta(hours=4))
        df.set_index('date', inplace=True)
        # df['count_tweet'] = df['tweet'].rolling(int(len(df) / 5)).mean()
        df.dropna(inplace=True)
        df = df_resample_sizes(df)
        X = df.index
        Y = df.volume.values
        sY = df.sentiment
        fig = make_subplots(specs=[[{'secondary_y': True}]])
        fig.add_trace(px.bar(x=X, y=Y).data[0])
        fig.add_trace(px.line(x=X, y=sY).data[0], secondary_y=True)  # sY.where(sY > 0).dropna()
        fig.add_trace(px.line(x=X, y=sY.where(sY >= 0)).data[0], secondary_y=True)
        # fig.data[1].mode = 'markers+lines'
        # fig.data[2].mode = 'markers+lines'
        fig.data[1].line.color = 'red'
        fig.data[2].line.color = 'green'

        if len(df) > 1:
            fig.update_traces({'line': {'width': 4}}, secondary_y=True)
            fig.update_layout(yaxis_title='Total Tweets', showlegend=False,
                              yaxis=dict(range=[min(Y), max(Y * 4)]),
                              margin=dict(l=0, r=0, t=50, b=50),
                              title=dict(text='Showing results for: {}'.format(keyword)))
            fig.update_yaxes(title_text='Sentiment', secondary_y=True)
            fig.add_annotation(showarrow=True,
                               x=df[df.sentiment == df.sentiment.max()].idxmax().max(),
                               y=df[df.sentiment == df.sentiment.max()].sentiment.max(),
                               text='Happy :)', xref='x', yref='y2')
            fig.add_annotation(showarrow=True,
                               x=df[df.sentiment == df.sentiment.min()].idxmin().min(),
                               y=df[df.sentiment == df.sentiment.min()].sentiment.min(),
                               text='Sad :(', xref='x', yref='y2')
        else:
            fig.update_traces({'line': {'width': 4}}, secondary_y=True)
            fig.update_yaxes(title_text='Sentiment', secondary_y=True)
        return fig
    except Exception as e:
        with open('errors.txt', 'a') as f:
            f.write(str(e) + ' - def live_graph')
            f.write('\n')
        return dict(layout=dict(xaxis=dict(visible=False), yaxis=dict(visible=False),
                                annotations=[dict(text='Error', xref='paper', yref='paper',
                                                  showarrow=False, font=dict(size=28))]))


# Pie Chart --- Second graph
@app.callback(Output('pie-chart', 'figure'),
              [Input('pie-update', 'n_intervals'),
               Input('keyword', 'value'),
               Input('user-dropdown', 'value')])
def pie_sentiment(n_intervals, keyword, user, calc='sum'):
    try:
        df = pulling_data(keyword, user)

        df['tweet'] = [re.sub('^RT|(@[^\s]+)|(http[^\s]+)', '', i).strip() for i in df['tweet']]
        df = df[df['tweet'] != '']

        compound = [x for x in [i['compound'] for i in df['sentiment'].apply(lambda x: json.loads(x.replace('\'', '"')))
                                if isinstance(i, dict)]]
        # positive = [x for x in [i['pos'] for i in df['sentiment'].apply(lambda x: json.loads(x.replace('\'', '"')))
        #                         if isinstance(i, dict)]]
        # negative = [x for x in [i['neg'] for i in df['sentiment'].apply(lambda x: json.loads(x.replace('\'', '"')))
        #                         if isinstance(i, dict)]]
        # neutral = [x for x in [i['neu'] for i in df['sentiment'].apply(lambda x: json.loads(x.replace('\'', '"')))
        #                        if isinstance(i, dict)]]

        pos = [x for x in compound if x > 0]
        neg = [abs(x) for x in compound if x < 0]
        names = ['Positive', 'Negative']
        if calc == 'sum':
            fig = px.pie(names=names, values=[sum(pos), sum(neg)],
                         color=names, color_discrete_sequence=['green', 'red'])
        else:
            fig = px.pie(names=names, values=[sum(pos) / len(pos), sum(neg) / len(neg)],
                         color=names, color_discrete_sequence=['green', 'red'])

        fig['data'][0]['hovertemplate'] = '<b>%{label}</b><br><b>%{value}</b>'
        fig.update_traces(textinfo='percent+label')
        fig.update_layout(margin=dict(l=50, r=30, t=0, b=0),
                          title={'text': '<b>Sentiment</b>', 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
                          showlegend=False)
        return fig
    except Exception as e:
        with open('errors.txt', 'a') as f:
            f.write(str(e) + ' - def pie_sentiment')
            f.write('\n')
        return {}


# Histogram or etc --- Third graph
@app.callback(Output('hist-graph', 'figure'),
              [Input('hist-update', 'n_intervals'),
               Input('keyword', 'value'),
               Input('user-dropdown', 'value')])
def update_graph_scatter(n_intervals, keyword, user):
    try:
        df = pulling_data(keyword, user, rows=10000)

        # Adjusting DataFrame for use
        # df['source'] = df['source'].apply(lambda x: x.split('>')[1].split('<')[0])  # replace('Twitter for', '')
        df = df[df['timestamp'].apply(lambda x: str(x).isdigit())]
        df.sort_values('timestamp', inplace=True)
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms').add(datetime.timedelta(hours=4))
        df.set_index('date', inplace=True)
        # df['count_tweet'] = df['tweet'].rolling(int(len(df) / 5)).mean()
        df.dropna(inplace=True)
        df['sentiment'] = [x for x in
                           [i['compound'] for i in df['sentiment'].apply(lambda x: json.loads(x.replace('\'', '"')))
                            if isinstance(i, dict)]]
        # df = df_resample_sizes(df, maxlen=100)
        df['volume'] = 1
        # df['volume'].groupby(df.index.hour).sum()
        # df['sentiment'].groupby(df.index.hour).mean()

        a = pd.DataFrame([i for i in range(24)]).join(
            (df['volume'].groupby(df.index.hour).sum(), df['sentiment'].groupby(df.index.hour).mean())).drop(columns=0)

        # Y = df.resample('d').sum()
        X = a.index
        Y = a.volume.values
        sY = a.sentiment.values
        fig = make_subplots(specs=[[{'secondary_y': True}]])

        fig.add_trace(px.bar(x=X, y=Y).data[0])
        fig.add_trace(px.line(x=X, y=sY).data[0], secondary_y=True)  # sY.where(sY > 0).dropna()
        fig.add_trace(px.line(x=X, y=np.positive(sY)).data[0], secondary_y=True)
        fig.data[1].mode = 'markers+lines'
        fig.data[2].mode = 'markers+lines'
        fig.data[1].line.color = 'red'
        fig.data[2].line.color = 'green'
        fig.update_traces({'line': {'width': 4}}, secondary_y=True)

        fig.update_layout(yaxis_title='<b>Tweets</b>', margin=dict(l=0, r=0, t=50, b=50), showlegend=False,
                          xaxis=dict(dtick=1, range=[-0.5, 23.5]), hovermode='x',
                          yaxis=dict(range=[min(Y), max(Y * 4)]))
        fig.update_yaxes(title_text='<b>Sentiment</b>', secondary_y=True)
        return fig

    except Exception as e:
        with open('errors.txt', 'a') as f:
            f.write(str(e) + ' - def histogram')
            f.write('\n')
        return {}


# Bar --- Forth graph --- Most used words Freq
@app.callback(Output('word-freq', 'figure'),
              [Input('word_freq_int', 'n_intervals'),
               Input('keyword', 'value'),
               Input('user-dropdown', 'value')])
def word_counts(n_intervals, keyword, user, top_words=10):
    try:
        df = pulling_data(keyword, user)

        # Clean text part of df from RTs and usernames
        # a = [i.split(':', 1)[1].strip() for i in [re.sub(r'([@\bhttp\b])\w+', '', i).strip() for i
        # in df['tweet'] if '@' in i] if 'RT' in i[:2]]
        a = [re.sub('^RT|(@[^\s]+)|(http[^\s]+)', '', i).strip() for i in df['tweet']]

        # a = [i.split(':', 1)[1].strip() if 'RT' in i else i.split(' ', 1)[1].strip() if '@' in i else i for i
        # in df['tweet']]
        # a = [i.split('http', 1)[0].strip() if 'http' in i else i for i in a]
        # make a counter with blacklist words and empty word with some big value - we'll use it later to filter counter
        stop_words.append('')
        blacklist_counter = Counter(dict(zip(stop_words, [1000000] * len(stop_words))))
        # complie a regex for split operations (punctuation list, plus space and new line)
        punctuation = [str(i) for i in string.punctuation]
        split_regex = re.compile("[ \n" + re.escape("".join(punctuation)) + ']')
        words = split_regex.split(' '.join(a).lower())
        word_count = pd.DataFrame((Counter(words) - blacklist_counter).most_common(top_words), columns=['X', 'Y'])
        # Graph
        X = [x[:10]+'...' if len(str(x)) > 10 else x for x in word_count['X']]
        fig = px.bar(word_count, x='Y', y=X, orientation='h',
                     hover_name=[(str(x[0]) + '<br>' + str(x[1]) + ' words') for x in word_count.values],
                     text=[(str(x[1]) + ' words') for x in word_count.values], labels={'X': '', 'Y': ''})
        fig['data'][0]['hovertemplate'] = '<b>%{hovertext}</b><extra></extra>'
        fig.update_layout(
            title={'text': 'Most Used Words', 'y': 0.95, 'x': 0.5, 'xanchor': 'center',
                   'yanchor': 'top'}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=50, b=50),
            yaxis=dict(autorange='reversed'),
            showlegend=False)
        return fig

    except Exception as e:
        with open('errors.txt', 'a') as f:
            f.write(str(e) + ' - def word_counts')
            f.write('\n')
        return {}


# Bar --- Fifth graph --- Most tweeting people Freq
@app.callback(Output('user-freq', 'figure'),
              [Input('user_freq_int', 'n_intervals'),
               Input('keyword', 'value'),
               Input('user-dropdown', 'value')])
def user_counts(n_intervals, keyword, user, top_people=10):
    try:
        df = pulling_data(keyword, user)

        user_count = pd.DataFrame(Counter(df.name).most_common(top_people), columns=['X', 'Y'])
        # Graph
        fig = px.bar(user_count, x='Y', y='X', orientation='h',
                     hover_name=[(str(x[0]) + '<br>' + str(x[1]) + ' tweets') for x in user_count.values],
                     text=[(str(x[0])[:10] + '... | ' + str(x[1]) + ' tweets') if len(str(x[0])) > 10 else (
                             str(x[0]) + ' | ' + str(x[1]) + ' tweets') for x in user_count.values],
                     labels={'X': '', 'Y': ''})
        # fig['layout']['yaxis']['autorange'] = 'reversed'
        fig['data'][0]['hovertemplate'] = '<b>%{hovertext}</b><extra></extra>'
        fig.update_layout(yaxis=dict(autorange='reversed', visible=False),
                          title={'text': 'Most Tweeting People', 'y': 0.95, 'x': 0.5, 'xanchor': 'center',
                                 'yanchor': 'top'}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          margin=dict(l=50, r=0, t=50, b=50), showlegend=False)
        return fig

    except Exception as e:
        with open('errors.txt', 'a') as f:
            f.write(str(e) + ' - def user_counts')
            f.write('\n')
        return {}


# Calculating time passed from tweets
def time_counter(given):
    secs = (datetime.datetime.now() - given).total_seconds()
    if secs < 60:
        secs = str(round(secs)) + 's'
    elif secs < 3600:
        secs = str(round(secs / 60)) + 'm'
    elif secs < 86400:
        secs = str(round(secs / 3600)) + 'h'
    elif secs < 604800:
        secs = str(round(secs / 86400)) + 'd'
    elif secs < 2419200:
        secs = str(round(secs / 604800)) + 'w'
    else:
        secs = str(round(secs / 2419200)) + 'm'
    return secs


# Encoding svg to use in dash
def svg_to_dash(svg_file):
    encoded = base64.b64encode(open(svg_file, 'rb').read()).decode()
    return 'data:image/svg+xml;base64,{}'.format(encoded)


# Generating Table for Fifth graph --- Twitter style / UI


def generate_table(df):
    a = []
    for i in df.values.tolist():
        a.append(
            html.Section(className='home', children=[
                html.Div(className='', children=[
                    html.Main(className='auth-center', children=[
                        html.Div(className='auth-center-tweets', children=[
                            html.Div(className='tweet-card', children=[
                                html.Div(className='tweet-card-head', children=[
                                    html.Div(className='tweet-head-title', children=[
                                        html.Div(className='tweet-head-icon', children=[
                                            # svg for category logo
                                        ]),
                                        html.Div(className='tweet-head-name', children=[
                                            html.Span(i[8])  # category name will be changed to tweet source
                                        ])
                                    ])]),
                                html.Div(className='tweet-card-body', children=[
                                    html.Div(className='tweet-body-pic', children=[
                                        html.Img(src=i[10])]),
                                    html.Div(className='tweet-body-info', children=[
                                        html.Div(className='tweet-body-info-user', children=[
                                            html.Div(className='tweet-info-user-credits', children=[
                                                html.Div(className='user-info-name', children=[
                                                    html.Div(className='user-info-fullname',
                                                             children=[html.A(
                                                                 href='https://twitter.com/{}/status/{}'.format(i[4],
                                                                                                                i[6]),
                                                                 target='_blank', children=[html.Span(i[3])])]),
                                                    html.Div(className='user-info-username', children=[
                                                        html.A(
                                                            href='https://twitter.com/{}/status/{}'.format(i[4], i[6]),
                                                            target='_blank', children=[html.Span('@' + i[4])])]),
                                                    html.Div(className='user-info-timestamp', children=[html.A(
                                                        href='https://twitter.com/{}/status/{}'.format(i[4], i[6]),
                                                        target='_blank',
                                                        children=[html.Span(time_counter(i[12]))])])
                                                ])]),
                                            # there is an svg for more button
                                            html.Div(className='tweet-info-user-more', children=[
                                                html.Div(className='user-more-btn', children=[
                                                    html.Img(className='more-svg',
                                                             src=svg_to_dash('assets/img/dots.svg'))])
                                            ])
                                        ]),
                                        html.Div(className='tweet-body-info-text', children=[html.P(i[5])]),
                                        html.Div(className='tweet-body-info-comments', children=[
                                            html.Div(className='tweet-reply', children=[
                                                html.Div(className='reply-icon', children=[
                                                    html.Img(className='reply-svg',
                                                             src=svg_to_dash('assets/img/reply.svg'))]),
                                                html.Div(className='reply-counter', children=[html.Span('0')])]),
                                            html.Div(className='tweet-retweet', children=[
                                                html.Div(className='retweet-icon', children=[
                                                    html.Img(className='retweet-svg',
                                                             src=svg_to_dash('assets/img/retweet.svg'))]),
                                                html.Div(className='retweet-counter', children=[html.Span('0')])]),
                                            html.Div(className='tweet-like', children=[
                                                html.Div(className='like-icon', children=[
                                                    html.Img(className='like-svg',
                                                             src=svg_to_dash('assets/img/like.svg'))]),
                                                html.Div(className='like-counter', children=[html.Span('0')])]),
                                            html.Div(className='tweet-share', children=[
                                                html.Div(className='share-icon', children=[
                                                    html.Img(className='share-svg',
                                                             src=svg_to_dash('assets/img/share.svg'))])])
                                        ])
                                    ])
                                ])
                            ])
                        ])
                    ])
                ])
            ])
        )
    return a


# Table or UI --- Fifth graph
@app.callback(Output('recent-tweets', 'children'),
              [Input('text_tweet_update', 'n_intervals'),
               Input('keyword', 'value'),
               Input('user-dropdown', 'value')])
def recent_tweets(n_intervals, keyword, user):
    try:
        df = pulling_data(keyword, user, rows=3)
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms') + datetime.timedelta(hours=4)

        return generate_table(df)
    except Exception as e:
        with open('errors.txt', 'a') as f:
            f.write(str(e) + ' - def user_counts')
            f.write('\n')
        return {}


server = app.server
# dev_server = app.run_server

if __name__ == '__main__':
    app.run_server(debug=True)
