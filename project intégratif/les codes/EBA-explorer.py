from nltk.tokenize import TweetTokenizer
import pandas as pd
from nltk.corpus import stopwords
from nltk import FreqDist
import string
import matplotlib
from wordcloud import WordCloud,ImageColorGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from imageio import imread
import numpy as np
import chart_studio.plotly as py
import plotly.graph_objs as go
from nltk.collections import *
import nltk
from textblob import TextBlob
import pandas as pd
import re
df = pd.read_json('t.json')
df_original = df[~df.is_retweet].copy()
df_original['created_month'] = [str(t)[:7] for t in df_original['created_at']]
created_month = sorted({str(t)[:7] for t in df_original['created_at']})
punctiuation = list(string.punctuation)
stop = stopwords.words('english') + punctiuation
tknzr = TweetTokenizer()
# séparer les mots , outiles: tweettokenizer
# retourné un liste de text séparé avec les mots


def tokenizer_tweets(df):

    text = ''
    for t in df['text']:
        text += t
    tokens = [i.lower() for i in tknzr.tokenize(text)]

    return tokens

# nettoyer les textes,les nombre, pour avoir que des vocabulaire
# outils utilisé: nltk , stopwords
# retourné les texts néttoyés
def clear_tokens(tokens,stop):

    tokens_cl = [t for t in tokens if (len(t) >= 3)
                 and (not t.startswith(('#', '@')))
                 and (not t.startswith('http'))
                 and (t not in stop)
                 and (t[0].isalpha())]

    return tokens_cl

# afficher les mots fréquents
def get_top10_of(tokens,i, n=10):

    resulat=FreqDist([t for t in tokens if t.startswith(i)]).most_common(n)

    #transmettre en dictionnaire
    resulat_dis={k:v for k,v in dict(resulat).items()}
    return resulat_dis

def bigfrequence(df_original):
    df_original.index = df_original.created_at
    tokens_original = clear_tokens(tokenizer_tweets(df_original),stop)
    bgs = nltk.bigrams(tokens_original)
    fdist = nltk.FreqDist(bgs)
    bigram_fq = fdist.most_common()
    bigram_fq_25 = {k: v for k, v in dict(bigram_fq[:25]).items()}
    return bigram_fq_25

def talk_about(monthly_tweets,name):

    name = name.lower()
    name_mentions_frequency = {}
    for m in monthly_tweets.keys():
        count = 0
        for t in monthly_tweets[m]:
            for i in t.split(' '):
                if i == name:
                    count += 1

        name_mentions_frequency[m] = ((count) / (len(monthly_tweets[m]))) * 100

    return name_mentions_frequency



def clean_tweet(tweet):

    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def get_tweet_sentiment(tweet):
 #Function to classify sentiments of passed tweets using TextBlob's sentiment method
    analysis = TextBlob(clean_tweet(tweet))
    #set sentiments
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

def emotion_get(df_original,created_month):
    #ajoute une
    df_original['sentiment']=[get_tweet_sentiment(t) for t in df_original.text]
    df_original['created_month'] = [str(t)[:7] for t in df_original['created_at']]
    monthly_tweets_emotion = {}
    for m in created_month:
        emotion = [t for t in df_original[(df_original.created_month == m)].sentiment]
        monthly_tweets_emotion[m] = emotion
    return monthly_tweets_emotion

def talk_neg(monthly_tweets_emotion,emotion):
    results = {}

    for m in list(monthly_tweets_emotion.keys()):
        counter = 0
        sentiment = monthly_tweets_emotion[m]
        for j in sentiment:
            if j ==emotion :
                counter = counter + 1

        results[m] = (counter / len(sentiment)) * 100

    return results




def get_tokens(tokens_original):
    pos_tokens = [t for t in tokens_original if get_tweet_sentiment(t) == 'Positive']
    neg_tokens = [t for t in tokens_original if get_tweet_sentiment(t) == 'Negative']
    neu_tokens = [t for t in tokens_original if get_tweet_sentiment(t) == 'Neutral']


    dis_tokens=[pos_tokens,neg_tokens,neu_tokens]
    return dis_tokens

def word_cloud(dis_tokens):
    colors=['white','black','grey']
    bg_pic=imread('img.jpg')
    for i in range(len(dis_tokens)):

        wc = WordCloud(background_color=colors[i],
                       stopwords=stop,
                       mask=bg_pic,
                       scale=3,
                       max_words=2000,
                        max_font_size=70,
                        random_state=200
                        ).generate_from_frequencies(FreqDist(dis_tokens[i]))
        image_colors=ImageColorGenerator(bg_pic)
        plt.imshow(wc)
        plt.axis("off")
        plt.show(wc.recolor(color_func=image_colors))




def visualisation(x_list):
    plt.figure(figsize=(20, 8))
    plt.subplots_adjust(wspace=0.4)
    for i in range(len(x_list)):
        x_part=x_list[i][0]
        y_part=x_list[i][1]
        if i==2:
            break
        else:
            plt.subplot(1,2,i+1)
        plt.style.use('ggplot')
        colors=['pink','lightskyblue','grey']
        titles=["Top 10 # dans tweets de Trump","Top 10 @ dans tweets de Trump",
               "Fréquence de Bigram"]
        plt.barh(range(len(x_part)), y_part, height=0.7, color=colors[i], alpha=0.8)
        plt.yticks(range(len(x_part)), x_part)
        plt.xlabel("Quantité")
        plt.tick_params(axis='y', colors='red')
        plt.grid('x')
        plt.title(titles[i])
    plt.show()

    monthly_tweets_emotion = emotion_get(df_original, created_month)

    pos = talk_neg(monthly_tweets_emotion,'Positive')
    nega = talk_neg( monthly_tweets_emotion,'Negative')
    neutral = talk_neg( monthly_tweets_emotion,'Neutral')
    x = list(pos.keys())
    y = list(pos.values())

    x1 = list(nega.keys())
    y1 = list(nega.values())

    x2 = list(neutral.keys())
    y2 = list(neutral.values())


    plt.figure(figsize=(10, 8))
    plt.style.use('grayscale')
    plt.plot(x, y, '-', color='coral', label='Positive')
    plt.plot(x, y1, '-', color='lightgreen', label='Negative')
    plt.plot(x, y2, '-', color='grey', label='Neutral')
    plt.xlabel('mois')
    plt.ylabel('% de tweets')
    plt.title(" les tweets positive, négative et neutral")
    plt.grid(True)
    plt.legend()
    plt.show()


    monthly_tweets = {}
    for m in created_month:
        tweets = [t.lower() for t in df_original[(df_original.created_month == m)].text]
        monthly_tweets[m] = tweets

    plt.figure(figsize=(10,8))

    hillary_mentions_fq = talk_about(monthly_tweets,'Biden')

    x = list(hillary_mentions_fq.keys())
    y = list(hillary_mentions_fq.values())

    plt.style.use('ggplot')
    plt.plot(x, y, '-', color='r')
    plt.xlabel('mois')
    plt.ylabel('% de tweets')
    plt.title(" Tweets de Trumps parlant de Joe Biden")
    plt.grid(True)
    plt.show()



def main():
    df = pd.read_json('trump_tweets_2020_new.json')
    df_original = df[~df.is_retweet].copy()
    # les top 10 hastag  et les top 10 
    tokens=tokenizer_tweets(df)
    tokens_original=clear_tokens(tokenizer_tweets(df_original),stop)

    data=get_top10_of(tokens,'#')
    data1=get_top10_of(tokens,'@')
    data2=bigfrequence(df_original)

    x=list(data.keys())
    y=list(data.values())
    x1=list(data1.keys())
    y1=list(data1.values())
    x2=list(data2.keys())
    y2=list(data2.values())
    x_list=[(x,y),(x1,y1),(x2,y2)]

    visualisation(x_list)
    dis_tokens=get_tokens(tokens_original)
    word_cloud(dis_tokens)













