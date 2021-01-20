import pandas as pd

df=pd.read_csv('tweets_trump.csv')

import re
import numpy as np
import pandas as pd
from pprint import pprint


# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization ( analyser les texts)
import spacy


import nltk
from nltk.corpus import wordnet
nltk.download('stopwords')
nltk.download('words')

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

pd.set_option('display.max_colwidth', -1)

# enlever les tweets doubleé

df.drop_duplicates(inplace=True)

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt

# Utiliser la formulaire 're' pour nettoyer les données
def clean_tweets(lst):

    lst = np.vectorize(remove_pattern)(lst, "RT @[\w]*:")

    lst = np.vectorize(remove_pattern)(lst, "@[\w]*")

    lst = np.vectorize(remove_pattern)(lst, "https?://[A-Za-z0-9./]*")

    lst = np.core.defchararray.replace(lst, "[^a-zA-Z#]", " ")

    return lst
df['Tweet']=clean_tweets(df['Tweet'])

print(df.head(5))
'''
def return_lis(lst):
    lis=clean_tweets(lst)
    data = lis.tolist()


    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]


    data = [re.sub(r"(?<!\\)\\n|\n", ' ', sent) for sent in data]


    data = [re.sub("\'", "", sent) for sent in data]

    data = [re.sub(r'(?:\b\w{,1}\s|\s\w{,1}\b|\b\w{,1}\b)', ' ', sent) for sent in data]


    data = [re.sub('\s+', ' ', sent) for sent in data]
    return data

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

lis=clean_tweets(df['Tweet'])
data=return_lis(lis)
data_words = list(sent_to_words(data))
# Construire les models de bigram et trigram
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)


bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)



# Utilise NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'nan', '&amp', 'amp', 'xaa', 'xac', 'xa', 'xb', 'xc', 'xf', 'xe', 'co'])
def remove_stopwords(texts):

    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):

    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out
data_words_nostops = remove_stopwords(data_words)

# Construire les Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Construire le modèle en anglais sur  spacy '
nlp = spacy.load("en_core_web_sm", disable=["parser",'ner'])

# Garder les noun, adjective, verb...
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

id2word = corpora.Dictionary(data_lemmatized)

texts = data_lemmatized



corpus = [id2word.doc2bow(text) for text in texts]


#Construire le modèle LDA

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=4,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)

pyLDAvis.show(vis)




#modele mallet

# Compute Coherence Score
coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('\nCoherence Score: ', coherence_ldamallet)


def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=data)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
df_dominant_topic.head(10)
'''