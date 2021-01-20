#la procecing
from nltk.collections import *
import pandas as pd
import nltk
import re
from gensim import corpora, models
import pyLDAvis.gensim
from pprint import pprint
df = pd.read_json('trump_tweets_2020_new.json')
df['created_month']=[str(t)[:7] for t in df['created_at']]

# retweet
df_original = df[~df.is_retweet].copy()
df_original['created_month'] = [str(t)[:7] for t in df_original['created_at']]
created_month = sorted({str(t)[:7] for t in df_original['created_at']})
subtweetsDF = df_original[['text','created_month']]
subtweetsDict = subtweetsDF.groupby('created_month')['text'].apply(list).to_dict()
print("No of documents before data preprocessing: %d" % len(subtweetsDict.keys()))
# enlève les mots qui a peu de fréquence
filtered1stsubtweetsDict = {}
for k in subtweetsDict.keys():
    if len(subtweetsDict[k]) >= 10:
        filtered1stsubtweetsDict[k] = " ".join(subtweetsDict[k])

#enlève URLs
for k in filtered1stsubtweetsDict.keys():
    filtered1stsubtweetsDict[k] = re.sub(r"(?:\@|http?\://)\S+", "", filtered1stsubtweetsDict[k])

filtered2ndsubtweetsDict = {}
for k in filtered1stsubtweetsDict.keys():
    if len(filtered1stsubtweetsDict[k]) >= 1000:
        filtered2ndsubtweetsDict[k] = filtered1stsubtweetsDict[k]


#tokens procecing:
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
for k in filtered2ndsubtweetsDict.keys():
    filtered2ndsubtweetsDict[k] = tokenizer.tokenize(filtered2ndsubtweetsDict[k].lower())
#remove stopwords and statistically insignificant tokens
stoplist_tw=['amp','get','got','hey','hmm','hoo','hop','iep','let','ooo','par',
            'pdt','pln','pst','wha','yep','yer','aest','didn','nzdt','via',
            'one','com','new','like','great','make','top','awesome','best',
            'good','wow','yes','say','yay','would','thanks','thank','going',
            'new','use','should','could','best','really','see','want','nice',
            'while','know','https']
unigrams = list(set([ w for doc in filtered2ndsubtweetsDict.values() for w in doc if len(w)==1]))

bigrams  = list(set([ w for doc in filtered2ndsubtweetsDict.values() for w in doc if len(w)==2]))

stoplist  = set(nltk.corpus.stopwords.words("english") + stoplist_tw + unigrams + bigrams)

for k in filtered2ndsubtweetsDict.keys():
    filtered2ndsubtweetsDict[k] = [token for token in filtered2ndsubtweetsDict[k] if token not in stoplist]

token_frequency = defaultdict(int)
#par mois
for doc in filtered2ndsubtweetsDict.values():
# par mot
    for token in doc:
        token_frequency[token] += 1
for k in filtered2ndsubtweetsDict.keys():
    filtered2ndsubtweetsDict[k] = [token for token in filtered2ndsubtweetsDict[k] if token_frequency[token] > 1]


# construire dictionnaire pour le modèle
for k in filtered2ndsubtweetsDict.keys():
    filtered2ndsubtweetsDict[k].sort()


dictionaries = {}

#
dictionaries['all'] = corpora.Dictionary(filtered2ndsubtweetsDict.values())

#on donne un id pour tous les tokens
dictionaries['all'].compactify()

# construire dictionnaire par moi
for k in filtered2ndsubtweetsDict.keys():
    #tous les mois, il y a un valeur de tokens
    dictionaries[k] = corpora.Dictionary([filtered2ndsubtweetsDict[k]])
    #tous les tokens ont un id
    dictionaries[k].compactify()


#construire  un corpus pour touts les mois
corpora = {}

corpora['all'] = [dictionaries['all'].doc2bow(doc) for doc in filtered2ndsubtweetsDict.values()]
for k in filtered2ndsubtweetsDict.keys():

    corpora[k] = [dictionaries[k].doc2bow(filtered2ndsubtweetsDict[k])]


# modeling

#select hyperparameters
lda_params = {'num_topics': 3, 'passes': 10, 'alpha': 0.01}
print("Training LDA models with: %s  " % lda_params)
LDAs = {}
for k in corpora.keys():
    LDAs[k] = models.LdaModel(corpora[k],
                    id2word=dictionaries[k],
                    num_topics=lda_params['num_topics'],
                    passes=lda_params['passes'],
                    alpha = lda_params['alpha'])

# LDA model for all documents
# le modèle LDA, le matrice de vocabulaire, dictionnaire( tous les mots sans ID)
all_docs_topics =  pyLDAvis.gensim.prepare(LDAs['all'], corpora['all'], dictionaries['all'])
pprint(LDAs['all'].print_topics())
pyLDAvis.show(all_docs_topics)




