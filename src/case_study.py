import numpy as np
import pandas as pd
import json 
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS,CountVectorizer
import numpy as np
import pandas as pd
import re
import string
import matplotlib.pyplot as plt 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models import CoherenceModel
from nltk.tokenize import word_tokenize
import gensim.corpora as corpora

def build_df(annies_lst):
    df = pd.DataFrame(data = annies_lst, columns = ['year','season','month','state','county'])
    return df

def clean_function(value):
    for idx, val in enumerate(value):
        if val == ':':
            return(value[idx+2:])

def get_article(article):
    soup = BeautifulSoup(article['html'], 'lxml')
    table = soup.find('table', id='Table1')
    info = table.findAll('p')
    i = 0
    year = 0
    season = 0
    month = 0
    state = 0
    county = 0
    while i < len(info) and (year == 0 or season == 0 or month == 0 or state == 0 or county == 0):
        if info[i].get_text() == '':
            i+=1
        elif info[i].get_text()[0] == 'Y':
            year = clean_function(info[0].get_text())
            i += 1 
        elif info[i].get_text()[0:2] == 'SE':
            season = clean_function(info[i].get_text())
            i += 1 
        elif info[i].get_text()[0] == 'M':
            month = clean_function(info[i].get_text())
            i += 1 
        elif info[i].get_text()[0:2] == 'ST':
            state = clean_function(info[i].get_text())
            i += 1 
        elif info[i].get_text()[0] == 'C':
            county = clean_function(info[i].get_text())
            i += 1
        else:
            i += 1
    return [year, season, month, state, county]

def get_content(article):
    soup = BeautifulSoup(article['html'], 'lxml')
    table = soup.find('table', id='Table1')
    # index = int(re.search(r'\d+', (table.find('span', class_='reportheader').get_text())).group())
    index = article['_id']['$oid']
    info = table.findAll('p')
    content = ''
    for i in range(len(info)+1):
        try:
            content += clean_function(info[i].get_text())
        except:
            continue
    try:
        block = table.find('blockquote').get_text().strip('\n')
        content + block
    except:
        return index, content
    return index, content

def display_topics(model, feature_names, num_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-num_top_words - 1:-1]]))

def tokenize(text):
    # docs = [word_tokenize(content) for content in text]
    # docs = [[word for word in words if word not in stop_words] for words in docs]
    # docs_wordnet = [[wordnet.lemmatize(word) for word in words] for words in docs]
    # return docs_wordnet
    return [wordnet.lemmatize(word) for word in word_tokenize(text)]


    

if __name__ == '__main__':
    file_path = 'data/bigfoot_first100records.json'
    file_path2 = 'data/bigfoot_data.json'
    records = []
    with open(file_path) as f:
        for i in f:
            records.append(json.loads(i))
    soup = BeautifulSoup(records[0]['html'], 'lxml')
    keywords = soup.find('meta', {'name':"KEYWORDS"})
    keywords['content']
    content = soup.find

    # all_data = {}
    # for i in range(len(records)):
    #     index, data = get_article(records[i])
    #     all_data[index]= data

    all_content = {}
    for i in range(len(records)):
        index, content = get_content(records[i])
        all_content[index] = content

    corpus = []
    for i in range(len(records)):
        index, content = get_content(records[i])
        corpus.append(content)

    val = []
    for num in range(len(records)-1):
        b = get_article(records[num])
        val.append(b)
    df = build_df(val)

    #TODO stem/lemmatize
    stop_words =set()
    stop_words = stop_words.union({'house','just', 'like', 'did', 'time', 'saw', 'right', 'left', 
                                            'road', 'county', 'year', 'road','said', 'area', 'nt',
                                            'woods', 'heard', '2009', '2012', '2011', '2013', '2009', 'km',
                                            '07', '09', 'didnt', 'got', 'went', 'know'})
    stop_words = stop_words.union(set(df['year']))
    stop_words = stop_words.union(set(df['season']))
    stop_words = stop_words.union(set(df['month']))
    stop_words = stop_words.union(set(df['state']))
    stop_words = stop_words.union(set(df['county']))
    stop_words =ENGLISH_STOP_WORDS.union(stop_words)
    
    lemmer=WordNetLemmatizer()
    
    # tokenized = [word_tokenize(content.lower()) for content in corpus]
    # docs = [[word for word in words if word not in stop_words] for words in tokenized]
   
    # corp_lem = [wordnet.lemmatize(word) for word in word_tokenize(corpus.lower())]
    n_corp =[]
    for i in corpus:
        n_corp.append(re.sub('[^A-Za-z0-9]+', ' ', i).lower())
    corp=[]
    for i in n_corp:
        corp.append(lemmer.lemmatize(i))
    vectorizer = TfidfVectorizer(stop_words =stop_words)
    X = vectorizer.fit_transform(corpus)
    tfidf_words = vectorizer.get_feature_names()

    countvect = CountVectorizer(stop_words=stop_words)
    count_vectorized = countvect.fit_transform(corpus)
    words_count = countvect.get_feature_names()

    lda = LatentDirichletAllocation(n_components= 5, max_iter=5, learning_method='online',random_state=0, n_jobs=-1)
    lda.fit(count_vectorized)
    num_top_words = 10
    display_topics(lda, words_count, num_top_words)


    print('\nPerplexity: ', np.log(lda.perplexity(count_vectorized)))
   