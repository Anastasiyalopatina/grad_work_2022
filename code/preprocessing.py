import pandas as pd
import re
from tqdm import tqdm
import nltk
import spacy
#nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


def text_cleaning (df, lang):

    for i in tqdm(range(len(df))):
        sentence = str(df[i])

        # normalization
        sentence = re.sub(r'https?://\S+|www\.\S+', r'', sentence) # URLS
        sentence = re.sub(r'@\w+','', sentence) # mentions
        sentence = re.sub(r'#\w+','', sentence) # hash
        sentence = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b','', sentence) # emails
        sentence = re.sub(r'\d+', '', sentence).strip() # number
        sentence = re.sub(r'[^\w\s\d]','', sentence) # puctuations
        sentence = re.sub(r'\s+',' ', sentence).strip() # space
        sentence = sentence.replace('\n', ' ')
        sentence = sentence.replace('\t', ' ')
        sentence = sentence.lower()

        if lang == 'russian':
            sentence = re.sub('[^а-яА-ЯёЁ]+', ' ', sentence)
        elif lang == 'english':
            sentence = re.sub('[^A-Za-z]+', ' ', sentence)

        # stop words removing
        STOPWORDS = set(stopwords.words(lang))
        sentence = ' '.join([x for x in sentence.split() if (x not in STOPWORDS) and (len(x) > 2)])
        sentence = sentence.strip()

        df[i] = sentence

    return df


def lemmatization (df, lang):

    if lang == 'russian':
        lemmatizer = spacy.load("ru_core_news_sm")
        for i in tqdm(range(len(df))):

            df[i] = [word.lemma_ for word in lemmatizer(df[i])]

            # tokenization
            # df[i] = nltk.word_tokenize(df[i])

    elif lang == 'english':
        #nltk.download('wordnet')
        lemmatizer = WordNetLemmatizer()
        for i in tqdm(range(len(df))):

            # tokenization
            df[i] = nltk.word_tokenize(df[i])

            df[i] = [lemmatizer.lemmatize(word) for word in df[i]]

    return df
    
