import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from BayesianNet import Bayesian_Net
from bs4 import BeautifulSoup
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import twitter_samples


# Responsible for text maintaining, training and classification
class Bayes_Classifier:
    stop_words = []
    BN = None
    pos_tweets = []
    neg_tweets = []

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.pos_tweets = self.pre_process(twitter_samples.strings('positive_tweets.json'))
        self.neg_tweets = self.pre_process(twitter_samples.strings('negative_tweets.json'))

    def pre_process(self, lines):
        pd.options.mode.chained_assignment = None  # default='warn'
        result = []
        for i,s in enumerate(lines):
            # Clean HTML data
            new = BeautifulSoup(s,features="html.parser")
            # Remove special characters and links
            nourl = re.sub('https?://[A-Za-z0-9./]+', '', str(new))
            nospec = re.sub('[^A-Za-z0-9 |]+', '', nourl)
            # Remove capitalization
            nocap = nospec.lower()
            # Remove stop words
            tokens = word_tokenize(nocap)
            cleaned = " ".join([w for w in tokens if w not in self.stop_words])
            result.append(cleaned)
        return result

    def stemming(self, reviews):
        stemmer = PorterStemmer()
        return [' '.join([stemmer.stem(word) for word in review.split()]) for review in reviews]

    def lemmatizer(self, reviews):
        lem = WordNetLemmatizer()
        return [' '.join([lem.lemmatize(word) for word in review.split()]) for review in reviews]

    def train(self, df):
        pre_processed = self.pre_process(df.text)
        self.BN = Bayesian_Net(pre_processed, str(df.sentiment), self.pos_tweets, self.neg_tweets)

    def classify(self, df):
        result = []
        pre_processed = self.pre_process(df.text)
        for i, s in enumerate(pre_processed):
            s_ = re.sub('[0-9|]', '', pre_processed[i])
            result.append(self.BN.c_map(s_))
        return result
