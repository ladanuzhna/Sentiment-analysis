import re
from nltk.tokenize import word_tokenize

# Bayesian_Net  is responsible for saving the distribution of probabilities, maintaining the conditions of classifier
class Bayesian_Net:
    pos_rev = 0  # number of positive reviews
    neg_rev = 0  # number of negative reviews
    positive_class = {}  # word:count
    negative_class = {}  # word : count
    total_words_pos = 0
    total_words_neg = 0
    V = {}  # general vocabulary
    positive_reviews = []
    negative_reviews = []
    neutral_reviews = []
    P_pos = 0
    P_neg = 0

    def __init__(self, lines, sentiment, pos_tweets, neg_tweets):
        self.twitter_samples_training(pos_tweets,neg_tweets)
        self.split_reviews(lines, sentiment)
        self.pos_rev = len(self.positive_reviews)
        self.neg_rev = len(self.negative_reviews)
        self.bag_of_words(self.positive_reviews, self.positive_class)
        self.bag_of_words(self.negative_reviews, self.negative_class)
        self.merge_vocab()
        self.total_words_pos = len(self.positive_class)
        self.total_words_neg = len(self.negative_class)
        self.P_pos = self.pos_rev / (self.pos_rev + self.neg_rev)
        self.P_neg = self.neg_rev / (self.pos_rev + self.neg_rev)

    #Uses nltk corpus of positive and negative tweets to train the Bayesian net
    def twitter_samples_training(self, pos_tweets,neg_tweets):
        for pos in pos_tweets:
            self.positive_reviews.append(pos)
        for neg in neg_tweets:
            self.negative_reviews.append(neg)


    # Reads the input lines, splitting reviews as positive and negative
    def split_reviews(self, lines, sentiment):
        for i, sent in enumerate(sentiment):
            if sent == 5 or 4 or 3:
                rev = re.sub('[0-9|]', '', lines[i])
                self.positive_reviews.append(rev)
            elif sent == 2 or 1:
                rev = re.sub('[0-9|]', '', lines[i])
                self.negative_reviews.append(rev)
            else:
                rev = re.sub('[0-9|]', '', lines[i])
                self.neutral_reviews.append(rev)

    # Initializes bag of words of positive and negative reviews
    def bag_of_words(self, reviews_, class_):
        for rev in reviews_:
            words = word_tokenize(rev)
            for word in words:
                if word not in class_.keys():
                    class_[word] = 1
                else:
                    class_[word] += 1

    # Gets only the most frequent words, cleaning both bags of words
    # Merges positive and negative bags of words to get the general vocabulary
    def merge_vocab(self):
        for word in self.positive_class.keys():
            self.V[word] = self.positive_class[word]

        for word in self.negative_class.keys():
            if word in self.V.keys():
                self.V[word] = self.V[word] + self.negative_class[word]
            else:
                self.V[word] = self.negative_class[word]

    # Helper function for mapping the input document (string) to the class
    def c_map(self, doc):
        positivity = self.P_doc('5', doc) * self.P_pos
        negativity = self.P_doc('1', doc) * self.P_neg
        if positivity >= negativity:
            return '5'
        else:
            return '1'

    # Calculates probability of doc belonging to the given class
    def P_doc(self, class_, doc):
        P = 1
        for word in word_tokenize(doc):
            P = P * self.P_word(class_, word)
        return P

    # Calculates the probability of word belonging to the particular class
    def P_word(self, class_, word):
        if class_ == '5':
            if word in self.positive_class.keys():
                return self.positive_class[word] + 1 / (self.total_words_pos + len(self.V))
            else:
                return 1 / (self.total_words_pos + len(self.V))
        elif class_ == '1':
            if word in self.negative_class.keys():
                return self.negative_class[word] + 1 / (self.total_words_neg + len(self.V))
            else:
                return 1 / (self.total_words_neg + len(self.V))
