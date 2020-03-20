import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import ClassifierInterface

class ModelPerfomance:
    df = None

    def load(self):
        cols = ['_unit_id', '_golden', '_unit_state', '_trusted_judgments', '_last_judgment_at',
                'sentiment', 'sentiment:confidence', 'our_id', 'sentiment_gold', 'reason', 'text']
        self.df = pd.read_csv("Twitter-sentiment-self-drive-DFE.csv",header=None, names=cols,encoding ="ISO-8859-1")
        self.df.drop(['_unit_id', '_golden','_unit_state','_trusted_judgments', '_last_judgment_at','our_id','reason',
                      'sentiment_gold'],
                axis=1, inplace=True)

    def extract_actual_sentiment(self, test):
        test_sent = []
        for sent_ in test.sentiment:
            if sent_ == 5 or 4 or 3:
                test_sent.append('5')
            else:
                test_sent.append('1')
        return test_sent

    def evaluate(self):
        train, test = train_test_split(self.df, test_size=0.2)
        classifier = ClassifierInterface.Bayes_Classifier()
        classifier.train(train)
        predicted = classifier.classify(test)
        actual = self.extract_actual_sentiment(test)
        print("Accuracy score = " + accuracy_score(actual,predicted))

if __name__ == "__main__":
    M = ModelPerfomance()
    M.load()
    M.evaluate()

