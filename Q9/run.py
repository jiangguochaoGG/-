import os
import random
from collections import Counter
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import NaiveBayesClassifier, classify

stoplist = stopwords.words('english')
data_dir = os.path.split(os.path.realpath(__file__))[0]
data_path = [os.path.join(data_dir, 'data', 'enron{}'.format(i)) for i in range(1, 7)]
random.seed(4396)

def init_lists(data_path):
    key_list = []
    file_content = os.listdir(data_path)
    for a_file in file_content:
        try:
            with open(data_path + a_file, 'r', encoding='utf-8') as f:
                key_list.append(f.read())
        except:
            continue
    return key_list

def preprocess(sentence):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(sentence)]

def get_features(text, setting):
    if setting=='bow':
        return {word: count for word, count in Counter(preprocess(text)).items() if not word in stoplist}
    else:
        return {word: True for word in preprocess(text) if not word in stoplist}

def train(features, samples_proportion):
    train_size = int(len(features) * samples_proportion)
    # initialise the training and test sets
    train_set, test_set = features[:train_size], features[train_size:]
    print ('Training set of size= ' + str(len(train_set)) + ' mails')
    print ('Test set of size = ' + str(len(test_set)) + ' mails')
    # train the classifier
    classifier = NaiveBayesClassifier.train(train_set)
    return train_set, test_set, classifier

def evaluate(train_set, test_set, classifier):
    # test accuracy of classifier on training and test set
    print ('Training set accuracy = ' + str(classify.accuracy(classifier, train_set)))
    print ('Test set accuracy = ' + str(classify.accuracy(classifier, test_set)))
    # check most informative words for the classifier
    classifier.show_most_informative_features(20)

if __name__ == '__main__':
    # initialise the data
    spam, ham = [], []
    for path in data_path:
        spam += init_lists(path+'/spam/')
        ham += init_lists(path+'/ham/')
    all_mails = [(mail, 'spam') for mail in spam]
    all_mails += [(mail, 'ham') for mail in ham]
    random.shuffle(all_mails)
    print ('Corpus of size = ' + str(len(all_mails)) + ' mails')

    # extract the features
    all_features = [(get_features(mail, ''), label) for (mail, label) in all_mails]
    print ('Fetched ' + str(len(all_features)) + ' feature sets')

    # train the classifier
    train_set, test_set, classifier = train(all_features, 0.8)

    # evaluate performance
    evaluate(train_set, test_set, classifier)