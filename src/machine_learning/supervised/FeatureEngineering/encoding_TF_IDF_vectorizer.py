from sklearn.feature_extraction.text import TfidfVectorizer

data = ['I Love Bangladesh',
        'My name is Rakib',
        'I am a Java developer',
        'My Father Name is Tariqual Islam']


# TF --term frequency (number of repetition of word in sentence/ total word in sentence
# IDF --Inverse Document Frequency (log total number of sentence/ number of sentence contain in the word)

tfidf = TfidfVectorizer()
x_train = tfidf.fit_transform(data)
x_train.toarray()