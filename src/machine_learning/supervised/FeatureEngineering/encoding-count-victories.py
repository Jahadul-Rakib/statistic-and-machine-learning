from sklearn.feature_extraction.text import CountVectorizer

data = ['I Love Bangladesh',
        'My name is Rakib',
        'I am a Java developer',
        'My Father Name is Tariqual Islam']

# CountVectorizer will use when my dataset will become a meaningful sentence
cv = CountVectorizer()
# transform data to row and column
x = cv.fit_transform(data)
# get feature name from text dataset
feature = cv.get_feature_names()

x_array = x.toarray()