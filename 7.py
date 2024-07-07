import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
msg = pd.read_csv('naivetext.csv', names=['message', 'label'])
print("The dimension of the dataset", msg.shape)
msg['labelnum'] = msg.label.map({'pos': 1, 'neg': 0})
X = msg.message
y = msg.labelnum
print(X)
print(y)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
print("\nThe total number of training data", ytrain.shape)
print("\nThe total number of testing data", ytest.shape)
count_vect = CountVectorizer()
Xtrain_dtm = count_vect.fit_transform(Xtrain)
Xtest_dtm = count_vect.transform(Xtest)
print("\nThe words or tokens in text document\n")
print(count_vect.get_feature_names())
df = pd.DataFrame(Xtrain_dtm.toarray(), columns=count_vect.get_feature_names())
clf = MultinomialNB().fit(Xtrain_dtm, ytrain)
predicted = clf.predict(Xtest_dtm)
print("\nAccuracy of Classifier", metrics.accuracy_score(ytest, predicted))
print("\nConfusion Matrix")
print(metrics.confusion_matrix(ytest, predicted))
print("\nThe value of precision", metrics.precision_score(ytest, predicted))
print("\nThe value of recall", metrics.recall_score(ytest, predicted))

#output
