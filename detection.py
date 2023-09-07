#Importing Libraries
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer

#Creating Sample Positive Texts
positive_texts = [
    "we love you",
    "they love us",
    "you are good",
    "he is good",
    "they love mary"
]

#Creating Sample Negative Texts
negative_texts =  [
    "we hate you", 
    "they hate us",
    "you are bad",
    "he is bad",
    "we hate mary"
]

#Creating Sample Test Texts
test_texts = [
    "they love mary",
    "they are good",
    "why do you hate mary",
    "they are almost always good",
    "we are very bad"
]

#Creating Training Data by Combining Positive and Negative texts in one array
training_texts = negative_texts + positive_texts
training_labels = ["negative"] * len(negative_texts) + ["positive"] * len(positive_texts)

#Vectorizer to convert training texts into vectors
vectorizer = CountVectorizer()
vectorizer.fit(training_texts)
training_vectors = vectorizer.transform(training_texts)
testing_vectors = vectorizer.transform(test_texts)

#Adding the classifiers for prediction
classifier = tree.DecisionTreeClassifier()
classifier.fit(training_vectors, training_labels)
predictions = classifier.predict(testing_vectors)
print(predictions)
