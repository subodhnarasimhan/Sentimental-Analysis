
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_excel('C:/Users/admin/Desktop/Sentiment-Analysis-and-Stock-Values/CompanyNewsHeadlines_ICICI_Consolidated.xls')

X = df['tweet_text']
y = df['Azure label']

X.head()

y.head()

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_vectorized, y_train)

y_pred = model.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

print('\nClassification Report:')
print(classification_report(y_test, y_pred))

new_data = ["Some positive news about the company", "Bad"]
new_data_vectorized = vectorizer.transform(new_data)
predictions = model.predict(new_data_vectorized)

predicted_sentiments = label_encoder.inverse_transform(predictions)

print('\nPredicted Sentiments for our New Data:')
for data, sentiment in zip(new_data, predicted_sentiments):
    print(f'Text: "{data}" | Predicted Sentiment: {sentiment}')
