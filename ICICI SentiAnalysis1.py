#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# In[51]:


# Reading the data from the CSV file
df = pd.read_excel('C:/Users/subnarasimhan/Desktop/Sentimental Analysis/CompanyNewsHeadlines_ICICI_Consolidated.xls')


# In[52]:


# Extracting features (tweet_text) and labels (Azure label)
X = df['tweet_text']
y = df['Azure label']


# In[53]:


X.head()


# In[54]:


y.head()


# In[55]:


# Converting labels to numerical values using LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


# In[56]:


# Spliting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[57]:


# Converting text data to numerical features using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


# In[58]:


# Training a machine learning model (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_vectorized, y_train)


# In[59]:


# Making predictions on the test set
y_pred = model.predict(X_test_vectorized)


# In[60]:


# Ensuring the vocabulary size is the same for both training and testing data
# X_test_vectorized = vectorizer.transform(X_test)


# In[61]:


# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


# In[62]:


# Printing classification report
print('\nClassification Report:')
print(classification_report(y_test, y_pred))


# In[63]:


# Predicting the sentiment for a new data
new_data = ["Some positive news about the company", "Bad"]
new_data_vectorized = vectorizer.transform(new_data)
predictions = model.predict(new_data_vectorized)


# In[64]:


# Converting predictions back to sentiment labels using inverse_transform
predicted_sentiments = label_encoder.inverse_transform(predictions)


# In[65]:


# Printing predicted sentiments for our new data
print('\nPredicted Sentiments for our New Data:')
for data, sentiment in zip(new_data, predicted_sentiments):
    print(f'Text: "{data}" | Predicted Sentiment: {sentiment}')


# In[ ]:





# In[ ]:




