#!/usr/bin/env python
# coding: utf-8

# In[185]:


####1) Importing the datasets


# In[186]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[187]:


import spacy
nlp = spacy.load('en_core_web_sm')


# In[188]:


df = pd.read_excel('C:/Users/ujjaw/Desktop/Ujjawal/ISB/FP1/CompanyNewsHeadlines_ICICI_Consolidated.xls')
df.head()


# In[189]:


df.head(1500)


# In[190]:


# Drop columns 
columns_to_drop = ['Company', 'Date','Score','TextBlob Label','Comparision']
df = df.drop(columns=columns_to_drop)


# In[191]:


df


# In[192]:


df['Azure label'].value_counts()


# In[193]:


# check for null values
df.isnull().sum()
df=df.dropna()
df = df.reset_index(drop=True)

# no null values in the data


# In[194]:


####2) Data Cleaning
# here we will remove stopwords, punctuations
# as well as we will apply lemmatization


# In[195]:


import re
clean =  re.compile('<.*?>')
re.sub(clean,'',df.iloc[200].tweet_text)


# In[196]:


def clean_html(text):
    clean =  re.compile('<.*?>')
    return  re.sub(clean, '' , text)


# In[197]:


df['tweet_text'] = df['tweet_text'].apply(clean_html)


# In[198]:


##function to remove special character. 

def remove_special(tweet_text):
    if tweet_text is None:
        return None
    x=''

    for  i in tweet_text:
        if i.isalnum():
            x=x+i
        elif i.isspace():
            x=x+' '
        else:
            x=x +  ''
    return x


# In[199]:


df['tweet_text'] = df['tweet_text'].apply(remove_special)


# In[200]:


import re

def remove_non_ascii(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

# Example usage
original_text = "à¤¨à¥à¤•à¤¸à¤¾à¤¨à¥€à¤&^%^%$^šà¥à¤¯à¤¾ à¤¸à¤°à¥à¤"
cleaned_text = remove_non_ascii(original_text)

print("Original Text:", original_text)
print("Cleaned Text:", cleaned_text)


# In[201]:


df['tweet_text'] = df['tweet_text'].apply(remove_non_ascii)


# In[202]:


df


# In[203]:


df.describe()


# In[204]:


from bs4 import BeautifulSoup
import re


# In[205]:


df['tweet_text'] = df['tweet_text'].apply(lambda x: BeautifulSoup(x).get_text())
df['tweet_text'] = df['tweet_text'].apply(lambda x: re.sub(r'http\S+', '', x))


# In[206]:


import re

def contractions(s):
    s = re.sub(r"won't", 'will not', s)
    s = re.sub(r"wouldn't", 'would not', s)
    s = re.sub(r"couldn't", 'could not', s)
    s = re.sub(r"can't", 'can not', s)
    s = re.sub(r"n't", ' not', s)
    s = re.sub(r"'d", ' would', s)
    s = re.sub(r"'re", ' are', s)
    s = re.sub(r"'s", ' is', s)
    s = re.sub(r"'ll", ' will', s)
    s = re.sub(r"'t", ' not', s)
    s = re.sub(r"'ve", ' have', s)
    s = re.sub(r"'m", ' am', s)
    return s

# Assuming 'tweet_text' is the column containing text data in your DataFrame
df['tweet_text'] = df['tweet_text'].apply(contractions)


# In[207]:


df


# In[208]:


import nltk


# In[209]:


df['tweet_text']=df['tweet_text'].apply(lambda x: ' '.join([re.sub('[^A-Za-z]+','', x) for x in nltk.word_tokenize(x)]))


# In[210]:


df


# In[211]:


df['tweet_text']=df['tweet_text'].apply(lambda x: re.sub(' +', ' ', x))


# In[212]:


df


# In[213]:


df['tweet_text']


# In[214]:


import string


# In[215]:


punct = string.punctuation


# In[216]:


punct


# In[217]:


from spacy.lang.en.stop_words import STOP_WORDS


# In[218]:


stopwords = list(STOP_WORDS) # list of stopwords


# In[219]:


# creating a function for data cleaning


# In[220]:


def text_data_cleaning(sentence):
  doc = nlp(sentence)

  tokens = [] # list of tokens
  for token in doc:
    if token.lemma_ != "-PRON-":
      temp = token.lemma_.lower().strip()
    else:
      temp = token.lower_
    tokens.append(temp)
 
  cleaned_tokens = []
  for token in tokens:
    if token not in stopwords and token not in punct:
      cleaned_tokens.append(token)
  return cleaned_tokens


# In[221]:


# if root form of that word is not pronoun then it is going to convert that into lower form
# and if that word is a proper noun, then we are directly taking lower form, because there is no lemma for proper noun


# In[222]:


text_data_cleaning("Hello all, It's a beautiful day outside there!")
# stopwords and punctuations removed


# In[223]:


df['tweet_text'] = df['tweet_text'].apply(text_data_cleaning)


# In[224]:


df['tweet_text']


# In[225]:


# Perform Stemming
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[226]:


y = []
def stem_words(text):
    for i in text:
        y.append(ps.stem(i))
    z=y[:]
    y.clear()
    return z
    


# In[227]:


df['tweet_text'] = df['tweet_text'].apply(stem_words)


# In[228]:


def join_back(list_input):
    return " ".join(list_input)


# In[229]:


df['tweet_text'] = df['tweet_text'].apply(join_back)


# In[230]:


df['tweet_text']


# In[231]:


# This is the clean and processed data.
df


# In[232]:


#We will not Split train and test data .

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train, Y_test = train_test_split(df['tweet_text'], df['Azure label'], test_size=0.25, random_state=30)
print('Train: ',X_train.shape,Y_train.shape,'Test: ',(X_test.shape,Y_test.shape))


# In[233]:


##RANDOM FOREST


# In[234]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# In[235]:


vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# In[236]:


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, Y_train)


# In[237]:


# Predictions
predictions = rf_model.predict(X_test_tfidf)


# In[238]:


# Evaluation
print(f"Accuracy: {accuracy_score(Y_test, predictions)}")
print(classification_report(Y_test, predictions))


# In[239]:


###SVM


# In[240]:


from sklearn.svm import LinearSVC
clf = LinearSVC(random_state=0)


# In[241]:


clf.fit(X_train_tfidf,Y_train)


# In[242]:


y_test_pred=clf.predict(X_test_tfidf)


# In[243]:


from sklearn.metrics import classification_report
report=classification_report(Y_test, y_test_pred,output_dict=True)


# In[244]:


print(classification_report(Y_test, y_test_pred))


# In[245]:


##### Logistic Regression


# In[246]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=1000,solver='saga')


# In[247]:


clf.fit(X_train_tfidf,Y_train)


# In[248]:


y_test_pred=clf.predict(X_test_tfidf)


# In[249]:


from sklearn.metrics import classification_report
report=classification_report(Y_test, y_test_pred,output_dict=True)


# In[250]:


print(classification_report(Y_test, y_test_pred))


# In[251]:


##Naive Bayes


# In[252]:


#from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#from sklearn.pipeline import make_pipeline





# In[253]:


model = MultinomialNB()


# In[254]:


from sklearn.metrics import classification_report


# In[255]:


# Train the model on the training data

model.fit(X_train_tfidf, Y_train)

# Make predictions on the test data
labels = model.predict(X_test_tfidf)


# Evaluate the model
print(classification_report(Y_test, labels))


# In[256]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB


# In[257]:


dense_tf_x_train = X_train_tfidf.toarray()
dense_tf_x_test = X_test_tfidf.toarray()


# In[258]:


clf1 = GaussianNB()
clf2 = MultinomialNB()
clf3 = BernoulliNB()


# In[259]:


clf1.fit(dense_tf_x_train , Y_train)
clf2.fit(dense_tf_x_train , Y_train)
clf3.fit(dense_tf_x_train , Y_train)


# In[260]:


y_pred1 = clf1.predict(dense_tf_x_test)
y_pred2 = clf2.predict(dense_tf_x_test)
y_pred3 = clf3.predict(dense_tf_x_test)


# In[261]:


from sklearn.metrics import accuracy_score


# In[262]:


print( 'Gaussian',accuracy_score(Y_test,y_pred1))
print( 'MultinomialNB',accuracy_score(Y_test,y_pred2))
print( 'BernoulliNB',accuracy_score(Y_test,y_pred3))


# In[263]:


###CONCLUSION: Best Accuracy is 80% from RANDOM FOREST so we will go for that.

