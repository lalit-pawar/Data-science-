#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize


# In[2]:


#Initialize the text

text = "Tokenization is the first step in text analytics.The process of breaking down a text paragraph into smaller chunks such as words or sentences is called Tokenization."


# In[3]:


#Sentence Tokenization
from nltk.tokenize import sent_tokenize
tokenized_text= sent_tokenize(text)
print(tokenized_text)


# In[4]:


# Word Tokenization
from nltk.tokenize import word_tokenize

text = "Tokenization is the first step in text analytics.The process of breaking down a text paragraph into smaller chunks such as words or sentences is called Tokenization."
tokens = word_tokenize(text)
print(tokens)


# In[5]:


#4: Removing Punctuations and Stop Word

from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))
print(stop_words)


# In[6]:


from nltk.corpus import stopwords
import re

text= "How to remove stop words with NLTK library in Python?"
text= re.sub('[^a-zA-Z]', ' ',text)
tokens = word_tokenize(text.lower())
filtered_text=[]
for w in tokens:
 if w not in stop_words:
  filtered_text.append(w)
print("Tokenized Sentence:",tokens)
print("Filterd Sentence:",filtered_text)


# In[7]:


#perform Stemming

from nltk.stem import PorterStemmer
e_words= ["wait", "waiting", "waited", "waits"]
ps =PorterStemmer()
for w in e_words:
  rootWord=ps.stem(w)
print(rootWord)


# In[8]:


#perofrm Lemmatization
import nltk
nltk.download('wordnet')

nltk.download('omw-1.4')


# In[9]:



from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
text = "studies studying cries cry"
tokenization = nltk.word_tokenize(text)
for w in tokenization:
   print("Lemma for {} is {}".format(w,
wordnet_lemmatizer.lemmatize(w)))


# In[10]:


#Apply POS Tagging to text
import nltk
nltk.download('averaged_perceptron_tagger')

from nltk.tokenize import word_tokenize
data="The pink sweater fit her perfectly"
words=word_tokenize(data)
for word in words:
  print(nltk.pos_tag([word]))


# In[ ]:




