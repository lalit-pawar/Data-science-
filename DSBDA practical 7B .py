#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


#Initialize the Documents.

documentA = 'Jupiter is the largest Planet'
documentB = 'Mars is the fourth planet from the Sun'


# In[3]:


#Create BagofWords (BoW) for Document A and B.

bagOfWordsA = documentA.split(' ')
bagOfWordsB = documentB.split(' ') 


# In[4]:


#Create Collection of Unique words from Document A and B.

uniqueWords = set(bagOfWordsA).union(set(bagOfWordsB))


# In[9]:


# Create a dictionary of words and their occurrence for each document in the
#corpus

numOfWordsA = dict.fromkeys(uniqueWords, 0)
for word in bagOfWordsA:
 numOfWordsA[word] += 1
 numOfWordsB = dict.fromkeys(uniqueWords, 0)
for word in bagOfWordsB:
  numOfWordsB[word] += 1


# In[11]:


# Print the frequency dictionaries
print( numOfWordsA)
print( numOfWordsB)


# In[17]:


#Compute the term frequency for each of our documents.

def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict

tfA = computeTF(numOfWordsA, bagOfWordsA)
tfB = computeTF(numOfWordsB, bagOfWordsB)


# In[18]:


import math

def computeIDF(documents):
    N = len(documents)
    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1
    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict

idfs = computeIDF([numOfWordsA, numOfWordsB])
print(idfs)


# In[36]:


#Compute the term TF/IDF for all words.


def computeTFIDF(numOfWords, idfs):
    tfidf = {}
    for word, val in numOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf

tfidfA = computeTFIDF(numOfWordsA, idfs)
tfidfB = computeTFIDF(numOfWordsB, idfs)
df = pd.DataFrame([tfidfA, tfidfB])
print(df)


# In[ ]:





# In[ ]:




