#!/usr/bin/env python
# coding: utf-8

# # Product Matching
# ### Kelompok 1
# 1. 12S16002 - Diana Pebrianty Pakpahan
# 2. 12S16022 - Rosa Afresia Siagian
# 3. 12S16026 - Yolanda Nainggolan
# 4. 12S16028 - Kaleb Lonari Simanungkalit
# 
# 
# ### Dataset
# Data yang digunakan pada proyek ini berasal dari kaggle (https://www.kaggle.com/PromptCloudHQ/innerwear-data-from-victorias-secret-and-others) yang terdiri dari 9 file csv mengenai produk pakaian dalam.

# ## 1. Load Data

# In[1]:


import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
import gensim
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
import re

df_1 = pd.read_csv('C:/Users/Yolanda Nainggolan/Productmatching/ae_com.csv')
df_2 = pd.read_csv('C:/Users/Yolanda Nainggolan/Productmatching/amazon_com.csv')
df_3 = pd.read_csv('C:/Users/Yolanda Nainggolan/Productmatching/btemptd_com.csv')
df_4 = pd.read_csv('C:/Users/Yolanda Nainggolan/Productmatching/calvinklein_com.csv')
df_5 = pd.read_csv('C:/Users/Yolanda Nainggolan/Productmatching/hankypanky_com.csv')
df_6 = pd.read_csv('C:/Users/Yolanda Nainggolan/Productmatching/macys_com.csv')
df_7 = pd.read_csv('C:/Users/Yolanda Nainggolan/Productmatching/shop_nordstrom_com.csv')
df_8 = pd.read_csv('C:/Users/Yolanda Nainggolan/Productmatching/us_topshop_com.csv')
df_9 = pd.read_csv('C:/Users/Yolanda Nainggolan/Productmatching/victoriassecret_com.csv')


# In[2]:


dataFrame = pd.concat([df_1, df_2, df_3, df_4, df_5, df_6, df_7, df_8, df_9])
dataFrame.sample(5)


# ## 2. Data Preprocessing

# In[3]:


dataFrame = dataFrame[['product_name', 'description']]
dataFrame = dataFrame[pd.notnull(dataFrame['description'])]
dataFrame.sample(5)


# In[4]:


dataFrame.shape


# In[5]:


dataFrame.isnull().sum()


# #### Catatan:
# Karena jumlah data sangat besar, kita hanya memakai 10.000 data yang akan dipilih secara random

# In[6]:


df = dataFrame.sample(n=10000, random_state=42)
df.sample(10)


# In[7]:


df.shape


# In[8]:


df.index = range(10000)
df['description'].apply(lambda x: len(x.split(' '))).sum()


# In[9]:


from bs4 import BeautifulSoup
def cleanText(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text)
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    return text
df['description'] = df['description'].apply(cleanText)


# In[10]:


def print_description(index):
    example = df[df.index == index][['description', 'product_name']].values[0]
    if len(example) > 0:
        print('Description:\n', example[0])
        print('\n product_name:', example[1])
        
print_description(12)


# In[30]:


print_description(50)


# ## 3. Features Extraction

# In[12]:


import nltk
def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens
tokens = df.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['description']), tags=[r.product_name]), axis=1)


# In[13]:


tokens.values[30]


# In[14]:


import multiprocessing
cores = multiprocessing.cpu_count()


# In[15]:


model = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample = 0, workers=cores)
model.build_vocab([x for x in tqdm(tokens.values)])


# In[16]:


len(model.docvecs)


# In[17]:


model.wv.vocab.keys()


# In[19]:


model.most_similar('top')


# In[20]:


get_ipython().run_cell_magic('time', '', 'for epoch in range(30):\n    model.train(utils.shuffle([x for x in tqdm(tokens.values)]), total_examples=len(tokens.values), epochs=1)\n    model.alpha -= 0.002\n    model.min_alpha = model.alpha')


# In[ ]:


model.save("model.doc2vec")


# In[ ]:


model = gensim.models.Doc2Vec.load('model.doc2vec')  


# In[21]:


from sklearn.decomposition import PCA
from matplotlib import pyplot

get_ipython().run_line_magic('matplotlib', 'inline')

X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)

pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))

pyplot.show()


# In[ ]:





# ## 4. Kalkulasi Kemiripan

# In[22]:


def matching(inputs):
    new_sentence = inputs.split(" ")
    print("Berikut adalah 10 produk yang mirip")
    tag = model.docvecs.most_similar(positive=[model.infer_vector(new_sentence)],topn=10)
    
    return tag


# In[34]:


matching("pure seamless thong")


# In[35]:


matching("Body by Victoria NEW! Unlined Demi Bra")


# In[36]:


matching("aerie cutout longline bralette")


# In[ ]:




