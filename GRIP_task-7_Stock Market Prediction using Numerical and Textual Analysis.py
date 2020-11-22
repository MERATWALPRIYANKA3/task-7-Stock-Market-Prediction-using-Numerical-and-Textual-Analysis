#!/usr/bin/env python
# coding: utf-8

# # author: Priyanka Meratwal
# # task 7:Stock Market Prediction using Numerical and Textual Analysis

# In[34]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame
start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2020, 1, 11)

df = web.DataReader("AAPL", 'yahoo', start, end)
df.tail()


# Exploring Rolling Mean and Return Rate of Stocks¶
# 
# Rolling Mean
# Rolling mean/Moving Average (MA) smooths out price data by creating a constantly updated average price. This is useful to cut down “noise” in our price chart. Furthermore, this Moving Average could act as “Resistance” meaning from the downtrend and uptrend of stocks you could expect it will follow the trend and less likely to deviate outside its resistance point.
# 

# In[4]:


close_px = df['Adj Close']
mavg = close_px.rolling(window=100).mean()
mavg.head(20)


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from matplotlib import style

# Adjusting the size of matplotlib
import matplotlib as mpl
mpl.rc('figure', figsize=(10, 10))
mpl.__version__

# Adjusting the style of matplotlib
style.use('ggplot')

close_px.plot(label='AAPL')
mavg.plot(label='Moving Average')
plt.legend()
plt.show()


# Return Deviation — For the determination risk and return
# Expected Return measures the mean, or expected value, of the probability distribution of investment returns. The expected return of a portfolio is calculated by multiplying the weight of each asset by its expected return and adding the values for each investment — Investopedia.

# In[6]:


rets = close_px / close_px.shift(2) - 1
rets.plot(label='return')
plt.show()


# # Analysing Competitors Stocks¶

# In[7]:


dfcomp = web.DataReader(['AAPL', 'GE', 'GOOG', 'IBM', 'MSFT'],'yahoo',start=start,end=end)['Adj Close']
dfcomp.head()


# 
# # Analysis of Correlations (Dependency on one another)
# We can analyse the competition by running the percentage change and correlation function in pandas. Percentage change will find how much the price changes compared to the previous day which defines returns. Knowing the correlation will help us see whether the returns are affected by other stocks’ returns

# In[8]:


retscomp = dfcomp.pct_change()
corr = retscomp.corr()
retscomp.head(10)


# In[9]:


plt.imshow(corr, cmap='hot', interpolation='none')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns)
plt.yticks(range(len(corr)), corr.columns)


# # Returns rate of Stock and Risk
# Apart from correlation, we also analyse each stock’s risks and returns. In this case we are extracting the average of returns (Return Rate) and the standard deviation of returns (Risk)

# In[12]:


plt.scatter(retscomp.mean(), retscomp.std())
plt.xlabel('Expected returns')
plt.ylabel('Risk')
for label, x, y in zip(retscomp.columns, retscomp.mean(), retscomp.std()):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (20, -20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))


# # Loading News data in textual form.

# In[13]:


df1 = pd.read_csv('india-news-headlines.csv')
df1 = df1.iloc[0:10000]
df1.head(10)


# In[14]:


df1.tail(10)


# # Information of dataset

# In[15]:


df1.info()


# # Description of Dataset
# 

# In[16]:



df1.describe()


# In[17]:



df1.max()


# In[18]:



df1['headline_text'].unique()


# In[19]:


df1.isna().any()


# In[20]:


sns.set_palette('viridis')
sns.pairplot(df1)
plt.show()


# In[21]:


df1['headline_text'].value_counts()


# # EDA using NLP & NLTK tools
# 

# In[22]:



df1['headline_text'].str.len().hist()
plt.show()


# The histogram shows that news headlines range from 10 to 70 characters and generally, it is between 25 to 55 characters. Now, we will move on to data exploration at a word-level. Let’s plot the number of words appearing in each news headline.

# In[23]:


df1['headline_text'].str.split().   apply(lambda x : [len(i) for i in x]).    map(lambda x: np.mean(x)).hist()
plt.show()


# In[24]:


import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop=set(stopwords.words('english'))


# In[25]:


corpus=[]
new= df1['headline_text'].str.split()
new=new.values.tolist()
corpus=[word for i in new for word in i]

from collections import defaultdict
dic=defaultdict(int)
for word in corpus:
    if word in stop:
        dic[word]+=1


# In[26]:



from collections import Counter
counter=Counter(corpus)
most=counter.most_common()
x, y= [], []
for word,count in most[:40]:
    if (word not in stop):
        x.append(word)
        y.append(count)
        
sns.barplot(x=y,y=x)
plt.show()


# # Ngram exploration
# Ngrams are simply contiguous sequences of n words. For example “riverbank”,” The three musketeers” etc. If the number of words is two, it is called bigram. For 3 words it is called a trigram and so on. Looking at most frequent n-grams can give you a better understanding of the context in which the word was used.
# 

# In[27]:


from nltk.util import ngrams
list(ngrams(['I' ,'went','to','the','river','bank'],2))


# In[28]:


from sklearn.feature_extraction.text import CountVectorizer
def get_top_ngram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) 
                  for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:10]
top_n_bigrams=get_top_ngram(df1['headline_text'],2)[:10]
x,y=map(list,zip(*top_n_bigrams))
sns.barplot(x=y,y=x)
plt.show()


# # Using Textblob
# 
# 

# In[29]:


from textblob import TextBlob
TextBlob('100 people killed in Iraq').sentiment


# In[30]:


def polarity(text):
    return TextBlob(text).sentiment.polarity
df1['polarity_score']=df1['headline_text'].   apply(lambda x : polarity(x))
df1['polarity_score'].hist()
plt.show()


# You can see that the polarity mainly ranges between 0.00 and 0.20. This indicates that the majority of the news headlines are neutral. Let’s dig a bit deeper by classifying the news as negative, positive and neutral based on the scores.

# In[31]:


def sentiment(x):
    if x<0:
        return 'neg'
    elif x==0:
        return 'neu'
    else:
        return 'pos'
    
df1['polarity']=df1['polarity_score'].   map(lambda x: sentiment(x))
plt.bar(df1.polarity.value_counts().index,
        df1.polarity.value_counts())
plt.show()


# 70 % of news is neutral with only 18% of positive and 11% of negative. Let’s take a look at some of the positive and negative headlines.

# In[32]:


df1[df1['polarity']=='pos']['headline_text'].head()


# In[33]:


df1[df1['polarity']=='neg']['headline_text'].head()


# In[ ]:




