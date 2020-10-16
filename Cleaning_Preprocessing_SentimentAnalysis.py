'''cleaning of dataset and saving it in new csv file New_sentiment_train.csv
New_sentiment_train.csv will contain two columns
1) Phrases - this includes lower case tokenized words, no punctuations and no stop words
2) Probability - this include the positivity probability for sentiment analysis
'''

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string


#reading the phrases.txt file and sorting the pharases according to ids
#this will store all the phrases in sentimentsPharases, sorted by ids
sentimentsPhrases = pd.read_table("trainingData\phrases.txt",sep='|')
sentimentsPhrases = sentimentsPhrases.sort_values(by = 'sentiment id')
sentimentsPhrases = sentimentsPhrases['phrase']

#reading the prob.txt file to get the positivity probability
# we can classify the review as :-
#[0, 0.2], (0.2, 0.4], (0.4, 0.6], (0.6, 0.8], (0.8, 1.0]
# very negative, negative, neutral, positive, very positive, respectively.
#File is already sorted according to phrase ids
sentimentsProb = pd.read_table("trainingData\prob.txt",sep='|')
sentimentsProb = sentimentsProb['sentiment values']


#creating a csv file called df_sentiment_train with phrase and the positivity probability as columns
#column 1. contains phrases
#column 2. contains corresponding positivity probability

df_sentiment_train = pd.DataFrame()
df_sentiment_train['Phrase'] = sentimentsPhrases
df_sentiment_train['Probability'] = sentimentsProb
df_sentiment_train.to_csv(r'D:\Strings\Scraping&NLP\sentiment_train.csv')

df_sentiment_train = pd.read_csv("sentiment_train.csv", usecols = ['Phrase','Probability'])

#df_sentiment_train is the training dataset
#cleaning dataset for NLP
# 1) converting text in lowercase
# 2) removing punctuations
# 3) tokenization
# 4) removing stop words

#converting string.punctuation to set and stopwords to set(stopwords)
#string.punctuation is iterated for every single character in initial text and stopwords is iterated for every token

#converting to lower case
df_sentiment_train['Phrase'] = [sentence.lower() for sentence in df_sentiment_train['Phrase']]

#removing punctuations
df_sentiment_train['Phrase'] = [sentence.translate(str.maketrans('','',string.punctuation)) for sentence in df_sentiment_train['Phrase']]

#tokenization
df_sentiment_train['Phrase'] = [word_tokenize(sentence,"english") for sentence in df_sentiment_train['Phrase']]

#removing stop words
for i in range(0,len(df_sentiment_train['Phrase'])):
    for word in df_sentiment_train['Phrase'][i]:
        if word in stopwords.words('english'):
            df_sentiment_train['Phrase'][i].remove(word)


#print(df_sentiment_train['Phrase'].head())

#saving all the clean data in new csv file
#We won't have to run this file again and again
New_df_sentiment_train = pd.DataFrame()
New_df_sentiment_train['Phrase'] = df_sentiment_train['Phrase']
New_df_sentiment_train['Probability'] = sentimentsProb
New_df_sentiment_train.to_csv(r'D:\Strings\Scraping&NLP\New_sentiment_train.csv')