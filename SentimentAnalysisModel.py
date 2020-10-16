'''Creating a machine learning model for sentiment analysis.
Training data will be the New_sentiment_train.csv file created with the original stanford sentiment Treebank.
Then, apply the ML model on web scraped data containing amazon reviews of 10 mobile phones.
'''

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('New_sentiment_train.csv',  usecols = ['Phrase','Probability'])

# we can classify the review as :-
#[0, 0.2], (0.2, 0.4], (0.4, 0.6], (0.6, 0.8], (0.8, 1.0]
# very negative, negative, neutral, positive, very positive, respectively.
# let each class is represented as:-
#   very negative --->  1
#   negative      --->  2
#   neutral       --->  3
#   positive      --->  4
#   very positive --->  5

for i in range(0,len(data['Probability'])):
    if(data['Probability'][i]<=0.2):
        data['Probability'][i]=1
    elif (data['Probability'][i] > 0.2 and data['Probability'][i] <= 0.4  ):
        data['Probability'][i] = 2
    elif (data['Probability'][i] > 0.4 and data['Probability'][i] <= 0.6  ):
        data['Probability'][i] = 3
    elif (data['Probability'][i] > 0.6 and data['Probability'][i] <= 0.8  ):
        data['Probability'][i] = 4
    elif (data['Probability'][i] > 0.8 and data['Probability'][i] <= 1.0  ):
        data['Probability'][i] = 5
    else:
        None

X = data['Phrase']

#converting x to TFidf vector
def identity_tokenizer(text):
    return text
tfidf = TfidfVectorizer(tokenizer=identity_tokenizer, max_features=100000, lowercase=False)

X= tfidf.fit_transform(X)
y= data['Probability']

#to evaluate model, we will need test data
#split X,y in train test data
X_train,X_test,y_train,y_test =  train_test_split(X,y,test_size=0.2,random_state=42)


#creating a SVM model for sentiment analysis
clf = LogisticRegression(random_state=0)
clf.fit(X_train, y_train)

#testing and evaluating model
y_pred = clf.predict(X_test)
print(classification_report(y_test,y_pred))

#saving the model
import pickle
pickle.dump(clf, open('MODEL_LR', 'wb'))


#Applying the model on the scraped amazon reviews
#Sentiment score of reviews for different mobile will be saved in different csv file
#It will be saved in SentimentResult dir

Sphones_dir = ['MOBILE_OnePlus 7T Pro (Haze Blue, 8GB', 'MOBILE_OnePlus Nord 5G (Gray Onyx, 8G',
               'MOBILE_OPPO F17 Pro (Magic Blue, 8GB ', 'MOBILE_Redmi 9 Prime (Space Blue, 4GB',
               'MOBILE_Redmi Note 9 (Pebble Grey, 4GB', 'MOBILE_Samsung Galaxy M01 (Blue, 3GB ',
               'MOBILE_Samsung Galaxy M21 (Blue, 4GB ', 'MOBILE_Samsung Galaxy M31 (Ocean Blue',
               'MOBILE_Samsung Galaxy M31s (Mirage Bl', 'MOBILE_Vivo Y91i (Ocean Blue, 2GB RAM']

Sphones_names = ['OnePlus 7T Pro', 'OnePlus Nord', 'OPPO F17', 'Redmi 9 Prime', 'Redmi Note 9',
                 'Samsung Galaxy M01', 'Samsung Galaxy M21', 'Samsung Galaxy M31', 'Samsung Galaxy M31s',
                 'Vivo Y91i']
#var k to track the Sphone_names
#These names will be used for file names for result
k=0

for Sph_dir in Sphones_dir:
    df_Sphone = pd.read_csv(Sph_dir+'.csv')

    #Using the model to predict sentiment for Scraped Amazon reviews
    #converting the review to tfidf vector
    review_vector = tfidf.transform(df_Sphone['Customer Review'])
    sentiment_score = clf.predict(review_vector)
    sentiment = []
    #Converting sentiment score (1-5) in sentiments like positive or negative
    #For better representation
    for i in range(0,len(sentiment_score)):
       if(sentiment_score[i] == 1):
           sentiment.append('Very Negative')
       elif(sentiment_score[i] == 2):
           sentiment.append('Negative')
       elif (sentiment_score[i] == 3):
           sentiment.append('Neutral')
       elif (sentiment_score[i] == 4):
           sentiment.append('Positive')
       else:
           sentiment.append('Very Positive')


    result = pd.DataFrame()
    result['Customer Name'] = df_Sphone['Customer Name']
    result['Customer Review'] = df_Sphone['Customer Review']
    result['Sentiment Score'] = sentiment_score
    result['Sentiment'] = sentiment
    result.to_csv(r'D:\Strings\Scraping&NLP\SentimentResult\Result_' + Sphones_names[k] + '.csv')
    k=k+1

