'''
The results of sentiment analysis were saved in SentimentResult directory.
Using matplotlib creating histogram plot of reviews sentiment
Using seaborn to create scatter plot of review sentiment
Saving the visualization in ResultVisualization directory
'''


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

mobiles_name = ['OnePlus 7T Pro', 'OnePlus Nord', 'OPPO F17', 'Redmi 9 Prime', 'Redmi Note 9',
                 'Samsung Galaxy M01', 'Samsung Galaxy M21', 'Samsung Galaxy M31', 'Samsung Galaxy M31s',
                 'Vivo Y91i']

#to track mobile_name declared variable k
k=0

#mobiles_result contain files with final sentiment score and results
mobiles_result = ['OnePlus 7T Pro','OnePlus Nord','OPPO F17','Redmi 9 Prime',
                  'Redmi Note 9','Samsung Galaxy M01','Samsung Galaxy M21','Samsung Galaxy M31',
                  'Samsung Galaxy M31s','Vivo Y91i']

#creating visualizations for every mobile
for mobRes in mobiles_result:
    #reading the result file
    path = 'SentimentResult\Result_'
    df = pd.read_csv(path+mobRes+'.csv')

    #creating seprate directory for each mobile phone
    newfolder = os.mkdir('D:\Strings\Scraping&NLP\SentimentResult\ResultVisualization\Vis_'+mobiles_name[k])

    #Creating and saving histogram plot for reviews sentiments
    plt.figure(figsize=(10,6), dpi=100)
    X=df['Sentiment']
    plt.hist(X)
    plt.title('Histogram Plot for '+mobiles_name[k])
    plt.xlabel('Sentiments')
    plt.ylabel('No. of reviews')
    plt.savefig('SentimentResult\ResultVisualization\Vis_' +mobiles_name[k]+ '\Hist.png')
    plt.clf()

    # Creating and saving Scatter plot for reviews sentiments
    plt.figure(figsize=(15, 7), dpi=200)
    svm=sns.scatterplot(x=np.arange(1,len(df)+1) , y=df['Sentiment'])
    plt.title('Scatter Plot for '+mobiles_name[k])
    plt.xlabel('No. of reviews')
    figure = svm.get_figure()
    figure.savefig('SentimentResult\ResultVisualization\Vis_' +mobiles_name[k]+ '\Scatter.png')

    k=k+1