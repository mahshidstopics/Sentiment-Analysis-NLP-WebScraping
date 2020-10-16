This is the project to scrape the customer reviews for 10 mobile phones listed on Amazon and perform Sentiment Analysis 
for the customer reviews.

Directories->

1) Scraped Reviews :- This directory contains all amazon reviews for 10 different mobiles that are web scraped 
	              form amazon.in

2) Sentiment Analysis Model :- This directory contains the trained Logistic regression model for sentiment analysis.

3) Sentiment Result :- This directory contains sentiment analysis result/predictions of amazon reviews.

4) SentimentResults/ResultVisualiztion :- This directory have all the visualizations of sentiment analysis for 10 
					  different mobile phones. 

5) Training Data:- Contains the training data set from stanform sentiment treebank.

Scripts ->

1) scrarping.py :- This contains the script used to scrape amazon reviews using beautiful soup.

2) Cleaning_Preprocessing_SentimentAnalysis.py :- Contains script for preprocessing and data cleaning of training dataset.

3) SentimentAnalysisModel.py :- Contains script for training model using cleaned dataset and then creating sentiment
			        analysis results for Reviews scraped from amazon.in

4) Visualization.py :- Used to create visualization for the final sentiment analysis results. 
