# Import libraries
import sys

import pandas as pd
import numpy as np
import re

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.corpus import stopwords
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import pickle

    # load data from database
def load_data(database_filepath):
    engine = create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql_table("DisasterResponse", con=engine)

    # some cleaning issue:
    # related column had three category values, exluded value 2(n=188)
    df = df[df.related != 2] 
    # feature to predict
    X = df['message'] 
    # select 36 categories, where message is predicted 
    y = df.drop(['message', 'genre', 'id', 'original'], axis = 1) 
    category_names = y.columns
    return X,y,category_names 

# regular expression to detect a url is given below
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


    
    # tokenize function
def tokenize(text,url_place_holder_string="urlplaceholder"):
    
    
   # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with placeholder
    for detected_url in detected_urls:
        text = text.replace(detected_url, url_place_holder_string)

    # tokenize text
    tokens = nltk.word_tokenize(text)
    
    # remove stopwords
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    
    # initiate lemmatizer
    lemmatizer = nltk.WordNetLemmatizer()

    # lemmatize, normalize case, and remove leading/trailing white space
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]
    
    return clean_tokens
    

     # build a pipeline funtion
def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
    'clf__estimator__n_estimators': [10,50],
    'clf__estimator__min_samples_split': [2,4],
}

    cv = GridSearchCV(estimator=pipeline, n_jobs = -1, param_grid=parameters)
   
    return cv

    # evaluate a model
def evaluate_model(model, X_test, Y_test, category_names):
    #testing predictions
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))

          
    # save the model     
def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))
    
        
   


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
