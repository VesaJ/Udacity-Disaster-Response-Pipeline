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
#from sklearn.model_selection import RandomizedSearchCV

from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import pickle


def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql_table("DisasterResponse", con=engine)

    # some cleaning issues
    df = df[df.related != 2] # related column had three category values, exluded value 2(with   n=188)
    df = df.drop(['child_alone'],axis=1) # child alone column had only zeros,dropped out 

    X = df['message'] # feature to predict
    y = df.drop(['message', 'genre', 'id', 'original'], axis = 1) # select 35 categories, where message is predicted
    category_names = y.columns
    return X,y,category_names 

# regular expression to detect a url is given below
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

'''
    Tokenize function
  
'''
    
def tokenize(text,url_place_holder_string="urlplaceholder"):
    
    
   # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    
    # rreplace each url in text string with placeholder
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


def evaluate_model(model, X_test, Y_test, category_names):
    #testing predictions
    y_pred = cv.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))

          
         
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
