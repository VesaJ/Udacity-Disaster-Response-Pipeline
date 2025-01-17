# import libraries
import sys

import pandas as pd
from sqlalchemy import create_engine
import sqlite3

# load data
def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv('data/messages.csv')
    categories = pd.read_csv('data/categories.csv')
    df = messages.merge(categories, on='id')
    return df

    # clean data
def clean_data(df):
    categories = df["categories"].str.split(";", expand=True)
    row = categories.iloc[0]
    category_colnames = row.transform(lambda x: x[:-2]).tolist()
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].transform(lambda x: x[-1:])
    
    # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)
    # drop duplicates
    df.drop_duplicates(inplace=True)
    

    return df

    # save dataframe to database in 'DisasterResponse' table
def save_data(df, database_filename):
    conn = sqlite3.connect('DisasterResponse.db')
    df.to_sql('messages', con=conn, if_exists='replace', index=False)
    engine = 'data/DisasterResponse.db.backends.sqlite3'
    name = 'data/DisasterResponse/data.sqlite3'
    
    

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
    
