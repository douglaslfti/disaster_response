# import libraries
import sys
import pickle
import nltk
nltk.download(['punkt', 'wordnet'])

import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data(database_filepath, database_tablename='DisasterResponse'):
    '''
    OBJECTIVE:
    The load_data function loads data from the database and puts it into a DataFrame. It also separates the DataFrame in two. X is an array of characteristic values, being known values, and y is a vector of target values, being the values you want to try to predict.
    
    INPUTS:
    database_filepath: This variable represents the database path and file
    database_tablename: This variable represents the table after the data treatment that is contained in the database file mentioned above.
    
    OUTPUTS:
    X: The variable X contains the data from the columns,['related','request','offer',
    'aid_related','medical_help','medical_products','search_and_rescue','security',
    'military','child_alone','water','food','shelter','clothing','money','missing_people',
    'refugees','death','other_aid','infrastructure_related','transport','buildings',
    'electricity','tools','hospitals','shops','aid_centers','other_infrastructure',
    'weather_related','floods','storm','fire','earthquake','cold','other_weather','direct_report'] of       the DataFrame.
    
    y: The variable y contains the data from the message column of the DataFrame
    
    target_names: Name of the columns of variable X
    '''
    # Creating a database connection and creating a DataFrame
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(database_tablename, con=engine)
    
    X = df["message"]
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    target_names = Y.columns.values

    return X, Y, target_names


def tokenize(text):
    '''
    OBJECTIVE:
    The tokenize function treats the data, transforming it into lower, removing special characters that can get in the way when analyzing the text. To do this, we use normalization, tokenization and lemmatization techniques
    
    INPUT: 
    text: The text variable in question are the messages that are contained in variable X 
    
    OUTPUT:
    clean_tokens: Returns a list of the processed text
    '''
    
    # Regex rule for eliminating improper characters
    regex = '[^a-zA-Z0-9 ]'
    
    # This block serves to find the elements contained in the regex rule and replace them with empty.
    detected_elements = re.findall(regex, text)
    for element in detected_elements:
        text = text.replace(element, "")
   
    # Applying the Tokenization technique to text
    tokens = word_tokenize(text)
    
    # Instantiated WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    # The following block of code is used to apply the lemmatizer technique to text and appending in a list
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


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