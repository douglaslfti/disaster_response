# import libraries
import sys
import pickle
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data(database_filepath, database_tablename='DisasterResponse'):
    '''
    load_data:
    The load_data function loads data from the database and puts it into a DataFrame. It also separates the DataFrame in two. X is an array of characteristic values, being known values, and y is a vector of target values, being the values you want to try to predict.
    
    Input:
    database_filepath       This variable represents the database path and file
    database_tablename      This variable represents the table after the data treatment that is contained in the database file mentioned above.
    
    Returns:
    X              The variable X contains the data from the columns,['related','request','offer',
                    'aid_related','medical_help','medical_products','search_and_rescue','security',
                    'military','child_alone','water','food','shelter','clothing','money','missing_people',
                    'refugees','death','other_aid','infrastructure_related','transport','buildings',
                    'electricity','tools','hospitals','shops','aid_centers','other_infrastructure',
                    'weather_related','floods','storm','fire','earthquake','cold','other_weather','direct_report'] of       the DataFrame.
    
    y               The variable y contains the data from the message column of the DataFrame
    
    target_names    Name of the columns of variable X
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
    tokenize
    The tokenize function treats the data, transforming it into lower, removing special characters that can get in the way when analyzing the text. To do this, we use normalization, tokenization and lemmatization techniques
    
    Input: 
    text        The text variable in question are the messages that are contained in variable X 
    
    Returns:
    clean_tokens    Returns a list of the processed text
    '''
    
    # Regex rule for eliminating improper characters
    regex = '[^a-zA-Z0-9 ]'
    
    # This block serves to find the elements contained in the regex rule and replace them with empty.
    detected_elements = re.findall(regex, text)
    for element in detected_elements:
        text = text.replace(element, "")
   
    # Applying the Tokenization technique to text
    tokens = word_tokenize(text)
    
    # Defining stop words
    stop_words = set(stopwords.words('english'))

    # Remove stop words
    tokens = [tok for tok in tokens if tok not in stop_words]

    # Instantiated WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    # The following block of code is used to apply the lemmatizer technique to text and appending in a list
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    build_model:
    The build_model function is for creating the model based on pipeline features and using GridSearch to find the best parameters for the model. To do this, you must pass a dictionary with some values for GridSearch to search.
    
    Input:
    The function has no input variable
    
    Returns:
    cv      Returns the model with the best parameter
    '''
    
    # Creating the Pipeline
    pipeline = Pipeline([
    
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),
    
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # Creating parameters for GridSearch
    parameters = {
        'clf__estimator__n_estimators': [10],
        'clf__estimator__min_samples_split': [2, 3, 4]
    }

    # Searching for the best parameter
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluate_model:
    This function evaluates how good the model is through the chosen metrics.
    
    Input:
    model           The model
    X_test          The variable X_test contains the predictions of our model
    Y_test          The variable Y_test contains the test values
    category_names  List with the names of evaluated columns

    Returns:
    NONE
    '''

    # Predict function to calculate our predicted y
    Y_pred = model.predict(X_test)
    
    # Print the metrics
    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    '''
    save_model:
    This function is for saving the trained model.
    
    Input:
    model           The model
    model_filepath  Path where the model is going to be saved
    
    Returns:
    NONE
    '''
    
    # This block of code export the model as a pickle file
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath, database_tablename='DisasterResponse')
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