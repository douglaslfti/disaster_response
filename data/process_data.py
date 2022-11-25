import sys
import sqlite3
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    load_data
    This function reads two CSVs and transforms them into two distinct DataFrames. After that, they merged into a single DataFrame.
    
    Input:
    messages_filepath       Message archive path
    categories_filepath     Categories Message archive path
    
    Returns:
    df      Returns a DataFrame with the combination of information from the message and category datasets.
    '''
    
    # Import message data
    messages = pd.read_csv(messages_filepath)
    
    # Import category data
    categories = pd.read_csv(categories_filepath)
    
    # Merge the two datasets together using the id
    df = messages.merge(categories, how='outer', on=['id'])
        
    return df

def clean_data(df):
  '''
  clean_data
  This function is used to process the data. It removes duplicate data, renames the columns for better understanding, and corrects the data to be binarized.
  
  Input:
  df    DataFrame containing the data.
  
  Returns:
  df    Returns the DataFrame with the processed data.
  '''
  # Create a dataframe of the 36 individual category columns
  categories = df['categories'].str.split(';', expand=True)
  categories.head()

  # Select the first row of the categories dataframe
  row = categories.iloc[0]

  # Use this row to extract a list of new column names for categories.
  # Removes the two last characters
  category_colnames = list([item[0:-2] for item in row])

  # Rename the columns of `categories`
  categories.columns = category_colnames

  for column in categories:
    # Set each value to be the last character of the string
    categories[column] = categories[column].str.replace(column+'-', '')
    
    # Convert column from string to numeric
    categories[column] = categories[column].astype(int)

  # In the related column replace the value 2 with 1
  categories['related'] = categories['related'].replace(2, 1)

  # Drop the original categories column from `df`
  df.drop('categories',inplace=True, axis=1)

  # Concatenate the original dataframe with the new `categories` dataframe
  df = pd.concat([df, categories], sort=False, axis=1)

  # Check number of duplicates
  if len(df[df.duplicated()]) != 0:
    # Drop duplicates
    df.drop_duplicates(keep=False, inplace=True)

  return df


def save_data(df, database_filename):
    '''
    save_data
    The function is for saving the DataFrame to a database

    Input:
    df    The variable contains a DataFrame with the data handled by the clean_data function
    database_filename - Database file name

    Returns:
    NONE
    '''

    # Connect to the sqlite database
    engine = create_engine('sqlite:///' + database_filename)

    # Save the dataframe in the sqlite database
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace') 


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