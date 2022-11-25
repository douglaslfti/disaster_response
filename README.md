# Disaster Response Pipeline Project

### Table of Contents
1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Run](#run)
5. [Licensing, Author and Acknowledgements](#laa)

## Installation <a name="installation"></a>

For the project to run perfectly, the following packages must be installed in the environment where the code will run: sys, pickle, nltk [word_tokenize, WordNetLemmatizer], re, numpy, pandas, sqlalchemy [create_engine] and sklearn [GridSearchCV, MultiOutputClassifier, RandomForestClassifier, train_test_split, classification_report, Pipeline, CountVectorizer, TfidfTransformer].

## Project Motivation<a name="motivation"></a>

This project plays a significant role in the community, as it will be possible to help people or companies to cope better in critical situations, such as a disaster caused by some natural phenomenon. 

It also allowed me to use data from Figure Eight to understand better:

1. How to work with natural language.
2. Understand how Pipelines to facilitate project development.
3. And finally put together everything that was learned in the course module to put the knowledge into practice.

## File Descriptions <a name="files"></a>

In this project it is organized as follows:

app
| - template
| |- master.html # main page of web app
| |- go.html # classification result page of web app
|- run.py # Flask file that runs app
data
|- disaster_categories.csv # data to process
|- disaster_messages.csv # data to process
|- process_data.py
|- DisasterResponse.db # database to save clean data to
models
|- train_classifier.py
|- classifier.pkl # saved model
README.md

The'app' folder  contains the file 'run.py', which you need to "run" to get the project "online", and a folder named 'template'. In addition, this folder contains two HTML files ('go.html' and 'master.html'). These HTML files developed with HTML and flask serve to show the project's result and to make input for new predictions.

The 'data' folder has four files in total. First, two CSV files ("disaster_categories.csv" and "disaster_messages.csv") contain the data to be worked on in this project. Next, the "DisasterResponse.db" file includes the data handled from the above CSV files. Finally, the file "process_data.py" contains the code where the transformations of the information available in the CSV files occur.

Finally, the 'models' folder consists of the files "classifier.pkl" and "train_classifier.py". The first file represents the trained and saved model. The second file contains the code with machine learning and natural language techniques.

## Run <a name="run"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Licensing, Author and Acknowledgements <a name="laa"></a>

Thanks to Udacity for making it possible to apply the knowledge learned with excellent partners and to Figure Eight for making the data available.