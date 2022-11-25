import json
import plotly
import heapq  
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Graph representing the percentage of values recorded with 1 per column
    perc_values = (((df.iloc[:,4:] == 1).sum().values / df.shape[0]) * 100).round(2)
    perc_columns = df.iloc[:,4:].columns
    
    # Graph representing the top 5 percent of recorded values with 1 per column
    perc_5_values = (((df.iloc[:,4:] == 1).sum().values / df.shape[0]) * 100).round(2)
    perc_5_columns = df.iloc[:,4:].columns
    zipped = zip(perc_5_values, perc_5_columns)
    # Takes the top 5 values
    highest_values = heapq.nlargest(5,zipped)
    column_5_name = []
    perc_5_value = []
    [column_5_name.append(highest_values[index][1]) for index in range(len(highest_values))]
    [perc_5_value.append(highest_values[index][0]) for index in range(len(highest_values))]
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        # GRAPH 1 - genre graph
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        # GRAPH 2 - percentage graph    
        {
            'data': [
                Bar(
                    x=perc_columns,
                    y=perc_values
                )
            ],

            'layout': {
                'title': 'Distribution of Percentage equal 1 per Categories',
                'yaxis': {
                    'title': "Percentage"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 35
                }
            }
        },
        
        # GRAPH 2 - Top 5 percentage graph    
        {
            'data': [
                Bar(
                    x=column_5_name,
                    y=perc_5_value
                )
            ],

            'layout': {
                'title': 'Distribution of top 5 Percentage equal 1 per Categories',
                'yaxis': {
                    'title': "Percentage"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 35
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()