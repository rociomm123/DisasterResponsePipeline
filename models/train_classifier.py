import sys
import nltk
nltk.download(['punkt', 'wordnet'])

import numpy as np
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
import pickle

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def load_data(database_filepath):
    '''
    Input:
        database_filepath: path to SQL database
    Output:
        X: Messages
        y: Categories 
        category_names: 36 categories
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df.message
    y = df[df.columns[4:]]
    #category_names = y.columns
    col_names = list(y.columns.values)
    return X, y, col_names 


def tokenize(text):
    '''
    Input:
        text: message
    Output:
        clean: tokenized, cleaned text
    '''
    # Normalize text
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    # Tokenize text
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    # Reduce words to their root form
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Output:
        GridSearch output
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # parameters set 
    parameters = {
        'clf__estimator__n_estimators': [50,100],
        "clf__estimator__max_depth":[8],
        'clf__estimator__min_samples_split': [2,3,4]
    }
    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=4, verbose=2, cv=3)
    return model


def evaluate_model(model, X_test, y_test, col_names):
    """
    Input:
        model: pipeline built in build_model
        X_test: test features
        Y_test: test labels
        col_names: 36 category labels
    Output:
        classification report for 36 categories
    """
    y_pred = model.predict(X_test)
    
    for i in range(len(col_names)):
        print((y_test.columns[i]).upper(),':')
        print(classification_report(y_test.iloc[:,i],y_pred[:,i],target_names=col_names))
   # class_report = classification_report(y_test, y_pred, target_names=category_names)
   # print(class_report)


def save_model(model, model_filepath):
    """
    Input:
        model: built model from build_model
        model_filepath: destination path to save .pkl file
    Output:
        saved model .pkl file
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


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
