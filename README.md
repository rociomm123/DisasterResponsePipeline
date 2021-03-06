# DisasterResponsePipeline
Disaster Response Pipeline


### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Project structure](#structure) 
5. [Results](#results)
6. [Instructions](#licensing)

## Installation <a name="installation"></a>

There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python. The code should run with no issues using Python versions 3.*. Outside of anaconda there were 2 packages installed
test
## Project Motivation<a name="motivation"></a>

In this project,  1000s of real messages will be analyzed providing by figuring that were sent during natural disasters, either via social media, or directly to disaster response organizations. This project caterogize the disaster messages. This application helps people or organization during an event of a disaster.


## File Descriptions <a name="files"></a>

There are 2 notebooks available here to showcase work related to the project goals. 
 - disaster_categories.csv: raw category.
 - disaster_messages.csv: raw messages. 

Each of the notebooks is exploratory in searching through the data pertaining to the questions showcased by the notebook title. 
The ETL pipeline that processes message and category data from CSV files, and load them into a sequel lite database, which your machine learning pipeline will then read from to create and save a multi output supervised learning model.

There is an additional `run.py` flask file that runs the app. train_classifier build, create and evelauate the necessary code to obtain the final model classifier.pkl used to categorize the disaster message.

## Project Structure <a name="structure"></a>

app

| - template

| |- master.html # main page of web app

| |- go.html # classification result page of web app

|- run.py # Flask file that runs app

data

|- disaster_categories.csv # data to process

|- disaster_messages.csv # data to process

|- process_data.py

|- InsertDatabaseName.db # database to save clean data to
models

|- train_classifier.py

|- classifier.pkl # saved model

README.md

## Results<a name="results"></a>

Machine learning is critical to helping different organizations understand which messages are relevant to them, and which messages to prioritize during these disasters is when they have the least capacity to filter out messages that matter and find basic messages such as using keyword searches to provide trivial results.

## Instructions<a name="licensing"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`


