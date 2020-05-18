# Udacity-Disaster-Response-Pipeline
Udacity Data Science Nanaodegree - Data Engineer Project
## Description


This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The aim of the project is to build a Natural Language Processing tool that categorize messages.

1. Data Processing, ETL Pipeline to extract data from the source, clean data and save data in a proper database structure
2. Machine Learning Pipeline to train a model able to classify text message in categories
3. Web App to show model results in real-time.


### Instructions:
Run the following commands in the project's root directory to set up your database and model.

To run ETL pipeline that cleans data and stores in database python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db

To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

Run the following command in the app's directory to run your web app. python run.py

Go to http://0.0.0.0:3001/


### Author

Vesa Jaakola

### About
This project was prepared as part of the Udacity Data Scientist nanodegree programme. The data was provided by Figure Eight.

### License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### Acknowledgements
Must give credit to Udacity for the amazing learning environment and Figure Eight for the data. 


