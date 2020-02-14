# hello and  welcome to my advanced python  assignement. 

# Web application

## Run the App
The app.py file contains a simplified application that can be run with the command flask run

## Acess
You can acess the website through :
http://127.0.0.1:5000/

to get the prediction with  data, go to the link: you cange change or igmore params and the application will impute them automatically
 http://127.0.0.1:5000/predict?date=2010-10-01T18:00:00&weathersit=1&temperature_C=15&feeling_temperature_C=14&humidity=20&windspeed=5

If you want to get the train score of the model you choose(xgboost or ridge), leave it blank if you want to have the default xcgboost:
http://127.0.0.1:5000/score?model=ridge
