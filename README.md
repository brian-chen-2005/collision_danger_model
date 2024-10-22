# traffic_proj

Model to predict danger of crash; How dangerous a crash would be given: 
*  1.location
*  2.road condition
*  3.lighting condition
*  4.weather

Pandas, numpy used to clean and process City of Seattle traffic collision data (CC0: Public Domain). 

Sklearn used to train RandomForestRegressor with optimization from grid search hyperparameter tuning. 

Further optimization attempts using polynomial features and onehot encoding. 

Model saved as joblib.
