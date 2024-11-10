# Overview

Model to predict danger of crash; How dangerous a crash would be given: 
*  1.location
*  2.road condition
*  3.lighting condition
*  4.weather

Pandas, numpy used to clean and process City of Seattle traffic collision data (CC0: Public Domain). 

Sklearn used to train RandomForestRegressor with optimization from grid search hyperparameter tuning. 

Further optimization attempts using polynomial features and onehot encoding. 

Model saved as joblib.

# To run
**Both dataset and model are too large to upload:**

Download dataset from City of Seattle: https://data-seattlecitygis.opendata.arcgis.com/datasets/SeattleCityGIS::sdot-collisions-all-years/about

**Open model_building.py to change the directories at:**

line 35: df = pd.read_csv('/_your/directory/folder/here_/SDOT_Collisions_All_Years.csv')

line 196: joblib.dump(best_rf_regressor, '/_your/directory/folder/here_/random_forest_model.joblib')

**Run model_building.py in virtual environment (Mac):**

cd /Users/[user]/[folder trafficProj is contained]

source trafficProj/bin/activate

python /Users/brian/trafficProj/model_building.py 

**Run model_calling.py:**

python /Users/brian/trafficProj/model_calling.py

(change input values in model_calling as desired)

# Dataset Interpretation

Arbitrary danger value intepretation of severity_code
**The model tries to predict this danger value**
severity_code | 1 | 2 | 4 | 3
--- | --- | --- | --- | ---
semantic meaning | Property Damage | Injury | Serious Injury | Fatality
assigned danger value | 1 | 6 | 8 | 10

Arbitrary interpretation of parameters to integers (used without onhot encoding in randomforestregression):
Road Condition | Int Interpretation | Lighting Condition | Int Interpretation | Weather | Int Interpretation 
--- | --- | --- | --- | --- | ---
Dry | 0 | Daylight | 0 | Clear | 0
Sand/Mud/Dirt | 1 | Dawn | 1 | Partly Cloudy | 1
Standing Water | 2 | Dusk | 2 | Overcast | 2
Snow/Slush | 3 | Dark - unknown | 3 | Fog/Smog/Smoke | 3
Wet | 4 | Dark - lights on | 4 | Blowing Sand/Dirt  | 4
Ice | 5 | Dark - no street lights | 5 | Blowing Snow | 5
Oil | 6 | Dark - dark lights off | 6 | Raining | 6
n/a | n/a | n/a | n/a | Snowing | 7
n/a | n/a | n/a | n/a | Severe Crosswind | 8
n/a | n/a | n/a | n/a | Sleet/Hail/Freezing Rain | 9


# Dataset Processing

Data sorted and cleaned to remove null entries and only retain relevant info

All Columns:

<img width="545" alt="Screenshot 2024-10-21 at 11 14 48" src="https://github.com/user-attachments/assets/dcc9b294-3d88-47c5-ba6f-db2a1a45fc39">

Sorted Columns With Null Entries:

<img width="337" alt="Screenshot 2024-10-21 at 11 14 17" src="https://github.com/user-attachments/assets/8bd5c598-d4a1-4a08-a247-92dd3d89ca3b">

Removing Null Entries:

<img width="445" alt="Screenshot 2024-10-21 at 11 18 23" src="https://github.com/user-attachments/assets/e6cd038c-a750-4ff5-996a-91c2a5067d51">

# Fun Graphs I Made From Dataset

Heatmaps:

<img width="566" alt="Screenshot 2024-10-21 at 18 10 04" src="https://github.com/user-attachments/assets/78fc7b3d-24ba-4464-a85e-fc22679d9093">
<img width="542" alt="Screenshot 2024-10-21 at 18 10 47" src="https://github.com/user-attachments/assets/c8622de1-ef24-44d8-bf16-5582d176ff30">

Observation Worth Noting:
under_infl is higher when streetlights are on in the dark and lower during daytime.

<img width="566" alt="Screenshot 2024-10-21 at 18 10 04 copy" src="https://github.com/user-attachments/assets/a156761e-ea4f-4703-8869-aba99c4b0f1e">


Bar Graphs:

<img width="704" alt="Screenshot 2024-10-21 at 12 10 01" src="https://github.com/user-attachments/assets/3d9864b2-de87-4949-af0a-c713c73a2411">
<img width="786" alt="Screenshot 2024-10-21 at 12 09 24" src="https://github.com/user-attachments/assets/592a5dfe-49d0-4d5c-af4a-3d8cc4e6f81c">

Scatterplots:

<img width="567" alt="Screenshot 2024-10-21 at 12 07 59" src="https://github.com/user-attachments/assets/d0392a03-cdd2-4db1-b169-dcb06a7fb537">
<img width="569" alt="Screenshot 2024-10-21 at 12 08 33" src="https://github.com/user-attachments/assets/3ddd5679-367c-4b06-a69c-5a01805308c0">

# Model Performance and Optimization

Best Hyperparameters:

<img width="1512" alt="Screenshot 2024-10-21 at 23 47 51" src="https://github.com/user-attachments/assets/384cfc18-375c-4721-bbfb-6a51863513a8">

**Residual Graphing**

Performance on training set:

<img width="879" alt="Screenshot 2024-10-26 at 16 37 18" src="https://github.com/user-attachments/assets/fb2455b0-7dfc-4b6c-bb8c-989e7d0bb73b">

<img width="289" alt="Screenshot 2024-10-26 at 16 48 50" src="https://github.com/user-attachments/assets/664a553e-7d95-4c92-8f59-56e10b7d35d6">

Performance on test set:

Hyperparamter RFR
<img width="897" alt="hyperparameter tuning" src="https://github.com/user-attachments/assets/9fa05ec2-2c93-4710-add5-97cceed72375">

Polynomial Features RFR
<img width="878" alt="polynomial" src="https://github.com/user-attachments/assets/90bfe717-273c-4b3a-b026-eec5d2e9b7e4">

Model Comparisons:

<img width="1510" alt="Screenshot 2024-10-26 at 14 57 59" src="https://github.com/user-attachments/assets/076c2f83-293c-4a0c-8c7c-da6bfa9b9b90">


# Future Avenues for Better Modeling:
**The distinct patterns in residual plots (bands and non-random distribution) are just a result of using integer danger values**
* Larger data set would likely help in more conclusive modeling
* Address overfitting - hyperparameter tuning likely introduced large amounts of overfitting
* Decomposition (might help to break model up into two steps):
  1. Model to calculate danger modifier for each location (intersection or road)
  2. Model for danger prediction using road conditions then applying danger modifier of the specific location
* Address multicollinearity - although random forest regression helps, strong multicollinearity still exists between values like road conditions and weather
* Test Gradient Boosting or XGBoost - help with residual via iterative modeling
* Test Weighted Least Squares - may help with heteroscedasticity
* Transform or engineer coordinates (e.g., cluster areas)
