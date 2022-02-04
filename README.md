# CEMENT STRENGTH PREDICTION
---
A project to determine the concrete compressive strength (MPa) for a given mixture based on 
parameters like Blast Furnace Slag ,Fly Ash ,Cement ,Coarse Aggregate etc. Goal of the  project is to create a Regression model with modular coding and find the best model by checking with R-Squared score or Mean Squared error.A sample prediction has been done . Initial results on the validation set indicate R-Ssquared score of around 89%-91% with Gradient Boosting Regressor to be the best performing model .

# WORKFLOW
---
### DATA COLLECTION 
Kaggle - https://www.kaggle.com/niteshyadav3103/cement-strength-prediction/data

### DATA PREPROCESSING
- Data checked for possible null values ,analyzed for correlation ,outliers and placed in a new SQLite database 
- Feature values standardized by using Standard Scaler. 
- Some column feature values needed to be converted to normal distribution.

### MODEL CREATION
- Data was separated into clusters based on K-Means algorithm and each cluster was used to run multiple regression algorithms and the best model for each cluster was found . 
- As new data comes in , it gets classified into one of the clusters and the applicable ML model is run on the data to provide the result.

### MDOEL DEPLOYMENT
- Upcoming...




