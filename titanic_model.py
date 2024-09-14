'''
Project Streamlit
modeling the Titanic dataset

- Course Name :         Applied Machine Learning
- Course instructor:    Sohail Tehranipour
- Student Name :        Afshin Masoudi Ashtiani
- Chapter 7 -           Building a Web App for Data Scientists
- Project:              Streamlit Project
- Date :                September 2024
'''

# Step 1: Install required libraries
# pip install pandas scikit-learn xgboost joblib

# Step 2: Import required libraries
import os
import re
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import ExtraTreeClassifier
from tabulate import tabulate
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score, cohen_kappa_score, matthews_corrcoef

# Step 3: Load the dataset
def load_dataset(path:str = r'./data/train.csv', display:bool = True) -> tuple:
    """Load the dataset from a CSV file."""
    df = pd.read_csv(path)
    X = df.drop(labels='Survived', axis=1)
    y = df.Survived
    if display: print(tabulate(df[:10], headers='keys', tablefmt='psql'))
    return X, y

# Step 5: Pre-Process the data
class PreProcessor(BaseEstimator, TransformerMixin): 
    def fit(self, X, y=None): 
        self.ageImputer = SimpleImputer()
        self.ageImputer.fit(X[['Age']])        
        return self 
        
    def transform(self, X, y=None):
        X['Age'] = self.ageImputer.transform(X[['Age']])
        X['CabinClass'] = X['Cabin'].fillna('M').apply(lambda x: str(x).replace(' ', '')).apply(lambda x: re.sub(r'[^a-zA-Z]', '', x))
        X['CabinNumber'] = X['Cabin'].fillna('M').apply(lambda x: str(x).replace(' ', '')).apply(lambda x: re.sub(r'[^0-9]', '', x)).replace('', 0) 
        X['Embarked'] = X['Embarked'].fillna('M')
        X = X.drop(['PassengerId', 'Name', 'Ticket','Cabin'], axis=1)
        return X

# Step 6: Save the model
def save_pipeline(pipeline:Pipeline, name:str) -> str:
    if pipeline:
        file_name = f'{(''.join(cap for cap in str(name) if cap.isupper())).lower()}pipe.joblib'
        joblib.dump(value= pipeline, filename= './models/' + file_name) 
        return './models/' + file_name   
    else:
        print("> No pipeline found to save ...!")
        return ''

# Step 7: Create pipelines
def create_pipelines(X_train: np.ndarray, y_train: np.ndarray, display:bool = True) -> pd.DataFrame:
    """Create pipelines for training."""
    preprocessor = PreProcessor()
    numeric_pipeline = Pipeline([('Scaler', StandardScaler())])
    categorical_pipeline = Pipeline([('OneHot', OneHotEncoder(handle_unknown='ignore'))])
    transformer = ColumnTransformer([('num', numeric_pipeline, ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'CabinNumber']), ('cat', categorical_pipeline, ['Sex', 'Embarked', 'CabinClass'])])
    
    lrpipe = Pipeline([('InitialPreProc', preprocessor), ('Transformer', transformer), ('Logistic Regression', LogisticRegression())]).fit(X_train, y_train)
    rcpipe = Pipeline([('InitialPreProc', preprocessor), ('Transformer', transformer), ('Ridge Classifier', RidgeClassifier())]).fit(X_train, y_train)
    etcpipe = Pipeline([('InitialPreProc', preprocessor), ('Transformer', transformer), ('Extra Tree Classifier', ExtraTreeClassifier())]).fit(X_train, y_train)
    xgbcpipe = Pipeline([('InitialPreProc', preprocessor), ('Transformer', transformer), ('XGB Classifier', XGBClassifier())]).fit(X_train, y_train)

    pipelines_df = pd.DataFrame([
        {'Model' : 'LogisticRegression', 'Filename' : save_pipeline(lrpipe, 'LogisticRegression')}, 
        {'Model' : 'RidgeClassifier', 'Filename' : save_pipeline(rcpipe, 'RidgeClassifier')}, 
        {'Model' : 'ExtraTreeClassifier', 'Filename' : save_pipeline(etcpipe, 'ExtraTreeClassifier')}, 
        {'Model' : 'XGBClassifier', 'Filename' : save_pipeline(xgbcpipe, 'XGBClassifier')}])

    if display: print(tabulate(pipelines_df, headers='keys', tablefmt='psql'))
    
    return pipelines_df

# Step 8: Evaluate the models
def evaluate_models(pipelines_df: pd.DataFrame, X_test: np.ndarray, y_test: np.ndarray, display:bool = True) -> pd.DataFrame:
    """Evaluate the models."""
    eval_list = []
    index_list = []

    if len(pipelines_df) != 0:
        start_time = time.process_time()
        for index, row in pipelines_df.iterrows():
            model = joblib.load(filename= row.Filename)
            y_pred = model.predict(X_test)
            current_time = time.process_time()
            
            eval_list.append({
                'Model' : row.Model, 
                'Accuracy' : accuracy_score(y_test, y_pred), 
                'AUC' : roc_auc_score(y_test, y_pred), 
                'Recall' : recall_score(y_test, y_pred), 
                'Prec.' : precision_score(y_test, y_pred), 
                'F1' : f1_score(y_test, y_pred),
                'Kappa' : cohen_kappa_score(y_test, y_pred),
                'MCC' : matthews_corrcoef(y_test, y_pred),
                'TT (Sec)' : current_time - start_time})
            index_list.append((''.join(cap for cap in str(row.Model) if cap.isupper())).lower())

            start_time = current_time
    else:
        print("> No model found to evaluate ...!")
        return pd.DataFrame()
    
    eval_df = pd.DataFrame(eval_list, index= index_list)

    if display: print(tabulate(eval_df, headers='keys', tablefmt='psql'))

    return eval_df

# Step 9: Find the best model
def find_best_model(eval_df: pd.DataFrame) -> tuple:
    """Find the best model."""
    best_index = eval_df.Accuracy.idxmax()
    best_name = eval_df.loc[best_index, 'Model']
    print(f'>> The best model is {best_name}.')
    best_path = f'./models/{best_index}pipe.joblib'
    best_model = joblib.load(best_path)
    return best_name, best_path, best_model

# Step 10: Predict the dataset
def predict_data(df: pd.DataFrame, model_path: str, display:bool = True) -> pd.DataFrame:
    """Predict the testing dataset."""
    if os.path.isfile(model_path):
        model = joblib.load(filename= model_path)
        y_pred = model.predict(df)
        y_pred_df = pd.DataFrame(y_pred, columns= ['Prediction'])

        if display: print(tabulate(y_pred_df[:10], headers='keys', tablefmt='psql'))
        
        return y_pred_df
    else:
        print("> No model found to predict ...!")
    
    return pd.DataFrame()
    
def main():
    train_path = r'./data/train.csv'
    display = True
    test_size = 0.1
    test_path = r'./data/test.csv'
    # Step 3: Load the dataset
    X, y = load_dataset(train_path, display)
    # Step 4: Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state= 123)
    # Step 7: Create pipelines
    pipelines_df = create_pipelines(X_train, y_train, display)
    # Step 8: Evaluate the models
    eval_df = evaluate_models(pipelines_df, X_test, y_test, display)
    # Step 9: Find the best model
    name, path, model = find_best_model(eval_df)
    # Step 10: Predict the dataset
    test_df = pd.read_csv(test_path)
    y_pred = predict_data(test_df, path)

if __name__ == '__main__':
    main()
    