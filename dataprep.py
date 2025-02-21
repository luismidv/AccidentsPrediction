import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_regression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns



class DataExtractor():

    def datapreparer(self,route):
        self.dataset = pd.read_csv(route)
        dataset_columns = self.dataset.columns
        mising_col = [col for col in self.dataset.columns if self.dataset[col].isnull().sum()>0]
        if len(mising_col) > 0:
            print("There is missing information")
        else:
            print("There is no missing data")

    def redundant_data(self):
        print(f"Dataset size {len(self.dataset)}")

        oneh = OneHotEncoder(sparse_output=False)
        encoded_data = oneh.fit_transform(self.dataset)
        encoded_dataframe = pd.DataFrame(encoded_data)
        plt.figure(figsize=(12,12))
        sns.heatmap(np.corrcoef(encoded_data, rowvar=False) > 0.6, annot = True, cbar = False)
        plt.show()

    def model_charging(self):
        models = {
            'linear regression' : {
                'model' : LinearRegression(),
                'parameters' : {

                }
            },

            'lasso' : {
                'model' : Lasso(),
                'parameters': {
                    'alpha': [1,2],
                    'selection': ['random', 'cyclic']
                }
            },

            'svr' : {
                'model': SVR(),
                'parameters': {
                    'gamma' : ['auto', 'scale']
                }
            },

            'random_forest': {
                'model' : RandomForestRegressor(criterion = 'squared_error'),
                'parameters': {
                    'n_estimators' : [5,10,15,20]
                }
            },
            'knn' : {
                'model' : KNeighborsRegressor(algorithm = 'auto'),
                'parameters': {
                    'n_neighbors' : [2,5,10,20]
                }
            }
        }

        return models
        

    def grid_preprocessor(self, models):

        oneh = OneHotEncoder(sparse_output=False)
        label = LabelEncoder()
        kFold = StratifiedKFold(n_splits = 5)

        print(self.dataset)

        features = self.dataset.drop('Industry Sector', axis = 'columns')
        encoded_features = oneh.fit_transform(features)
        print(f"Type after encoded features {type(encoded_features)}")

        
        labels = self.dataset['Industry Sector']
        print(type(labels))
        labels = pd.DataFrame(labels)
        encoded_labels = label.fit_transform(labels)
        print(f"Type after encoded labels {type(encoded_labels)}")

        

        scores = []

        for model_name, model_params in models.items():
            print(f"Using model {model_name}")
            gs = GridSearchCV(model_params['model'], model_params['parameters'], cv = 10, return_train_score=False)
            gs.fit(encoded_features,encoded_labels)
            scores.append({
                'model' : model_name,
                'best_parameters' : gs.best_params_,
                'score' : gs.best_score_
            })

            x_train, x_test, y_train, y_test = train_test_split(encoded_features, encoded_labels, test_size =0.2, random_state=0)
            predictions = gs.predict(x_test)

        return pd.DataFrame(scores, columns = ['model', 'best_parameters', 'score'])
    

route = "./archive/IHMStefanini_industrial_safety_and_health_database.csv"
new_data_extractor = DataExtractor()
new_data_extractor.datapreparer(route)
#new_data_extractor.redundant_data()

models = new_data_extractor.model_charging()
data = new_data_extractor.grid_preprocessor(models)
print(data)
