# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# %%
df=pd.read_csv('C:\\Users\\Lenovo\\Desktop\\PROGRAMS\\dataset.csv')
df.head()

# %%
df = df[df['Target'] != 'Enrolled']

# %%
df.rename(columns= {"Fathers qualification": "fathers_qualification", "Mothers qualification": "mothers_qualification", "Scholarship holder": "scholarship_holder", "Curricular units 2nd sem (approved)": "curricular_units_2nd_sem_approved", 'Tuition fees up to date': "tuition_fees_up_to_date", "Curricular units 2nd sem (grade)": "get_curricular_units_2nd_sem_grade", "Curricular units 2nd sem (enrolled)": "get_curricular_units_2nd_sem_enrolled", 'Curricular units 1st sem (evaluations)': "get_curricular_units_1st_sem_evaluations", 'Curricular units 1st sem (approved)': "curricular_units_1st_sem_approved", 'Curricular units 1st sem (grade)': 'curricular_units_1st_sem_grade',  'Curricular units 1st sem (enrolled)': 'get_curricular_units_1st_sem_enrolled', 'Curricular units 2nd sem (evaluations)': 'curricular_units_2nd_sem_evaluations'}, inplace=True)

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df is your DataFrame containing the 'Age at enrollment' column
# df = ...

# Define a function for Min-Max scaling
def min_max_scaling(data):
    return (data - data.min()) / (data.max() - data.min())

# Normalize 'Age at enrollment' column using Min-Max scaling
df['Age_Normalized'] = min_max_scaling(df['Age at enrollment'])

# Visualize the distribution before and after normalization
plt.figure(figsize=(10, 5))

# Plot original distribution
plt.subplot(1, 2, 1)
sns.histplot(df['Age at enrollment'], kde=True, color='blue')
plt.title('Original Distribution')

# Plot normalized distribution
plt.subplot(1, 2, 2)
sns.histplot(df['Age_Normalized'], kde=True, color='green')
plt.title('Normalized Distribution')

plt.tight_layout()
plt.show()

# %%
df.drop(columns = ["Age at enrollment"],inplace= True)

# %%
from sklearn.model_selection import train_test_split
y = df['Target']
X = df.drop('Target', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    stratify=y,
                                                    random_state=42)

# %%
train = X_train
train['Target'] = y_train

test = X_test
test['Target'] = y_test

# %%
occ_mapping = {0: "student",
                1: "managerial",
                2: "professional",
                3: "technical",
                4: "professional",
                5: "service",
                6: "agriculture",
                7: "craftsmen",
                8: "factory",
                9: "elementary",
                10: "armed forces",
                90: "unknown",
                99: "unknown",
                101: "armed forces",
                102: "armed forces",
                103: "armed forces",
                112: "managerial",
                114: "managerial",
                121: "professional",
                122: "professional",
                123: "professional",
                124: "professional",
                131: "technical",
                132: "technical",
                134: "technical",
                135: "technical",
                141: "clerical",
                143: "technical",
                144: "clerical",
                151: "service",
                152: "service",
                153: "service",
                154: "service",
                161: "agriculture",
                163: "agriculture",
                171: "craftsmen",
                172: "craftsmen",
                174: "craftsmen",
                175: "craftsmen",
                181: "factory",
                182: "factory",
                183: "factory",
                192: "elementary",
                193: "elementary",
                194: "elementary",
                195: "elementary"}



# %%
prev_qual_mapping = {1:'secondary_school',
                    2: 'graduate',
                    3: 'graduate',
                    4: 'masters', 
                    5: 'doctorate',
                    6: 'unknown',
                    9: 'highschool', 
                    10: 'highschool',
                    12: 'unknown', 
                    14: 'highschool', 
                    15: 'pre-highschool', 
                    19: 'highschool', 
                    38: 'pre-highschool', 
                    39: 'graduate', 
                    40: 'graduate', 
                    42: 'masters',
                    43: 'masters'}

# %%
qual_mapping = {1: "highschool",
                38:"pre highschool",
                37:"pre highschool",
                19:"highschool",
                11:"pre highschool",
                3:"graduate",
                2:"graduate",
                34:"unknown",
                4:"masters",
                27:"pre highschool", 
                12:"unknown",
                39:"graduate",
                42:"masters",
                5:"masters",
                40:"graduate",
                6:"unknown",
                36:"less than 4",
                44:"masters", 
                41:"graduate", 
                29:"pre highschool", 
                30:"pre highschool",  
                9:"highschool", 
                10:"highschool", 
                35:"less than 4", 
                14:"highschool", 
                43:"masters", 
                26:"pre highschool", 
                25:"pre highschool", 
                18:"graduate", 
                22:"masters", 
                31:"graduate", 
                20:"highschool"}


# %%
train["father_qual"] = train["fathers_qualification"].map(qual_mapping)
train["mother_qual"] = train["mothers_qualification"].map(qual_mapping)

# %%
train["father_occ"] = train["Fathers occupation"].map(occ_mapping)
train["mother_occ"] = train["Mothers occupation"].map(occ_mapping)


# %%
train["previous_qual"] = train["Previous qualification"].map(prev_qual_mapping)
test["previous_qual"] = test["Previous qualification"].map(prev_qual_mapping)

# %%
# Print the columns before renaming
print("Columns before renaming:")
print(train.columns)
print(test.columns)

# Rename the column
train.rename(columns={"Daytime/evening attendance": "Attendance_mode"}, inplace=True)
test.rename(columns={"Daytime/evening attendance": "Attendance_mode"}, inplace=True)

# Print the columns after renaming
print("\nColumns after renaming:")
print(train.columns)
print(test.columns)

# Mapping the values
mode_mapping = {0: "Evening", 1: "Daytime"}
train["Attendance_mode"] = train["Attendance_mode"].map(mode_mapping)
test["Attendance_mode"] = test["Attendance_mode"].map(mode_mapping)


# %% [markdown]
# ## Father's Qualification Hypothesis

# %%
from scipy.stats import chi2_contingency
father_cross_tab = pd.crosstab(train["father_qual"],  train["Target"])

stat, p, dof, expected = chi2_contingency(father_cross_tab)

## interpret p value
alpha = 0.05 # 95% confidence level
if p < alpha:
    print(f"P value is {p}")
    print("Null hypothesis is rejected.")
else:
    print(f"P value is {p}")
    print("Failed to reject the Null hypothesis.")

# %% [markdown]
# ## Mother's Occupation

# %%
from scipy.stats import chi2_contingency
father_cross_tab = pd.crosstab(train["scholarship_holder"],  train["Target"])

stat, p, dof, expected = chi2_contingency(father_cross_tab)

## interpret p value
alpha = 0.05 # 95% confidence level
if p < alpha:
    print(f"P value is {p}")
    print("Null hypothesis is rejected.")
else:
    print(f"P value is {p}")
    print("Failed to reject the Null hypothesis.")

# %%


# %%
#import libraries

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

from sklearn.model_selection import StratifiedShuffleSplit, cross_validate, GridSearchCV

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier

from sklearn.metrics import classification_report, ConfusionMatrixDisplay, f1_score

from sklearn import set_config

from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as imbpipeline

# %%
df.columns

# %%
features_to_include = ['fathers_qualification', 'mothers_qualification', 'curricular_units_1st_sem_approved', 'curricular_units_2nd_sem_approved', 'get_curricular_units_2nd_sem_grade', 'curricular_units_1st_sem_grade', 'curricular_units_2nd_sem_evaluations', 'tuition_fees_up_to_date', 'get_curricular_units_1st_sem_evaluations', 'Age_Normalized', 'get_curricular_units_1st_sem_enrolled', 'scholarship_holder', 'Debtor', 'get_curricular_units_2nd_sem_enrolled', 'Course', 'Gender','Target']

# %%
train_df = train[features_to_include]
test_df = test[features_to_include]

# %%
train_df.columns


# %%


# %%


# %%
df.columns

# %%
y_train = train_df["Target"]
X_train = train_df.drop("Target", axis=1)

y_test = test_df["Target"]
X_test = test_df.drop("Target", axis=1)

# %%
cat_features = ['Course', 'scholarship_holder', 'Debtor', 'mothers_qualification','fathers_qualification', 'Gender']
num_features = ['curricular_units_1st_sem_approved', 'curricular_units_2nd_sem_approved', 'get_curricular_units_2nd_sem_grade', 'curricular_units_1st_sem_grade', 'curricular_units_2nd_sem_evaluations', 'tuition_fees_up_to_date', 'get_curricular_units_1st_sem_evaluations', 'get_curricular_units_1st_sem_enrolled', 'get_curricular_units_2nd_sem_enrolled']
ordinal_features = ['Age_Normalized']

# %%
X_train

# %%
# import pandas as pd

# # Define the column names
# column_names = ['Fathers qualification', 'Mothers qualification', 'Curricular units 1st sem (approved)',
#                 'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)',
#                 'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (evaluations)',
#                 'Tuition fees up to date', 'Curricular units 1st sem (evaluations)',
#                 'Age_Normalized', 'Curricular units 1st sem (enrolled)', 'Scholarship holder',
#                 'Debtor', 'Curricular units 2nd sem (enrolled)', 'Course', 'Gender']

# # Input sample as a DataFrame
# input_sample = pd.DataFrame([[10, 0, 0, 2, 28, 1, 10, 10, 10, 10, 15, 1, 14, 11, 10, 0.075472]], columns=column_names)

# # Make the prediction
# prediction = logistic_clf.predict(input_sample)

# # Print the prediction
# print("Prediction:", prediction)

# %%
select_cat_features = ColumnTransformer([('select_cat', 'passthrough', cat_features)])
cat_transformers = Pipeline([('selector', select_cat_features),
                            ('onehot', OneHotEncoder(handle_unknown='ignore')),
                            ])

select_ord_features = ColumnTransformer([('select_cat', 'passthrough', ordinal_features)])
ordinal_transformers = Pipeline([('selector', select_ord_features),
                            ('ordinal_enc', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
                            ])
                            
select_num_features = ColumnTransformer([('select_num', 'passthrough', num_features)])
num_transformers = Pipeline([('selector', select_num_features),
                            ('scaler', StandardScaler()),
                            ])

preprocess_pipe = FeatureUnion([('cat', cat_transformers),
                                ('ord', ordinal_transformers),
                                ('num', num_transformers),
                                ])
set_config(display="diagram")
preprocess_pipe

# %% [markdown]
# ## Modelling

# %%
class Model:
    def __init__(self, model_name, estimator, preprocessor=None, scoring=None, cv=5, random_state=42):
        self.name = model_name
        self.estimator = estimator
        self.preprocess = preprocessor
        self.scoring = scoring
        self.cv = cv
        self.rs = random_state

    def make_model_pipeline(self):
        self.model = Pipeline([('preprocess', self.preprocess),
                               ('model', self.estimator)])

    def train(self, X_train, y_train):
        '''Trains the model
        Args:
            X_train: Training data feature matrix
            y_train: Training data label vector
            
        Returns:
            trained model
        '''
        self.make_model_pipeline()
        self.cv_results = cross_validate(self.model, X_train, y_train, cv=self.cv, scoring=self.scoring, return_train_score=True)

        mean_train_score = self.cv_results["train_score"].mean()
        mean_val_score = self.cv_results["test_score"].mean()
        
        std_train_score = self.cv_results["train_score"].std()
        std_val_score = self.cv_results["test_score"].std()

        print(f"Cross validated training results for {self.name} model")
        print("---------------------------------------------------------")
        print(f"Train score: {mean_train_score} +/- {std_train_score}" )
        print(f"Validation score: {mean_val_score} +/- {std_val_score}" )
        
        self.fitted_model = self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.fitted_model.predict(X_test)

        f1 = f1_score(y_test, y_pred, average="micro")
        print("F1 score on test set: ", f1)
        print()
        print(classification_report(y_test, y_pred))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    
    def tune(self, param_grid, X_train, y_train):
        '''Do hyper parameter tuning using GridSearch strategy
        
        Args:
            model: Model to be tuned
            param_grid: dict of parameters
            X_train: Feature matrix
            y_train: Label matrix
            
        Returns: 
            best parameters
            best estimator
        '''
        self.make_model_pipeline()
        search = GridSearchCV(self.model, param_grid=param_grid, cv=self.cv, scoring=self.scoring, return_train_score=True)
        
        search.fit(X_train, y_train)
        
        print("Best parameters: ", search.best_params_)
        
        print("-------------------Best model performance --------------------------")
        
        mean_train_score = search.cv_results_['mean_train_score'][search.best_index_]
        mean_val_score = search.cv_results_['mean_test_score'][search.best_index_]
        std_train_score = search.cv_results_['std_train_score'][search.best_index_]
        std_val_score = search.cv_results_['std_test_score'][search.best_index_]

        print(f"Score of the model on the train set:\n"
              f"{mean_train_score:.3f} +/- {std_train_score:.6f}")

        print(f"Score of the model on the validation set:\n"
              f"{mean_val_score:.3f} +/- {std_val_score:.6f}")
        
        self.fitted_model = search.best_estimator_

    def predict(self, X):
        '''Makes predictions using the trained model
        Args:
            X: Feature matrix for which to make predictions
            
        Returns:
            y_pred: Predicted labels
        '''
        if not hasattr(self, 'fitted_model'):
            raise ValueError("The model must be trained before making predictions.")
        
        y_pred = self.fitted_model.predict(X)
        return y_pred


# %%
##Let's supress sklearn warnings

def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn
## Cross validation scheme

cv = StratifiedShuffleSplit(n_splits = 5, test_size=0.2, random_state=32)

# %%
from sklearn.exceptions import FitFailedWarning
import warnings

baseline_clf = Model(
    model_name="Baseline classifier", 
    estimator=DummyClassifier(),
    preprocessor=preprocess_pipe, 
    scoring="f1_micro",
    cv=cv,
    random_state=32
)

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=FitFailedWarning)
        baseline_clf.train(X_train, y_train)
except FitFailedWarning as e:
    print("An error occurred during training:")
    print(e)
except Exception as e:
    print("An unexpected error occurred during training:")
    print(e)


# %%
logistic_clf = Model(model_name="Logistic Regression", 
                    estimator=LogisticRegression(penalty = "elasticnet", solver="saga", l1_ratio=0,  max_iter=1000),
                    preprocessor=preprocess_pipe,
                    scoring="f1_micro",
                    cv = cv,
                    random_state=32)

# %%
logistic_clf.train(X_train, y_train)

# %%
#param_grid = {"model__C": [0.01, 0.1, 1, 10, 100],
 #           "model__l1_ratio": np.linspace(0, 1, 11)}
#logistic_clf.tune(param_grid, X_train, y_train)

# %%
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    "model__C": [0.01, 0.1, 1, 10, 100],
    "model__l1_ratio": np.linspace(0, 1, 11)
}

random_search = RandomizedSearchCV(
    estimator=logistic_clf.model,
    param_distributions=param_dist,
    n_iter=10,  
    scoring="accuracy",
    cv=logistic_clf.cv,
    n_jobs=-1,  
    random_state=42
)

random_search.fit(X_train, y_train)

print("Best parameters:", random_search.best_params_)

# %%
logistic_clf.evaluate(X_test, y_test)

# %%
#features_to_include = ['Fathers qualification', 'Mothers qualification', 'Curricular units 1st sem (approved)', 'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)', 'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (evaluations)', 'Tuition fees up to date', 'Curricular units 1st sem (evaluations)', 'Age_Normalized', 'Curricular units 1st sem (enrolled)', 'Scholarship holder', 'Debtor', 'Curricular units 2nd sem (enrolled)', 'Course', 'Gender']

# %%
# logistic_clf.predict([["Biofuel Production Technologies", "Yes", "Yes", "Secondary Education - 12th Year of Schooling or Eq.","Secondary Education - 12th Year of Schooling or Eq.", "Male", 20, 20, 10, 9, 1, 4, 5, 0.3]])

# %%
ridge_clf = Model(model_name = "Ridge classifier", 
                    estimator = RidgeClassifier(),
                    preprocessor = preprocess_pipe,
                    scoring = "f1_micro",
                    random_state = 32)

# %%
ridge_clf.train(X_train, y_train)

# %%
param_grid = {"model__alpha": [0.01, 0.1, 1, 10, 100]}
ridge_clf.tune(param_grid, X_train, y_train)

# %%
ridge_clf.evaluate(X_test, y_test)

# %%
## SVM

# %%
svm_clf = Model(model_name = "SVM classifier", 
                    estimator = LinearSVC(),
                    preprocessor = preprocess_pipe,
                    scoring = "f1_micro",
                    random_state = 32)

# %%
svm_clf.train(X_train, y_train)

# %%
param_grid = {
    "model__penalty": ["l1", "l2"],
    "model__loss": ["squared_hinge"],
    "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
    "model__dual": [False]  # Set dual to False for l1 penalty
}

svm_clf.tune(param_grid, X_train, y_train)

# %%
svm_clf.evaluate(X_test, y_test)

# %%
## Decision Tree

# %%
tree_clf = Model(model_name = "Decision tree classifier", 
                    estimator = DecisionTreeClassifier(min_samples_leaf=10, max_depth=50, max_features=16, max_leaf_nodes= 10),
                    preprocessor = preprocess_pipe,
                    scoring = "f1_micro",
                    random_state = 32)

# %%
tree_clf.train(X_train, y_train)

# %%
param_grid = {"model__max_depth": [2, 3, 4, 5, 10],
                "model__min_samples_split":[2, 4, 6, 8, 10],
                "model__min_samples_leaf": [1, 2, 3, 6, 7, 8]}

tree_clf.tune(param_grid, X_train, y_train)

# %%
tree_clf.evaluate(X_test, y_test)

# %%
## Random Forest

# %%
rf_clf = Model(model_name = "Random forest classifier", 
                    estimator = RandomForestClassifier(min_samples_split=8,min_samples_leaf=6, max_depth=15, n_estimators=150),
                    preprocessor = preprocess_pipe,
                    scoring = "f1_micro",
                    random_state = 32)
rf_clf.train(X_train,y_train)

# %%
#param_grid = {"model__n_estimators": [125, 150, 175],
 #               "model__max_depth": [3, 4, 5, 10, 15],
  #              "model__min_samples_split":[2, 4, 6, 8, 10],
   #             "model__min_samples_leaf": [1, 2, 3, 6, 7, 8]}

#rf_clf.tune(param_grid, X_train, y_train)

# %%
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

# Create a base RandomForestClassifier
rf_base = RandomForestClassifier(random_state=42)

# Parameter grid for RandomizedSearchCV
param_dist = {
    "n_estimators": [125, 150, 175],
    "max_depth": [3, 4, 5, 10, 15],
    "min_samples_split": [2, 4, 6, 8, 10],
    "min_samples_leaf": [1, 2, 3, 6, 7, 8]
}

n_iter = 10

# RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_dist,
    n_iter=n_iter,
    scoring="accuracy",
    cv=5,  
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)

# Print the best parameters found
print("Best parameters:", random_search.best_params_)

# %%
rf_clf.evaluate(X_test, y_test)

# %%
dtree = tree_clf.fitted_model

# %%
from sklearn.tree import plot_tree

# %%
plot_tree(dtree[-1]);
plt.figure(figsize=(40, 40))

# %%
df.Target.value_counts()

# %%
from sklearn.ensemble import AdaBoostClassifier
ada_boost_clf=Model(model_name="AdaBoostClassifier", estimator= AdaBoostClassifier(), random_state=42 )
ada_boost_clf.train(X_train,y_train)

# %%
ada_boost_clf=AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=5,random_state=42), n_estimators=50, learning_rate=0.3,random_state=42)

# %%
ada_boost_clf.fit(X_train,y_train)

# %%
xscore=ada_boost_clf.score(X_test, y_test)
xscore

# %%
import pickle

# Assuming logistic_clf is your trained model object
with open('rf_clf.pkl', 'wb') as f:
    pickle.dump(rf_clf, f)

print("Model saved successfully.")

# %%
import dill as pickle

# loading the trained model
with open('rf_clf.pkl', 'rb') as file:
    classifier = pickle.load(file)


# %%



