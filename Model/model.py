#%% https://paperswithcode.com/dataset/forest-covertype
import numpy as np
import torch
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns

#%%  Data Investigation
df=pd.read_csv('../Data/covtype.csv')
df=df[:100000]
print(f"Data set contain {len(df)} rows, with {len(df.keys())} columns.")
print(f"An easy way to access base stats of the data frame is the .describe() method \n{df.describe()}")

# %% Data visualization: distribution of our numeric data
metrics=['Meters', 'Compass Degree', 'Degree from Horizontal', 'Meters', 'Meters' , 'Meters', 
         'Index, 0-256', 'Index, 0-256', 'Index, 0-256','Meters', 'Area', ' Classes']

def hist_plotter(df, metrics):
    fig, axs = plt.subplots(3,4, figsize=(15, 12), facecolor='w', edgecolor='k')
    fig.tight_layout(pad=0.4, w_pad=2, h_pad=3.0)
    axs = axs.ravel()
    for i in range(11):
        sns.histplot(df[df.keys()[i]], ax=axs[i])
        axs[i].set_title(df.keys()[i])
        axs[i].set_xlabel(metrics[i])

    sns.histplot(df['Cover_Type'], ax=axs[i+1])
    axs[i+1].set_title('Cover_Type')
    axs[i+1].set_xlabel(metrics[i+1])   #----> What type of task can we preform with this type of data?

hist_plotter(df, metrics)
# %% Feature Clustering
# Hands on portion: We wish to choose variables that separate the different species of vegetation.
#                   Use pairplot_features to see if the data can be separated.

# df.keys()==['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
#        'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
#        'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
#        'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1',
#        'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',
#        'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5',
#        'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10',
#        'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14',
#        'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18',
#        'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22',
#        'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26',
#        'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30',
#        'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34',
#        'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38',
#        'Soil_Type39', 'Soil_Type40', 'Cover_Type']

pairplot_features=['Elevation', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Horizontal_Distance_To_Fire_Points', 'Cover_Type']

sns.pairplot(df[pairplot_features].head(10000), hue='Cover_Type')
# %% Can we do better using a classic classifier model?  All distance based classifiers require scaling to ensure all features 
# have similar importance in the classification of data. 
from sklearn.preprocessing import StandardScaler , MaxAbsScaler, MinMaxScaler, RobustScaler, PowerTransformer, Normalizer

s=StandardScaler()  #---->  Hands on: experiment with the different scalers, listed above, to see how the data is transformed
#                           for your previous feature selection

scaled_data=s.fit_transform(df.drop('Cover_Type', axis=1)) #scaler returns a np.array
scaled_df=pd.DataFrame(scaled_data, columns=df.keys()[:-1])
scaled_df['Cover_Type']=df['Cover_Type']
hist_plotter(scaled_df, metrics)

sns.pairplot(scaled_df[pairplot_features].head(10000), hue='Cover_Type')


# %% Data separation (Supervised Modeling)
# We wish to build a classifier model. This model should be able to predict the Cover_Type of a previously unseen data point.
# We need to separate our labeled data in to a training and testing set, for the model to learn.
from sklearn.model_selection import train_test_split 

#first we will separate our dependant and independent variables. 
X=scaled_df[scaled_df.keys()[:-1]]
y=scaled_df['Cover_Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(f"The length of the train (X,y):({len(X_train)}, {len(y_train)})")
print(f"The length of the test  (X,y):({len(X_test)}, {len(y_test)})")
print(f"The ratio of test/total: {len(X_test)/len(scaled_df)}")

# %% Modeling 
# Sevral classic classifer models exist. They very based on the methods used to sort data into different classes. Each has
# unique hyperparams that may be used to adjust the model. By adjusting the model we determine how well the model fits the
# training and testing data. 
####################### K Nearest Neighbors (pooled voting) ##########################
#                                                        Documentation for HyperParameters 
from sklearn.neighbors import KNeighborsClassifier  # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
time1=time.time()
# training
k_model=KNeighborsClassifier(n_neighbors=15, n_jobs=-1) #----> Hands on: Change hyper parameter value to better fit the model
k_model.fit(X_train, y_train)

# scoring
kte_accuracy=k_model.score(X_test, y_test)
ktr_accuracy=k_model.score(X_train, y_train)
time2=time.time()
print(f"Kneighbors train score:{ktr_accuracy}, testing score: {kte_accuracy}, runtime:{np.round(time2-time1, 2)}")

# %% ####################### Random Forrest Classifier (best split ensemble method) ##########################
#                                                        Documentation for HyperParameters 
from sklearn.ensemble import RandomForestClassifier # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
time1=time.time()
# training
r_model=RandomForestClassifier(n_estimators=1000, max_depth=20, min_samples_split=5, n_jobs=-1)  #----> Hands on: Change hyper parameter value to better fit the model
r_model.fit(X_train, y_train)

# scoring
rte_accuracy=r_model.score(X_test, y_test)
rtr_accuracy=r_model.score(X_train, y_train)
time2=time.time()
print(f"RandomForrest train score:{rtr_accuracy}, testing score: {rte_accuracy}, runtime:{np.round(time2-time1, 2)}")

# %% ####################### Multi Layer Perceptron (Artificial Neural Network) ##########################
#                                                       Documentation for HyperParameters 
from sklearn.neural_network import MLPClassifier    # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
time1=time.time()
# training
n_model=MLPClassifier(alpha=.0001, learning_rate='adaptive', activation='relu', max_iter=500)  #----> Hands on: Change hyper parameter value to better fit the model
n_model.fit(X_train, y_train)

# scoring
nte_accuracy=n_model.score(X_test, y_test)
ntr_accuracy=n_model.score(X_train, y_train)
time2=time.time()
print(f"Neural Net train score:{ntr_accuracy}, testing score: {nte_accuracy}, runtime:{np.round(time2-time1, 2)}")


# %%
