import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def Inspect_Data(df):
    print('The type and number of the data:','\n',df.info())
    print('*'*40)
    print('The unique number of the features in the data:','\n',df.nunique())
    print('*'*40)
    print('The outlook of the Y we want to analyse:','\n',df.loc[:,Y_col].value_counts())

class Feature_Filter:
    def __init__(self,df_without_Y,Y):
        self.X = df_without_Y
        self.y = Y
        
    def corr_map(self):
        numCol = []
        for col in self.X:
            if self.X[col].dtype == float:
                numCol.append(col)
        corr = self.X[numCol].corr()
        ax = sns.heatmap(
            corr,
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True)
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45, # x-ticks rotation
            horizontalalignment='right')
        
    def Random_Forest_Importance(self): 
        scaler = StandardScaler()
        scaler.fit(self.X)
        X_standard= scaler.transform(self.X)
        
        forest = RandomForestClassifier()
        forest.fit(X_standard, self.y)
        importances = forest.feature_importances_
        indices = np.argsort(importances)[::-1] # get the ranking's original index
        dic = {'feature':[],'importance':[]}
        for ind in range(X_standard.shape[1]):
            dic['feature']+=[self.X.columns[indices[ind]]] 
            dic['importance']+= [round(importances[indices[ind]], 4)]
        pd.DataFrame(dic).set_index('feature').plot(kind = 'bar')
        
    def Logistic_Regression_Importance(self, penalty = 10):
        scaler = StandardScaler()
        scaler.fit(self.X)
        X_standard= scaler.transform(self.X)
        
        LRmodel_l1 = LogisticRegression(penalty="l1", C = penalty, solver='liblinear')
        LRmodel_l1.fit(X_standard, self.y)
        indices = np.argsort(abs(LRmodel_l1.coef_[0]))[::-1]
        dic = {'feature':[],'importance':[]}
        for ind in range(X_standard.shape[1]):
            dic['feature']+=[self.X.columns[indices[ind]]] 
            dic['importance']+= [round(LRmodel_l1.coef_[0][indices[ind]], 4)]
        pd.DataFrame(dic).set_index('feature').plot(kind = 'bar')


def Train_Model(X, Y, model_names, model_list, test_size_ = 0.2):
    scaler = StandardScaler()
    scaler.fit(X)
    X_standard= scaler.transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_standard, Y, random_state=42, test_size = test_size_)

    best = 0
    count = 0

    for i in range(len(model_list)):
        cv_score = model_selection.cross_val_score(model_list[i], X_train, y_train, cv=5)
        if cv_score.mean() > best:
            best = cv_score.mean()
            count = i

    print('{} has the best performance in these four models'.format(model_names[count]))
    model_list[count].fit(X_train, y_train)
    prediction_Regression = classifier_logistic.predict(X_test)
    print(classification_report(y_test,prediction_Regression))
    
class Inspect_Features:
    def __init__(self,df_without_Y):
        self.X = df_without_Y
        
    def Geo_distribution(self):
        import plotly.graph_objects as go
        df = self.X
        df_location = df.groupby("addr_state",).sum().reset_index()
        df_location = df_location[["addr_state", "loan_amnt"]]
        fig = go.Figure(data=go.Choropleth(
            locations=df_location['addr_state'], 
            z = df_location['loan_amnt'].astype(float), 
            locationmode = 'USA-states',
            colorscale = 'Reds',
            colorbar_title = "USD",))
        fig.update_layout(
            title_text = 'Total amount issued by State',
            geo_scope='usa')
        fig.show()
    
    def Count_Features(self,inspect_list,Y_col):
        df = self.X
        _,axss = plt.subplots(3,2, figsize=[30,20])
        hue_lst = inspect_list
        i = 0
        j = 0
        for hue_ in hue_lst:
            col = i // 2
            row = j % 2
            sns.countplot(x=Y_col, hue=hue_, data=df, ax=axss[col][row])
            i = i+1
            j = j+1
    
def Labelling(df):
    numCol = []
    for col in df:
        if df[col].dtype == 'object':
            numCol.append(col)

    for col in numCol:
        try:
            mapping = {label:idx for idx, label in enumerate(np.unique(df[col]))}
        except TypeError:
            pass
        else:
            df[col]=df[col].map(mapping)
            
    return df

#Import data, denote Y 
df = pd.read_csv('loan-clean-version.csv')
Y_col = 'loan_status'

#Get X and Y
X = df.drop([Y_col], axis=1)
Y = df.loc[:,Y_col]

#Start the project
# Inspect data
Inspect_Data(df)

# Inspect corr_map
corr_ = Feature_Filter(X,Y)
corr_.corr_map()

#Inspect features distribution
F_D = Inspect_Features(X)
inspect_lst = ['term','grade','emp_length','home_ownership','verification_status','purpose']
F_D.Count_Features(inspect_lst,Y)
F_D.Geo_distribution()

#Train Model
df = Labelling(df)
df = df.select_dtypes(include=[np.number]).interpolate().dropna()
drop_lst=['total_pymnt','total_pymnt_inv','total_rec_int','id','total_rec_prncp']
for item in drop_lst:
    df = df.drop([item], axis=1)

classifier_logistic = LogisticRegression()
classifier_KNN = KNeighborsClassifier()
classifier_RF = RandomForestClassifier()
classifier_SVC = SVC()
model_names = ['Logistic Regression','KNN','Random Forest','SVC']
model_list = [classifier_logistic, classifier_KNN, classifier_RF,classifier_SVC]
test_size=0.2

X = df.drop([Y_col], axis=1)
Y = df.loc[:,Y_col]
Train_Model(X, Y, model_names, model_list, test_size)

#feature selection
F_I = Feature_Filter(X,Y)
print('Feature Selection with Random Forest:')
F_I.Random_Forest_Importance()
F_I.Logistic_Regression_Importance()
