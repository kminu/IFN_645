# dm_tools.py
import numpy as np
import pandas as pd
from io import StringIO
from sklearn.tree import export_graphviz

def get_data():
    
    data = pd.read_csv("datasets/online_shoppers_intention.csv")
    data=data.dropna(subset=['Administrative', 'Administrative_Duration','Informational','Informational_Duration','ProductRelated','ProductRelated_Duration','BounceRates','ExitRates'])
    
    data['VisitorType'] = data['VisitorType'].replace('Other', 'Returning_Visitor')
    
    #data = pd.concat([data,pd.get_dummies(data['Month'], prefix='Month', prefix_sep='_', columns= (''))], axis=1)
    data = pd.concat([data,pd.get_dummies(data['VisitorType'], prefix='VisitorType', prefix_sep='_', columns= (''))], axis=1)
    #data.drop(['Month'],axis=1, inplace=True)
    data.drop(['VisitorType'],axis=1, inplace=True)
    
    data['Administrative_Duration'] = data["Administrative_Duration"].replace('-1', 0)
    data['Informational_Duration'] = data["Informational_Duration"].replace('-1', 0)
    data['ProductRelated_Duration'] = data["ProductRelated_Duration"].replace('-1', 0)
    data['Administrative_Duration'] = data["Administrative_Duration"].replace(np.nan, 0.0)
    data['Informational_Duration'] = data["Informational_Duration"].replace(np.nan, 0.0)
    data['ProductRelated_Duration'] = data["ProductRelated_Duration"].replace(np.nan, 0.0)
    
    data['Month'] = data['Month'].replace('Feb', '2')
    data['Month'] = data['Month'].replace('Mar', '3')
    data['Month'] = data['Month'].replace('May', '5')
    data['Month'] = data['Month'].replace('Oct', '10')
    data['Month'] = data['Month'].replace('June', '6')
    data['Month'] = data['Month'].replace('Jul', '7')
    data['Month'] = data['Month'].replace('Aug', '8')
    data['Month'] = data['Month'].replace('Nov', '11')
    data['Month'] = data['Month'].replace('Sep', '9')
    data['Month'] = data['Month'].replace('Dec', '12')
    
    return data
