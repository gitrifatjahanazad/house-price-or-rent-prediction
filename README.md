
## 1 Introduction
### 1.1 Who's the target audience
This example is intended for audience who are comfortable with developing models using Jupyter notebook. A basic understanding of Azure Machine Learning (Azure ML) experiment is helpful. The purpose here is to give a complete walkthrough on using Jupyter notebook within Azure Machine Studio. Using linear regression as an example, we'll have the opportunity to see how Azure ML's Jupyter notebook can be used to fit a model, set up a web service on Azure ML and consume the service.

If you are new to Jupyter notebook, you can learn more about it from [The IPython Notebook][ipython link] and [Jupyter/IPython Notebook Quick Start Guide][jupyter link]. 

If you are new to Azure ML and want to learn more about it, the [Data Scientists' Guide][guide link] can help you get started.

[ipython link]: http://ipython.org/notebook.html
[jupyter link]: http://jupyter-notebook-beginner-guide.readthedocs.org/en/latest/index.html
[guide link]: https://gallery.cortanaanalytics.com/Experiment/Tutorial-for-Data-Scientists-3

### 1.2 Why Azure ML notebook
The major advantage of using Azure ML notebook is that you won't need to install Python on your local computer. This makes it possible for anyone with internet access to write Python programs using a web browser. Azure ML's notebook is based on the [Anaconda][Anaconda link] distribution, which has many packages installed. 

[Anaconda link]: https://www.continuum.io/

## 2 Data
In this example we'll be using the Boston housing dataset. There are 506 rows in the dataset. The target variable is median home price. There are 13 predictor variables including average number of rooms per dwelling, crime rate by town, etc. More information about this dataset can be found at [UCI][uci link].
[uci link]: https://archive.ics.uci.edu/ml/datasets/Housing


```python
from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
%matplotlib inline


```


```python
boston = load_boston()
X = boston.data
y = boston.target
X, X_test, y, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
feature_names = boston.feature_names
print(X.shape)
print(y.shape)
print(feature_names)
print('boston data set object exploring:')
pprint(dir(boston))
```

    (303, 13)
    (303,)
    ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
     'B' 'LSTAT']
    boston data set object exploring:
    ['DESCR',
     '__class__',
     '__cmp__',
     '__contains__',
     '__delattr__',
     '__delitem__',
     '__dict__',
     '__doc__',
     '__eq__',
     '__format__',
     '__ge__',
     '__getattribute__',
     '__getitem__',
     '__gt__',
     '__hash__',
     '__init__',
     '__iter__',
     '__le__',
     '__len__',
     '__lt__',
     '__module__',
     '__ne__',
     '__new__',
     '__reduce__',
     '__reduce_ex__',
     '__repr__',
     '__setattr__',
     '__setitem__',
     '__sizeof__',
     '__str__',
     '__subclasshook__',
     '__weakref__',
     'clear',
     'copy',
     'data',
     'feature_names',
     'fromkeys',
     'get',
     'has_key',
     'items',
     'iteritems',
     'iterkeys',
     'itervalues',
     'keys',
     'pop',
     'popitem',
     'setdefault',
     'target',
     'update',
     'values',
     'viewitems',
     'viewkeys',
     'viewvalues']



```python
print(boston.DESCR)
```

    Boston House Prices dataset
    
    Notes
    ------
    Data Set Characteristics:  
    
        :Number of Instances: 506 
    
        :Number of Attributes: 13 numeric/categorical predictive
        
        :Median Value (attribute 14) is usually the target
    
        :Attribute Information (in order):
            - CRIM     per capita crime rate by town
            - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
            - INDUS    proportion of non-retail business acres per town
            - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
            - NOX      nitric oxides concentration (parts per 10 million)
            - RM       average number of rooms per dwelling
            - AGE      proportion of owner-occupied units built prior to 1940
            - DIS      weighted distances to five Boston employment centres
            - RAD      index of accessibility to radial highways
            - TAX      full-value property-tax rate per $10,000
            - PTRATIO  pupil-teacher ratio by town
            - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
            - LSTAT    % lower status of the population
            - MEDV     Median value of owner-occupied homes in $1000's
    
        :Missing Attribute Values: None
    
        :Creator: Harrison, D. and Rubinfeld, D.L.
    
    This is a copy of UCI ML housing dataset.
    http://archive.ics.uci.edu/ml/datasets/Housing
    
    
    This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.
    
    The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
    prices and the demand for clean air', J. Environ. Economics & Management,
    vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
    ...', Wiley, 1980.   N.B. Various transformations are used in the table on
    pages 244-261 of the latter.
    
    The Boston house-price data has been used in many machine learning papers that address regression
    problems.   
         
    **References**
    
       - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.
       - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.
       - many more! (see http://archive.ics.uci.edu/ml/datasets/Housing)
    


### Loading sk learn data into pandas


```python
import pandas as pd
from sklearn.linear_model import LinearRegression
pdHousingData = pd.DataFrame(data= np.c_[boston['data'], boston['target']],
                     columns= np.append(boston['feature_names'], ['target']))
pdHousingData.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.593761</td>
      <td>11.363636</td>
      <td>11.136779</td>
      <td>0.069170</td>
      <td>0.554695</td>
      <td>6.284634</td>
      <td>68.574901</td>
      <td>3.795043</td>
      <td>9.549407</td>
      <td>408.237154</td>
      <td>18.455534</td>
      <td>356.674032</td>
      <td>12.653063</td>
      <td>22.532806</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.596783</td>
      <td>23.322453</td>
      <td>6.860353</td>
      <td>0.253994</td>
      <td>0.115878</td>
      <td>0.702617</td>
      <td>28.148861</td>
      <td>2.105710</td>
      <td>8.707259</td>
      <td>168.537116</td>
      <td>2.164946</td>
      <td>91.294864</td>
      <td>7.141062</td>
      <td>9.197104</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.006320</td>
      <td>0.000000</td>
      <td>0.460000</td>
      <td>0.000000</td>
      <td>0.385000</td>
      <td>3.561000</td>
      <td>2.900000</td>
      <td>1.129600</td>
      <td>1.000000</td>
      <td>187.000000</td>
      <td>12.600000</td>
      <td>0.320000</td>
      <td>1.730000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.082045</td>
      <td>0.000000</td>
      <td>5.190000</td>
      <td>0.000000</td>
      <td>0.449000</td>
      <td>5.885500</td>
      <td>45.025000</td>
      <td>2.100175</td>
      <td>4.000000</td>
      <td>279.000000</td>
      <td>17.400000</td>
      <td>375.377500</td>
      <td>6.950000</td>
      <td>17.025000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.256510</td>
      <td>0.000000</td>
      <td>9.690000</td>
      <td>0.000000</td>
      <td>0.538000</td>
      <td>6.208500</td>
      <td>77.500000</td>
      <td>3.207450</td>
      <td>5.000000</td>
      <td>330.000000</td>
      <td>19.050000</td>
      <td>391.440000</td>
      <td>11.360000</td>
      <td>21.200000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.647423</td>
      <td>12.500000</td>
      <td>18.100000</td>
      <td>0.000000</td>
      <td>0.624000</td>
      <td>6.623500</td>
      <td>94.075000</td>
      <td>5.188425</td>
      <td>24.000000</td>
      <td>666.000000</td>
      <td>20.200000</td>
      <td>396.225000</td>
      <td>16.955000</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>88.976200</td>
      <td>100.000000</td>
      <td>27.740000</td>
      <td>1.000000</td>
      <td>0.871000</td>
      <td>8.780000</td>
      <td>100.000000</td>
      <td>12.126500</td>
      <td>24.000000</td>
      <td>711.000000</td>
      <td>22.000000</td>
      <td>396.900000</td>
      <td>37.970000</td>
      <td>50.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# sns.pairplot(pdHousingData)
```


```python
sns.distplot(pdHousingData['target'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fafa9ec66d0>




![png](output_8_1.png)



```python
sns.heatmap(pdHousingData.corr(), annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fafa866bed0>




![png](output_9_1.png)


## 3 Linear regression with sklearn
Now let's assign the data to a new variable "mydata" and use it to develop a linear model. In practice, you may want to spend more time on things like feature engineering and variable selection. In this example, however, we will just fit a model using all variables. 


```python
# initiate the linear model and fit with data
lm = LinearRegression()
lm.fit(X, y)

# print the R-squared
print("The R-squared value is: {0:0.4f} \n".format(lm.score(X, y)))

# save intercept and coefficients
param_df = pd.DataFrame({"Features": ['intercept'] + list(feature_names), "Coef": [lm.intercept_] + list(lm.coef_)})
cols = param_df.columns.tolist()
cols = cols[-1:]+cols[:-1]
param_df = param_df[cols]
print(param_df)
```

    The R-squared value is: 0.7502 
    
         Features       Coef
    0   intercept  41.281497
    1        CRIM  -0.077558
    2          ZN   0.042031
    3       INDUS   0.091153
    4        CHAS   4.133049
    5         NOX -19.976557
    6          RM   2.890190
    7         AGE   0.016153
    8         DIS  -1.264747
    9         RAD   0.260171
    10        TAX  -0.011125
    11    PTRATIO  -0.880556
    12          B   0.007024
    13      LSTAT  -0.643483



```python
predictions = lm.predict(X_test)
plt.axis(xmin=0,xmax=40,ymin=0,ymax=40)
plt.scatter(y_test,predictions)
plt.plot(range(40),'g')
```




    [<matplotlib.lines.Line2D at 0x7fafa4e6fd10>]




![png](output_12_1.png)


Next we'll use the model to make predictions. Typically, predictions are done on a validation dataset. Here, however, the training dataset is used for illustraton purposes.


```python
newX = X
newY = y

# predict and create a dataframe
predicted = lm.predict(newX)
predictedDf = pd.DataFrame({"predicted": predicted})
# dataframe for X
newXDf = pd.DataFrame(newX)
newXDf.columns = feature_names
# dataframe for y
newYDf = pd.DataFrame(newY)
newYDf.columns = ['MEDV']
# join X, y and prediction
mydata_with_pd = newXDf.join(newYDf).join(predictedDf)
mydata_with_pd.head()

```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>MEDV</th>
      <th>predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.06888</td>
      <td>0.0</td>
      <td>2.46</td>
      <td>0.0</td>
      <td>0.488</td>
      <td>6.144</td>
      <td>62.2</td>
      <td>2.5979</td>
      <td>3.0</td>
      <td>193.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.45</td>
      <td>36.2</td>
      <td>26.894764</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.14103</td>
      <td>0.0</td>
      <td>13.92</td>
      <td>0.0</td>
      <td>0.437</td>
      <td>5.790</td>
      <td>58.0</td>
      <td>6.3200</td>
      <td>4.0</td>
      <td>289.0</td>
      <td>16.0</td>
      <td>396.90</td>
      <td>15.84</td>
      <td>20.3</td>
      <td>19.819394</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5.58107</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.713</td>
      <td>6.436</td>
      <td>87.9</td>
      <td>2.3158</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>100.19</td>
      <td>16.22</td>
      <td>14.3</td>
      <td>16.661445</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.37578</td>
      <td>0.0</td>
      <td>10.59</td>
      <td>1.0</td>
      <td>0.489</td>
      <td>5.404</td>
      <td>88.6</td>
      <td>3.6650</td>
      <td>4.0</td>
      <td>277.0</td>
      <td>18.6</td>
      <td>395.24</td>
      <td>23.98</td>
      <td>19.3</td>
      <td>17.922946</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.14932</td>
      <td>25.0</td>
      <td>5.13</td>
      <td>0.0</td>
      <td>0.453</td>
      <td>5.741</td>
      <td>66.2</td>
      <td>7.2254</td>
      <td>8.0</td>
      <td>284.0</td>
      <td>19.7</td>
      <td>395.11</td>
      <td>13.15</td>
      <td>18.7</td>
      <td>18.151051</td>
    </tr>
  </tbody>
</table>
</div>



With the predicted values, we can calculate metrics such as mean absolute error, root mean squared error, relative absolute error, and relative squared error. 


```python
obs = mydata_with_pd.MEDV
pred = mydata_with_pd.predicted

mae = np.mean(abs(pred-obs))
rmse = np.sqrt(np.mean((pred-obs)**2))
rae = np.mean(abs(pred-obs))/np.mean(abs(obs-np.mean(obs)))
rse = np.mean((pred-obs)**2)/np.mean((obs-np.mean(obs))**2)

print("Mean Absolute Error: {0:0.6f}".format(mae))
print("Root Mean Squared Error: {0:0.6f}".format(rmse))
print("Relative Absolute Error: {0:0.6f}".format(rae))
print("Relative Squared Error: {0:0.6f}".format(rse))
```

    Mean Absolute Error: 2.938568
    Root Mean Squared Error: 4.263022
    Relative Absolute Error: 0.484363
    Relative Squared Error: 0.249850


## 4 Web service

### 4.1 Set up a web service
After developing the model, we can deploy it as a service so others can use it. The "azureml" package's services subpackage can be used for this purpose. The following lines of code are setting up a web service named "demoservice".


```python
# extract workspace info
from azureml import Workspace
ws = Workspace()
workdspace_id = ws.workspace_id
authorization_token = ws.authorization_token

# set up web service
from azureml import services
@services.publish(workdspace_id, authorization_token)
@services.types(crim=float, zn=float, indus=float, chas=float, nox=float, rm=float, age=float, 
                dis=float, rad=float, tax=float, ptratio=float, black=float, lstat=float)
@services.returns(float)
def demoservice(crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, black, lstat):
    # predict the label
    feature_vector = [crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, black, lstat]
    return lm.predict(feature_vector)
```

### 4.2 Consume a web service
After running the above code to set up a web service, we can consume it like the following example. This way of consuming the web service works only during the current session.


```python
demoservice(0.00632, 18, 2.31, 0, 0.538, 6.575, 65.2, 4.09, 1, 296, 15.3, 396.9, 4.98)
```




    29.462195987089871



To consume the web service out of the current session, we can use Python scripts by specifying the service url and the api\_key. The following lines of code return the url, the api\_key, the help\_url, and the service\_id. 


```python
service_url = demoservice.service.url 
api_key = demoservice.service.api_key
help_url = demoservice.service.help_url
service_id = demoservice.service.service_id
```

The purpose of the following code is to wait a few seconds so that the web service is successfully desployed and ready for consumption.


```python
# wait 
import time
time.sleep(10)
```

The code below can be used to consume the web service. For illustration purpose the data for two records are provided.


```python
import urllib2
# If you are using Python 3+, import urllib instead of urllib2

import json 

data =  {

        "Inputs": {

                "input1":
                {
                    "ColumnNames": ["crim", "zn", "lstat", "age", "tax", "rad", "black", 
                                    "chas", "nox", "rm", "indus", "ptratio", "dis"],
                    "Values": [ [ "0.00632", "18", "4.98", "65.2", "296", "1", "396.9", 
                                 "0", "0.538", "6.575", "2.31", "15.3", "4.09" ], 
                               ["0.02731", "0", "9.14", "78.9", "242", "2", "396.9", 
                                "0", "0.469", "6.421", "7.07", "17.8", "4.9671"],
                            ]
                },        },
            "GlobalParameters": {
}
    }

body = str.encode(json.dumps(data))

url = service_url
api_key = api_key # Replace this with the API key for the web service

headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

req = urllib2.Request(url, body, headers) 

try:
    response = urllib2.urlopen(req)

    # If you are using Python 3+, replace urllib2 with urllib.request in the above code:
    # req = urllib.request.Request(url, body, headers) 
    # response = urllib.request.urlopen(req)

    result = response.read()
    print(result) 
except urllib2.HTTPError, error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, 
    # which are useful for debugging the failure
    print(error.info())

    print(json.loads(error.read()))                 
```

    {"Results":{"output1":{"type":"table","value":{"Values":[["29.4621959870899"],["25.1658357600726"]]}},"output2":{"type":"table","value":{"Values":[["data:text/plain,Execution OK\r\n",null]]}}}}


The above service consumption code is based on those at the help\_url, which contains, among others, sample code written in C#, Python, and R for consuming the web service. To check out the code for yourself, copy the help\_url value to a new browser to open the web page. Scroll down the newly opened page till you see the section "Sample Code" as in [Figure 1][pic 1]. Click on the Python tab and you'll see the sample code. Two changes were made in the above code:
    1. fill in the url and api\_key with the values returned for the current web service, and 
    2. enter the values for the first two records. Notice that the order of the columns here may be different from 
    that in your definition of the web service. 

[![Python Script][pic 1]][pic 1] Figure 1

[pic 1]: https://az712634.vo.msecnd.net/notebooks/d132b658-c5bd-11e5-97a3-0ba617781c98.PNG

### 4.3 Re-publish a web service
By default, when you run the code in section 4.1 multiple times different services will be generated. They will have different url, api\_key, help\_url, and service\_id. If you just want to re-publish the same service, you need to specify the service\_id as shown below. This way, you can use the same code for consumption without having to change the url and api\_key.


```python
from azureml import services
@services.publish(workdspace_id, authorization_token)
@services.service_id(service_id)
@services.types(crim=float, zn=float, indus=float, chas=float, nox=float, rm=float, age=float, 
                dis=float, rad=float, tax=float, ptratio=float, black=float, lstat=float)
@services.returns(float)
def demoservice(crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, black, lstat):
    # predict the label
    feature_vector = [crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, black, lstat]
    return lm.predict(feature_vector)
```

## 5 Conclusion
Through this example, you've learned how to fit a model, deploy the model on Azure, and consume the service, all using Azure ML's Jupyter notebook. 

In addition to carrying out the entire model development process with Azure ML notebook, you can also use it as a supplement to Azure ML's experiments. Using Azure ML experiments you can develop a wide variety of models and easily compare their performance. However, certain tasks may not be accomplished there. Examples include certain feature selection techniques, advanced visualization options, a wide variety of GBM models, and time series analysis. For these tasks, Python offers a good alternative and Azure ML's Jupyter notebook allows you to write and run Python programs on the cloud.

As another example of using Jupyter noteebook, you can test out Python code that will be used in the "Execute Python Script" module in an Azure ML Studio experiment.

---  
## References
01. azure machine learning studio python note book house price prediction demo

