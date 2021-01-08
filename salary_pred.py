# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 15:27:15 2020

@author: Karan
"""

'''
USING PYSPARK TO PREDICT WHETHER THE SALARY OF A PERSON LIES ABOVE OR BELOW 
$50,000 ANUALLY

Pyspark parallelises the dataframes and the computations very efficiently
just like Apache Spark does, and is very efficient for handling big data

In this script, I have loaded the data using pyspark, preprocessed it and done 
some very brief and basic analysis of the data in order to gain some insights
into the features given. Further, the data gives almost 85% accuracy with Logistic
Regression, the performance falls with ensemble methods as the dataset appears to 
be linearly separable

Please note that this script is for a static dataset. If the data is being updated
over regular intervals, this script will need to be supplemented by a singer data
pipeline being run at regular intervals. If needed, do let me know and I will try
to implement this feature
'''
#Importing libraries and creating a spark session
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
from datetime import date
from pyspark.sql import Row, SparkSession
from pyspark.sql.functions import *
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.context import SparkContext
from pyspark.sql.types import StructField,StructType, IntegerType, StringType, FloatType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import DenseVector
from pyspark.ml import Pipeline
from helper_1 import *
from multiprocessing import Process
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
sc=SparkContext()
spark=SparkSession(sc)
sc.setLogLevel('FATAL')

#Creating Schema for the data to be loaded
schema= StructType(
    [
     StructField('Age',IntegerType(),nullable=False),
     StructField('workclass',StringType(),nullable=False),
     StructField('fnlwgt',FloatType(),nullable=False),
     StructField('education',StringType(),nullable=False),
     StructField('education-num',FloatType(),nullable=False),
     StructField('marital',StringType(),nullable=False),
     StructField('occupation',StringType(),nullable=False),
     StructField('relationship',StringType(),nullable=False),
     StructField('race',StringType(),nullable=False),
     StructField('sex',StringType(),nullable=False),
     StructField('capital-gain',FloatType(),nullable=False),
     StructField('capital-loss',FloatType(),nullable=False),
     StructField('hours-per-week',FloatType(),nullable=False),
     StructField('native-country',StringType(),nullable=False),
     StructField('salary',StringType(),nullable=False),
     ]
    )

#Reading the data
df=spark.read.schema(schema).csv(r'C:/Users/karan/OneDrive/Desktop/Github/PySpark_Sample/Data.csv',header=True)
df3=df.toPandas() #For analysis purpose

print('Data Schema :',df.printSchema) #To check for correct interpretations of data types

'''
DATA PREPROCESSING
'''

# Replacing '?' by NULL in columns 'workclass', 'occupation', 'native-country'
df = df.withColumn('workclass', when(
    df.workclass==' ?', lit('null')).otherwise(df.workclass))

df = df.withColumn('occupation', when(
    df.occupation==' ?', lit('null')).otherwise(df.occupation))

df = df.withColumn('native-country', when(
    col('native-country')==' ?', lit('null')).otherwise(col('native-country')))

print('Checking for proper replacement with NULL')
df.select(col('workclass')).show(30)

#Dropping rows with null values
print(f'Number of rows before dropping null values: {df.count()}')

df=df.filter(col('workclass')!='null') 
df=df.filter(col('occupation')!='null')
df=df.filter(col('native-country')!='null')

print(f'Number of rows after dropping null values: {df.count()}')


# Using the String Indexer to index the strings in the categorical data 
# present in the used dataset
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index")
            .fit(df) for column in list(set(df.columns)-
             set(['Age','fnlwgt','education-num','capital-gain',
            'capital-loss','hours-per-week'])) ]
pipeline = Pipeline(stages=indexers)
df2 = pipeline.fit(df).transform(df)

#Dropping the old non categorical columns as they have been replaced
#by newly indexed columns
df2=df2.drop('workclass','education','marital','occupation',
             'relationship','race','sex','native-country','salary')
print('Columns after categorical to numerical:', df2.columns)
print('Dataframe after categorical to numerical:', df2.show(3))


# Checking for Correlation to drop insignificant columns 
df3=df2.toPandas()
corr=df3.corr()
print('Correlation Matrix:')
print(corr)


# Removing the columns with very low correlation to the target i.e. salary index
df2=df2.drop('fnlwgt','education_index','race_index','native-country_index')

print('Columns after Dropping low correlation columns:', df2.columns)


#Statistical Summary of selected columns
# numeric_features = [t[0] for t in df2.dtypes if t[1] in ['int','float','double']]
# summary=df2.select(numeric_features).describe().toPandas().transpose()
# print('Summary of selected columns=',summary)


#Visualising the selected columns using Scatter Plots
# numeric_data = df2.select(numeric_features).toPandas()
# axs = scatter_matrix(numeric_data, figsize=(8, 8));
# n = len(numeric_data.columns)
# for i in range(n):
#     v = axs[i, 0]
#     v.yaxis.label.set_rotation(0)
#     v.yaxis.label.set_ha('right')
#     v.set_yticks(())
#     h = axs[n-1, i]
#     h.xaxis.label.set_rotation(90)
#     h.set_xticks(())


#One hot encoding the columns which have been converted from cat to numerical

#Categorical columns that need to be one hot encoded
cat_columns=['sex_index','marital_index','relationship_index',
             'workclass_index','occupation_index']

#new column names after OHE
cat_columns_enc=[ col+'_enc' for col in cat_columns]

ohe=OneHotEncoder(inputCols=cat_columns,outputCols=cat_columns_enc,dropLast=True)
ohe=ohe.fit(df2)
df2=ohe.transform(df2)

#dropping the older columns
for col in cat_columns:
    df2=df2.drop(col)

print('Dataframe and columns after OHE:')
df2.show(2)
print(df2.columns)


# Converting the preprocessed columns into vectors
# Pyspark expects the features to be coalesced together, so this snippet of code
# combines all the features that will be used for prediction of the target
# and stores them in the same dataframe under a new column 'vectors'

columns=df2.columns
columns.remove('salary_index')
assembled=VectorAssembler(inputCols=columns,outputCol='vectors')
X=assembled.transform(df2)
X=X.select(X.vectors,X.salary_index) #disposing of all the extra columns

print('After vectorising the data:')
X.show(3)



'''
MODEL TRAINING
'''
# The vectors generated above are sparse vectors, and need to be converted into 
# Dense Vectors to be accepted by the Logistic Regression function
input_data = X.rdd.map(lambda x: (x["salary_index"], DenseVector(x["vectors"])))
df_train = spark.createDataFrame(input_data, ["label", "features"])			

print('Final dataset to be used for the model:')
df_train.show(5)

#Train test split
train_data, test_data = df_train.randomSplit([.95,.05],seed=1234)

train_1,train_2,train_3=train_data.randomSplit([0.33,0.33,0.34],seed=1234)


'''
1) Training 1 individual model with all data
'''

print('**********')
print('For 1 Model')
print('**********')

#Training the Logistic Regression Model
classifier=LogisticRegression(featuresCol='features',labelCol='label')
classifier=classifier.fit(train_data)


#Making predictions
pred=classifier.transform(test_data)
print('Predictions:')
pred.show(10)

#Model Accuracy
cm = pred.select("label", "prediction")			
cm.show()

acc=cm.filter(cm.label == cm.prediction).count() / cm.count()		
print(f'Accuracy for one model: {acc}%')




'''
2) Training 3 individual models parallelly with 1/3 data each
'''


print('**********')
print('For 3 models trained separately used as an Ensemble')
print('**********')

# test_data2=test_data
# test_data3=test_data

# if __name__=='__main__':
#     p1=Process(target=log_reg_train,args=(train_1,test_data,))
#     p1.start()
#     p2=Process(target=log_reg_train,args=(train_2,test_data2))
#     p2.start()
#     p3=Process(target=log_reg_train,args=(train_3,test_data3))
#     p3.start()
#     p1.join()
#     p2.join()
#     p3.join()

#     print('multiprocessing')
# print(pred1)



#Using 3 models 
#Using the log_reg_train() defined in helper_1.py module to train and fit the model

pred1=log_reg_train(train_1,test_data)

pred2=log_reg_train(train_2,test_data)

pred3=log_reg_train(train_3,test_data)



#Combining the predictions made by the 3 models into 1 dataset
pred1=pred1.withColumnRenamed('prediction','prediction_1')
pred2=pred2.withColumnRenamed('prediction','prediction_2')
pred2=pred2.withColumnRenamed('label','label_2')
pred3=pred3.withColumnRenamed('prediction','prediction_3')
pred3=pred3.withColumnRenamed('label','label_3')

pred2=pred2.select('prediction_2')
pred3=pred3.select('prediction_3')

pred1 = pred1.withColumn("id", monotonically_increasing_id())
pred2 = pred2.withColumn("id_2", monotonically_increasing_id())
pred3 = pred3.withColumn("id_3", monotonically_increasing_id())

pred2=pred2.select('prediction_2','id_2')
pred3=pred3.select('prediction_3','id_3')

pred=pred1.join(pred2,pred1.id==pred2.id_2)
pred=pred.join(pred3,pred.id==pred3.id_3)
pred=pred.drop('id_2')
pred=pred.drop('id_3')
pred=pred.drop('id')


#Taking the average of the 3 predictions to be considered as the final prediction
cols=list(set(pred.columns)-set(['label']))

final_df = pred.withColumn('prediction', (pred['prediction_1']+
                                  pred['prediction_2']+pred['prediction_3'])>1)

final_df=final_df.select('label','prediction')
final_df=final_df.toPandas()
print('Final predictions after taking average of the 3 models:')
final_df.head()


y_true=final_df['label'].values
y_pred=final_df['prediction'].values
y_pred=y_pred.astype(float)

from sklearn.metrics import accuracy_score
#Accuracy of the 3 model ensemble
print('Final Accuracy of the 3 model ensemble:')
print(accuracy_score(y_true,y_pred))



