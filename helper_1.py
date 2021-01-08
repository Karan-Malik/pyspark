# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 13:13:57 2020

@author: karan
"""


from pyspark.ml.classification import LogisticRegression


def log_reg_train(train_data,test_data):
    classifier=LogisticRegression(featuresCol='features',labelCol='label')
    classifier=classifier.fit(train_data)
    pred=classifier.transform(test_data)
    cm = pred.select("label", "prediction")			
    return cm



    