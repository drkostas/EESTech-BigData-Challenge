# Databricks notebook source
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.sql.functions import *
from pyspark.ml.feature import *
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import *
from pyspark.ml.stat import *
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

# COMMAND ----------

# I.1.1
path = "/FileStore/tables/EESTechDataSet/Part1Full.csv"
dataSet = spark.read.load(path,
                     format="csv", sep=",", inferSchema="true", header="true")
dataSet = dataSet.select('*').withColumn("full_name", trim(dataSet.full_name))

# COMMAND ----------

# I.1.2
dataSet.printSchema()

# COMMAND ----------

# I.1.3
print("Number of rows: ", dataSet.count())
print("Number of Columns: ", len(dataSet.columns))
# display(dataSet.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in dataSet.columns]))

# COMMAND ----------

# I.2.1
display(dataSet.limit(100))

# COMMAND ----------

# I.2.2
display(dataSet.describe())

# COMMAND ----------

# I.3.1
dataSet_diameterDesc = dataSet.dropna(subset="diameter").sort(desc("diameter")).filter("diameter>100").sort("diameter", ascending=False).limit(50)

# COMMAND ----------

# I.3.2
dataSet_diameterDesc_present = dataSet_diameterDesc.select("full_name", "diameter", "q", "first_obs", "producer", "class")\
                              .withColumnRenamed("full_name", "Name").withColumnRenamed("diameter", "Diameter").withColumnRenamed("q", "Mean Distance From Sun (in AU)")\
                              .withColumnRenamed("first_obs", "Date Discovered").withColumnRenamed("producer", "Discoverer").withColumnRenamed("class", "Class")
display(dataSet_diameterDesc_present)

# COMMAND ----------

# I.4.3
dataSet_axisOrbit = dataSet.dropna(subset=["a", "per_y"]).select("a", "per_y")
dataSet_axisOrbit_pd = dataSet_axisOrbit.withColumnRenamed("a", "Semi-Major Axis").withColumnRenamed("per_y", "Orbit Period").toPandas()
scatterPlot = dataSet_axisOrbit_pd.plot.scatter(x='Semi-Major Axis', y='Orbit Period')
display()

# COMMAND ----------

# I.4.4
vecAssembler = VectorAssembler(inputCols=["a"], outputCol="features")
assembled_dataSet = vecAssembler.transform(dataSet_axisOrbit)
assembled_dataSet = assembled_dataSet.select("per_y", "features").withColumnRenamed("per_y", "label")
train = assembled_dataSet
lr = LinearRegression(maxIter=10, regParam=0.7, elasticNetParam=0.8)

# Fit the model
lrModel = lr.fit(train)

# Print the coefficients and intercept for linear regression
print("Coefficients: %s" % str(lrModel.coefficients))
print("Intercept: %s" % str(lrModel.intercept))

# Summarize the model over the training set and print out some metrics
trainingSummary = lrModel.summary
print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show()
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)
trainingSummary.predictions.show()

# COMMAND ----------

ax = dataSet_axisOrbit_pd.plot.scatter(x='Semi-Major Axis', y='Orbit Period', color='DarkBlue', label='Actual')
dataSet_axisOrbit_pd.loc[:,'Orbit Period'] *= 418.824218173
dataSet_axisOrbit_pd.loc[:,'Orbit Period'] -= -1539.7951993985082
dataSet_axisOrbit_pd.plot.scatter(x='Semi-Major Axis', y='Orbit Period', color='Green', label='Prediction', ax=ax);

display()

# COMMAND ----------

# I.5
display(dataSet.filter("lower(full_name) like '%tesla%' OR lower(full_name) like '%spacex%' OR lower(full_name) like '%roadster%' OR lower(name) like '%tesla%' OR lower(name) like '%spacex%' or lower(name) like '%roadster%'"))

