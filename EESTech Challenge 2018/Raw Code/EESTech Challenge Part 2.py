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

# II.1
directory = "/dbfs/FileStore/tables/EESTechDataSet"
if not os.path.exists(directory):
    os.makedirs(directory)
os.rename("/dbfs/FileStore/tables/nasa_jpl_confirmed_exoplanets_mar25_2018.csv", "/dbfs/FileStore/tables/EESTechDataSet/Part2Full.csv")

# COMMAND ----------

display(dataSet)

# COMMAND ----------

# I.2
forPlot = dataSet.dropna(subset=["pl_trandep", "pl_rade"]).select("pl_trandep", "pl_rade")
forPlot_pd = forPlot.withColumnRenamed("pl_trandep", "Transit Depth (percentage)").withColumnRenamed("pl_rade", "Planet Radius (Earth radii)").toPandas()
print(forPlot_pd)
scatterPlot = forPlot_pd.plot.scatter(x='Transit Depth (percentage)', y='Planet Radius (Earth radii)', color='Red')
display()

# COMMAND ----------

forPlot = dataSet.dropna(subset=["pl_orbper", "pl_rade"]).select("pl_orbper", "pl_rade")
forPlot_pd = forPlot.withColumnRenamed("pl_orbper", "Orbital Period (days)").withColumnRenamed("pl_rade", "Planet Radius (Earth radii)").toPandas()
print(forPlot_pd)
ax = forPlot_pd.plot.scatter(x='Orbital Period (days)', y='Planet Radius (Earth radii)', color='Blue', label='All Confirmed Planets')

forPlot = dataSet.dropna(subset=["pl_orbper", "pl_rade"]).select("pl_orbper", "pl_rade").filter('pl_kepflag == 1')
forPlot_pd = forPlot.withColumnRenamed("pl_orbper", "Orbital Period (days)").withColumnRenamed("pl_rade", "Planet Radius (Earth radii)").toPandas()

forPlot_pd.plot.scatter(x='Orbital Period (days)', y='Planet Radius (Earth radii)', color='Red', label='Kepler Field Planets', ax = ax)

display()
