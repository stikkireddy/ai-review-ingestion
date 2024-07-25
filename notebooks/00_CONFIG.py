# Databricks notebook source
# MAGIC %pip install ../
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

CATALOG = "main"
SCHEMA = "sri_winedb"
REVIEWS_TABLE = "reviews"
QUESTIONS_TABLE = "questions"
TARGET_TABLE = "reviews_predictions"
print("Variable: CATALOG = {}".format(CATALOG))
print("Variable: SCHEMA = {}".format(SCHEMA))
print("Variable: QUESTIONS_TABLE = {}".format(QUESTIONS_TABLE))
print("Variable: REVIEWS_TABLE = {}".format(REVIEWS_TABLE))
print("Variable: TARGET_TABLE = {}".format(TARGET_TABLE))

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# COMMAND ----------


