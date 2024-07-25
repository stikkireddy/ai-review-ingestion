# Databricks notebook source
# MAGIC %pip install dbtunnel[arize-phoenix,asgiproxy]
# MAGIC %pip install arize-phoenix==4.12.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

print("Running tracing mode")
from dbtunnel import dbtunnel
arize = dbtunnel.arize_phoenix()
# make sure to run the next cell to spawn the server
arize.ui_url()

# COMMAND ----------

arize.run()

# COMMAND ----------


