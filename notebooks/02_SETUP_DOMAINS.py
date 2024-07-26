# Databricks notebook source
# MAGIC %run ./00_CONFIG

# COMMAND ----------

from auto_topic.domains import DomainConfigTable, Domain

# COMMAND ----------

dct = DomainConfigTable(catalog=CATALOG, schema=SCHEMA, table=QUESTIONS_TABLE)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Define Domains
# MAGIC
# MAGIC **Define different domains and questions you want answered. You can have a lot of these and the llm will do a two pass, first identify the topic and then map to fill out all the information that needs to be filled out in the domain**
# MAGIC
# MAGIC Each domain will require:
# MAGIC 1. topic -> 1 word all lower case
# MAGIC 2. when ->  topic condition: succinct sentence aligning topic name with specific entities aligning with this topic, e.g. in retail topic name defect may align with sneaker but in manufacturing topic name 
# MAGIC 3. details -> single sentence expression what you want extracted (just additional info)
# MAGIC
# MAGIC You can optionally extract additional info in each topic using the `.with_additional_info` with the following two fields:
# MAGIC 1. item_name -> which is only alpha numeric json key for labeling the concept/question
# MAGIC 2. item_description -> the details in how to extract the item/concept/detail
# MAGIC
# MAGIC For example when someone is describing a review and if they describe the item was used as a gift you may want to extract that detail if you think this may be something that you can use to suggest specific items as great gifts during a specific season.
# MAGIC

# COMMAND ----------

dct.with_topic(
  topic=Domain(
    topic="product",
    when="generally when the user is referering to the product they have purchased",
    details="<summary of the review regarding the product>"
  ).with_additional_info(
    item_name="can_be_used_as_gift",
    item_description="the product was used as a gift or will be used as a gift <yes/no/unsure>"
  ).with_additional_info(
    item_name="as_advertized",
    item_description="the product behaved as advertized whether it be description or other reviews <yes/no/unsure>"
  )
)

dct.with_topic(
  topic=Domain(
    topic="defects",
    when="generally when the user is referering to the product they have purchased",
    details="<summary of the review regarding the product>",
  ).with_additional_info(
    item_name="defect_level",
    item_description="the level of defectiveness to the product use your best judgement <small/medium/large>"
  ).with_additional_info(
    item_name="misinformation",
    item_description="the product does not work as described <yes/no>"
  )
)


dct.with_topic(
  topic=Domain(
    topic="delivery",
    when="generally when the user is referering to the product they have purchased",
    details="<summary of the review regarding the product>",
  ).with_additional_info(
    item_name="defect_level",
    item_description="the level of defectiveness to the product use your best judgement <small/medium/large>"
  ).with_additional_info(
    item_name="misinformation",
    item_description="the product does not work as described <yes/no>"
  )
)

dct.with_topic(
  topic=Domain(
    topic="movies",
    when="generally when the user is referering to the movies they have purchased",
    details="<summary of the review regarding the product>",
  ).with_additional_info(
    item_name="enjoyment",
    item_description="the level of enjoyment to the moview use your best judgement <1-5>"
  ).with_additional_info(
    item_name="genre",
    item_description="the genre of the movie"
  ).with_additional_info(
    item_name="audio_issues",
    item_description="were there audio quality issues <yes/no>"
  ).with_additional_info(
    item_name="video_quality_issues",
    item_description="were there video quality issues <yes/no>"
  )
)


# COMMAND ----------

dct.setup(spark)

# COMMAND ----------

spark.table(f"{dct.catalog}.{dct.schema}.{dct.table}").display()

# COMMAND ----------


