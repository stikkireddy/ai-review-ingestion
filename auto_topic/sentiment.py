import copy
import json
from typing import Optional

import dspy
import pandas as pd
from openai import BaseModel
from pydantic import Field, create_model


def enable_arize_tracing(model_id="domainextract"):
    from arize_otel import register_otel, Endpoints

    # Setup OTEL via our convenience function
    register_otel(
        endpoints=Endpoints.PHOENIX_LOCAL,
        model_id=model_id,  # name this to whatever you would like
    )

    # Import the automatic instrumentor from OpenInference
    from openinference.instrumentation.dspy import DSPyInstrumentor
    from openinference.instrumentation.openai import OpenAIInstrumentor

    # Start the instrumentor for DSPy
    DSPyInstrumentor().instrument()
    OpenAIInstrumentor().instrument()


def get_valid_responses_for_categories(df: pd.DataFrame) -> dict:
    rows = df.to_dict(orient="records")
    return [row['topic'].lower() + "details" for row in rows]


def get_when_to_use_category(df: pd.DataFrame):
    rows = df.to_dict(orient="records")
    return {row['topic'].lower() + "details": row['when'].format(topic=row['topic'].lower() + "details") for row in
            rows}


class DefaultCatchAllDetails(BaseModel):
    details: str = Field(description="details of the feedback")
    sentiment: str = Field(description="sentiment of the feedback <positive/negative/neutral/mixed>")
    keywords: list[str] = Field(description='keywords in the feedback ["keyword1", "keyword2", "keyword3", ...]')
    ai_rating: str = Field(description="rating the feedback 1-5 1 being the worst 5 being the best")
    rationale: str = Field(description="rationale for the rating and sentiment")


def make_default_identify_categories(valid_categories, when_to_use_category_map):
    class IdentifyCategories(dspy.Signature):
        """You are a customer feedback analyst. Your oversee customer feedback across all channels, such as website, social media, in store reviews, customer reviews via email and your goal is to extract meaning from the customer reviews. Make sure when asked for json you start with { and end with }"""

        feedback = dspy.InputField(desc="a review from a customer. this can be a positive or negative review")
        categories = dspy.OutputField(desc="the categories of the feedback. one of: " + ", ".join(
            valid_categories) + f" and here is when to use each category {str(when_to_use_category_map)}. Respond with catchalldetails if not related to any of the categories. You can pick multiple categories and respond with a comma separated list.")
        rationale = dspy.OutputField(desc="rationale for picking the categories explain step by step")

    return IdentifyCategories


class DefaultBaseCategorize(dspy.Signature):
    """You are a customer feedback analyst. Your oversee customer feedback across all channels, such as website, social media, in store reviews, customer reviews via email and your goal is to extract meaning from the customer reviews. Make sure when asked for json you start with { and end with }"""

    feedback = dspy.InputField(desc="a review from a customer. this can be a positive or negative review")
    rating = dspy.InputField(
        desc="use this for the connotation of the feedback. the scale ranges from 1 - 5. 1 being horrible 5 being great")


def get_analyzer(df: pd.DataFrame,
                 language_model,
                 identify_catagories_signature=None,
                 base_categorize_signature=None,
                 catch_all_details_signature=None):
    dspy.settings.configure(lm=language_model)
    valid_responses = get_valid_responses_for_categories(df)
    print(valid_responses)
    when = get_when_to_use_category(df)
    print(when)

    IdentifyCategories = identify_catagories_signature or make_default_identify_categories(valid_responses, when)

    CatchAllDetails = catch_all_details_signature or DefaultCatchAllDetails

    list_validators = {
        "keywords": lambda x: json.loads(x)
    }

    def get_type(col_name):
        if col_name in list_validators:
            return list[str]
        return str

    def data_list_to_category_models(df: pd.DataFrame, cats_selected: list[str]):
        data_list = df.to_dict(orient="records")
        models = []
        for item in data_list:
            class_name = item["topic"].capitalize() + "Details"
            if class_name.lower() not in cats_selected:
                continue
            args = copy.deepcopy(json.loads(item["raw"]))
            args.pop("topic")
            args.pop("when")
            kwargs = {k: (Optional[get_type(k)], Field(description=v, default=None)) for k, v in args.items()}
            m = create_model(
                class_name,
                **kwargs,
                __base__=BaseModel
            )
            models.append(m)
        kwargs = {model.__name__.lower(): (model, Field(description=when[model.__name__.lower()]))
                  for model in models}
        if "catchalldetails" in cats_selected:
            kwargs["catchalldetails"] = (CatchAllDetails, Field(
                description="fill the parts of the feedback that are not relevant to the other categories."))
        return create_model("BreakDown",
                            **kwargs,
                            __base__=BaseModel)

    def identified_categories_to_categoriy_signature(prediction):
        cats = [cat.strip() for cat in prediction.categories.split(",")]
        cats = [cat for cat in cats if cat in valid_responses + ["catchalldetails"]]
        if len(cats) == 0:
            return ["catchalldetails"]
        return cats

    Categorize = base_categorize_signature or DefaultBaseCategorize

    def create_dynamic_typed_predictor(df, valid_categories):
        Breakdown = data_list_to_category_models(df, valid_categories)

        DynamicCategorize = create_model(
            'DynamicCategorize',
            breakdown=(Breakdown, dspy.OutputField(desc="fill in all the fields appropriately", prefix="breakdown")),
            __base__=Categorize
        )

        return dspy.TypedPredictor(DynamicCategorize)

    class Error:

        def __init__(self, message: str):
            self.message = message

        def to_dict(self):
            return {"error": self.message}

    class FeedbackAnalysis(dspy.Module):

        def __init__(self, valid_categories_df: pd.DataFrame):
            self.valid_categories_df = valid_categories_df
            self.identify = dspy.ChainOfThought(IdentifyCategories)

        def forward(self, feedback: str, rating: str):
            predict_categories = self.identify(feedback=feedback)
            categories = predict_categories.categories
            print("output categories", categories)
            valid_categories = identified_categories_to_categoriy_signature(predict_categories)
            print("valid categories", valid_categories)
            fill_categories = create_dynamic_typed_predictor(self.valid_categories_df, valid_categories)
            try:
                prediction = fill_categories(feedback=feedback, rating=rating)
                return dspy.Prediction(
                    breakdown=prediction.breakdown,
                    category_selection_rationale=predict_categories.rationale,
                    category_selection=predict_categories.categories
                )
            except Exception as e:
                return dspy.Prediction(
                    breakdown=Error(str(e)),
                    category_selection_rationale=predict_categories.rationale,
                    category_selection=predict_categories.categories
                )

    return FeedbackAnalysis(df)


def _analysis_domain_views_generator(*,
                                     catalog: str,
                                     schema: str,
                                     analysis_table: str,
                                     primary_key_col_name: str,
                                     domain_config_table: DomainConfigTable,
                                     analysis_column_name="analysis",
                                     top_level_key="category_breakdown",
                                     view_prefix="analysis_"):
    if view_prefix.endswith("_") is False:
        view_prefix = view_prefix + "_"
    if len(analysis_table.split(".")) == 1:
        analysis_table = f"{catalog}.{schema}.{analysis_table}"
    for topic in domain_config_table.topics:
        cols = topic.to_kwargs()
        cols.pop("topic")
        cols.pop("when")
        cols.pop("raw")
        select_cols = [primary_key_col_name]
        def_and_comments = [f"{primary_key_col_name} COMMENT 'the primary key of the reviews'"]
        for col_name, description in cols.items():
            if description is not None:
                escaped_description = description.replace("'", "\\'")
                def_and_comments.append(f"{col_name} COMMENT '{escaped_description}'")
            else:
                def_and_comments.append(f"{col_name}")
        for key in cols.keys():
            if key == "keywords":
                select_cols.append(
                    f"from_json({analysis_column_name}:{top_level_key}:{topic.topic}details:{key}, 'array<string>') as {key}")
            else:
                select_cols.append(f"{analysis_column_name}:{top_level_key}:{topic.topic}details:{key} as {key}")
        columns_stmt = ", ".join(def_and_comments)
        select_stmt = "SELECT " + ", ".join(
            select_cols) + " FROM " + analysis_table + f" WHERE {analysis_column_name}:{top_level_key}:{topic.topic}details is not null"
        yield f"\nCREATE OR REPLACE VIEW {catalog}.{schema}.{view_prefix}{topic.topic} ({columns_stmt}) AS {select_stmt};\n"


def _catchall_view_generator(*, catalog: str,
                             schema: str,
                             analysis_table: str,
                             primary_key_col_name: str,
                             analysis_column_name="analysis",
                             top_level_key="category_breakdown",
                             catch_all_details_model=None,
                             view_prefix="analysis_"):
    col_desc_mapping = {
        primary_key_col_name: "the primary key of the reviews",
    }
    col_def_mapping = {
        primary_key_col_name: primary_key_col_name,
    }
    for k, v in (catch_all_details_model or DefaultCatchAllDetails).__annotations__.items():
        description = DefaultCatchAllDetails.__fields__[k].description
        col_desc_mapping[k] = description
        if k == "keywords":
            col_def_mapping[
                k] = f"from_json({analysis_column_name}:{top_level_key}:catchalldetails:{k}, 'array<string>') as {k}"
        else:
            col_def_mapping[k] = f"{analysis_column_name}:{top_level_key}:catchalldetails:{k} as {k}"

    columns_stmt = ", ".join([f"{col} COMMENT '{desc}'" for col, desc in col_desc_mapping.items()])
    select_stmt = ", ".join([col_def_mapping[col] for col in col_desc_mapping.keys()])
    yield f"""
    CREATE OR REPLACE VIEW {catalog}.{schema}.{view_prefix}catchall 
    ({columns_stmt}) AS
    SELECT {select_stmt}
    FROM {catalog}.{schema}.{analysis_table}
    WHERE {analysis_column_name}:{top_level_key}:catchalldetails is not null;
    """


def _error_view_generator(*, catalog: str,
                          schema: str,
                          analysis_table: str,
                          primary_key_col_name: str,
                          analysis_column_name="analysis",
                          top_level_key="category_breakdown",
                          view_prefix="analysis_"):
    yield f"""
    CREATE OR REPLACE VIEW {catalog}.{schema}.{view_prefix}errors AS
    SELECT {primary_key_col_name}, 
        {analysis_column_name}:{top_level_key}:error as error_details
    FROM {analysis_table}
    WHERE {analysis_column_name}:{top_level_key}:error is not null;
    """


def analysis_view_generator(
        *,
        catalog: str,
        schema: str,
        analysis_table: str,
        primary_key_col_name: str,
        domain_config_table: "DomainConfigTable",
        analysis_column_name="analysis",
        top_level_key="category_breakdown",
        view_prefix="analysis_",
        catch_all_view_generator=None,
        error_view_generator=None,
        catch_all_details_model=None
):
    yield from _analysis_domain_views_generator(
        catalog=catalog,
        schema=schema,
        analysis_table=analysis_table,
        primary_key_col_name=primary_key_col_name,
        domain_config_table=domain_config_table,
        analysis_column_name=analysis_column_name,
        top_level_key=top_level_key,
        view_prefix=view_prefix
    )

    yield from (catch_all_view_generator or _catchall_view_generator)(
        catalog=catalog,
        schema=schema,
        analysis_table=analysis_table,
        primary_key_col_name=primary_key_col_name,
        analysis_column_name=analysis_column_name,
        top_level_key=top_level_key,
        view_prefix=view_prefix,
        catch_all_details_model=catch_all_details_model,
    )

    yield from (error_view_generator or _error_view_generator)(
        catalog=catalog,
        schema=schema,
        analysis_table=analysis_table,
        primary_key_col_name=primary_key_col_name,
        analysis_column_name=analysis_column_name,
        top_level_key=top_level_key,
        view_prefix=view_prefix
    )


def build_analysis_views(
        *,
        spark: "SparkSession",
        catalog: str,
        schema: str,
        analysis_table: str,
        primary_key_col_name: str,
        domain_config_table: "DomainConfigTable",
        analysis_column_name="analysis",
        top_level_key="category_breakdown",
        view_prefix="analysis_",
        catch_all_view_generator=None,
        error_view_generator=None,
        catch_all_details_model=None
):
    for stmt in analysis_view_generator(
            catalog=catalog,
            schema=schema,
            analysis_table=analysis_table,
            primary_key_col_name=primary_key_col_name,
            domain_config_table=domain_config_table,
            analysis_column_name=analysis_column_name,
            top_level_key=top_level_key,
            view_prefix=view_prefix,
            catch_all_view_generator=catch_all_view_generator,
            catch_all_details_model=catch_all_details_model,
            error_view_generator=error_view_generator
    ):
        print("executing statement: ", stmt)
        spark.sql(stmt)