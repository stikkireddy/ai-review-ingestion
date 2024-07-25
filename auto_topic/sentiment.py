import copy
import json
from typing import Optional

import dspy
import pandas as pd
from openai import BaseModel
from pydantic import Field, create_model


def enable_arize_tracing():
    from arize_otel import register_otel, Endpoints

    # Setup OTEL via our convenience function
    register_otel(
        endpoints = Endpoints.PHOENIX_LOCAL,
        model_id = "juan", # name this to whatever you would like
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
    return {row['topic'].lower()+"details": row['when'].format(topic=row['topic'].lower()+"details") for row in rows}
  

def get_analyzer(df: pd.DataFrame, language_model):
    dspy.settings.configure(lm=language_model)
    valid_responses = get_valid_responses_for_categories(df) 
    print(valid_responses)
    when = get_when_to_use_category(df) 
    print(when)

    class IdentifyCategories(dspy.Signature):
        """You are a customer feedback analyst. Your work at a large shoe store retailer and oversee customer feedback across all channels, such as website, social media, in store reviews, customer reviews via email and your goal is to identify which categories the feedback is about."""
        feedback = dspy.InputField(desc="a review from a customer. this can be a positive or negative review")
        categories = dspy.OutputField(desc="the categories of the feedback. one of: " + ", ".join(valid_responses) + f" and here is when to use each category {str(when)}. Respond with catchalldetails if not related to any of the categories. You can pick multiple categories and respond with a comma separated list.")
        rationale = dspy.OutputField(desc="rationale for picking the categories explain step by step")



    class CatchAllDetails(BaseModel):
        details: str = Field(description="details of the feedback")
        sentiment: str = Field(description="sentiment of the feedback <positive/negative/netural>")
        keywords: list[str] = Field(description='keywords in the feedback ["keyword1", "keyword2", "keyword3", ...]')
        ai_rating: str = Field(description="rating the feedback 1-5 1 being the worst 5 being the best")
        rationale: str = Field(description="rationale for the rating and sentiment")


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
            kwargs["catchalldetails"] = (CatchAllDetails, Field(description="fill the parts of the feedback that are not relevant to the other categories."))
        return create_model("BreakDown",
                            **kwargs,
                            __base__=BaseModel)

    def identified_categories_to_categoriy_signature(prediction):
        cats = [cat.strip() for cat in prediction.categories.split(",")]
        cats = [cat for cat in cats if cat in valid_responses + ["catchalldetails"]]
        if len(cats) == 0:
            return ["catchalldetails"]
        return cats
        
    # identified_categories = identified_categories_to_categoriy_signature(prediction)



    class Categorize(dspy.Signature):

        """You are a customer feedback analyst. Your work at shoe store retailer and oversee customer feedback across all channels, such as website, social media, in store reviews, customer reviews via email and your goal
        is to extract meaning from the customer reviews. Make sure the output is valid json. Make sure
        the output starts with { and ends with }."""

        feedback = dspy.InputField(desc="a review from a customer. this can be a positive or negative review")
        rating = dspy.InputField(desc="use this for the connotation of the feedback. the scale ranges from 1 - 5. 1 being horrible 5 being great")

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
            category_selection_rationale = predict_categories.rationale
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







