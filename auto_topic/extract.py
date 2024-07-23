import dspy

def make_extractor(model_name: str, api_url: str, token: str, model_kwargs: dict, ):
    lm = dspy.OpenAI(
        model=model_name,
        api_key=token,
        api_base=api_url,
        **{**model_kwargs, **{"max_tokens": 500, "temperature": 0.1}}
    )

    dspy.settings.configure(lm=lm)

