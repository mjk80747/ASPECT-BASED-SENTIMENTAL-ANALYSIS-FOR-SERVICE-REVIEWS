from app import classify_sentiment, extract_aspects


def test_model_loads_and_predicts_sentiment():
    sample = "The food was amazing and the staff were friendly."
    vectorized = classify_sentiment
    assert vectorized is not None

    from app import cv, model

    assert cv is not None
    assert model is not None

    prediction = classify_sentiment(sample, cv.transform([sample]).toarray())
    assert prediction in {
        "Very Negative",
        "Negative",
        "Neutral",
        "Mixed",
        "Positive",
        "Very Positive",
    }

    aspects = extract_aspects("delivery was slow, but customer support was helpful")
    assert isinstance(aspects, list)
    assert len(aspects) >= 1
