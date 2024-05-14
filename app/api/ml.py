from fastapi import APIRouter
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

router = APIRouter()

description_classifier = pipeline(
    "zero-shot-classification", model="facebook/bart-large-mnli"
)

sentiment_analyzer = SentimentIntensityAnalyzer()


@router.post("/classify_offer")
async def classify_offer(*, description: str):
    candidate_labels = [
        "Accommodation",
        "Sports",
        "Adventure",
        "Food",
        "Wellness",
        "Transportation",
        "Culture",
        "Drinks",
        "Café",
        "Games",
    ]

    classification = description_classifier(description, candidate_labels)

    tags_scores = zip(classification["labels"], classification["scores"])
    sorted_tags_scores = sorted(tags_scores, key=lambda x: x[1], reverse=True)
    tags = [label for label, score in sorted_tags_scores[:2]]

    return {"tags": tags}


@router.post("/sentiment_analysis")
async def sentiment_analysis(*, review: str):
    sentiment = sentiment_analyzer.polarity_scores(review)
    score = (sentiment["compound"] + 1) / 2

    return score
