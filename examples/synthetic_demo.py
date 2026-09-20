"""Small example that does not use the EastEnders dataset."""

from character_retrieval import CharacterDialogueRetriever, FeatureConfig


training = {
    "ALICE": "tea please lovely morning tea and biscuits",
    "BOB": "football match goal team football tonight",
    "CAROL": "garden roses flowers lovely garden plants",
}

# A simple splitter keeps this demo independent of downloadable NLTK corpora.
retriever = CharacterDialogueRetriever(
    features=FeatureConfig(min_df=1, include_style=False),
    preprocessor=lambda text: text.lower().split(),
).fit(training)

for character, similarity in retriever.rank("team scored a football goal"):
    print(f"{character:5s} {similarity:.3f}")
