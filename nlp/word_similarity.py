import numpy as np
import spacy
import pandas as pd

from pathlib import Path
from numpy import ndarray
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

RANDOM_STATE = 42

nlp = spacy.load("en_core_web_lg")


def get_token_vectors(text: str):
    return np.array([token.vector for token in nlp(text)])


def cosine_similarity(word1: ndarray, word2: ndarray):
    numerator = word1.dot(word2)
    denominator = np.sqrt(word1.dot(word1) * word2.dot(word2))
    return numerator / denominator


def word_similarity_main():
    word_vectors = get_token_vectors("king queen man woman actor actress")
    king = word_vectors[0]
    queen = word_vectors[1]
    man = word_vectors[2]
    woman = word_vectors[3]
    actor = word_vectors[4]
    actress = word_vectors[5]
    male_to_female_wordvec = woman - man
    print(f"similarity 'king-queen': {cosine_similarity(king, queen)}")
    print(f"similarity 'king + male_to_female_vector - queen': "
          f"{cosine_similarity(king + male_to_female_wordvec, queen)}")
    print(f"similarity 'actor-actress': {cosine_similarity(actor, actress)}")
    print(f"similarity 'actor + male_to_female_vector - actress:"
          f"{cosine_similarity(actor + male_to_female_wordvec, actress)}")
    print()


def yelp_review_classifier_main():
    yelp_data = pd.read_csv(Path("data") / "yelp.csv")
    print("yelp data overview:")
    print(yelp_data.describe())

    assert True not in yelp_data.stars.isnull().value_counts()  # no rows with missing target column

    # create target label (reviews with more than 3 stars are good)
    y = pd.Series([1 if stars > 3 else 0 for stars in yelp_data.stars])
    yelp_data["y"] = y # add target (used later for data augmentation)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(yelp_data,
                                                        y,
                                                        test_size=0.1,
                                                        random_state=RANDOM_STATE)

    # Augment useful tests for training set only (to avoid train-test contamination)
    # since 1.409300 is the std for usefulness
    # reviews with useful >= 3 are 2 stds to the right
    # Todo: test 'useful' for normality
    useful_reviews = X_train[X_train.useful >= 3]
    useful_review_ys = useful_reviews.y
    X_train = pd.concat([X_train, useful_reviews])
    y_train = pd.concat([y_train, useful_review_ys])

    # vectorize docs (average out words in every text to a single vector),
    # remove all other features (including target: y)
    X_train =  [nlp(text).vector for text in X_train.text.values]
    X_test = [nlp(text).vector for text in X_test.text.values]

    # train and test SVC model
    svc = LinearSVC(random_state=RANDOM_STATE, max_iter=500)
    print("training...")
    svc.fit(X_train, y_train)

    print(f"svc accuracy: {svc.score(X_test, y_test) * 100}")


if __name__ == '__main__':
    # compare similarity of different words, user vector operators to bridge between words
    # and semantic paths within the space
    word_similarity_main()

    # SVC classifier for yelp review using vectorized docs as input
    yelp_review_classifier_main()
