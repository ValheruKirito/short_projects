"""Get recomendations from docstring."""
import os
from cachetools import cached, TTLCache

from lime.lime_tabular import LimeTabularExplainer

from editor.data_manipulation import load_train_val_test_set
from editor.model_v3 import (
    FEATURE_LIST,
    MODEL,
    vectorise_data
)

DISPLAY_NAMES = {
    'freq_ques': 'frequency of question marks',
    'freq_peri': 'frequency of full stops',
    'freq_comm': 'frequency of commas',
    'freq_excl': 'frequency of exclamation marks',
    'polarity': 'positivity of emotional sentiment',
    'subjecti': 'subjectivity of the question',
    'num_sylla': 'question length',
    'num_diff_words': 'diversity of vocabulary',
    'avg_word_len': 'complexity of vocabulary',
    'num_stop_words': 'number of stop words',
    'flesch': 'Flesch reading ease score',
    'ADJ': 'frequency of adjectives',
    'ADP': 'frequency of adpositions',
    'ADV': 'frequency of adverbs',
    'AUX': 'frequency of auxiliary verbs',
    'DET': 'frequency of determiners',
    'INTJ': 'frequency of interjections',
    'NOUN': 'frequency of nouns',
    'NUM': 'frequency of numerals',
    'PART': 'frequency of particles',
    'PRON': 'frequency of pronouns',
    'PROPN': 'frequency of proper nouns',
    'PUNCT': 'frequency of punctuations',
    'SCONJ': 'frequency of subordinating conjunctions',
    'SYM': 'frequency of symbols',
    'VERB': 'frequency of verbs',
    'X': 'frequency of others'
}

CACHE = TTLCache(maxsize=256, ttl=600)


def make_explainer(model: str = 'v3'):
    """Creates and trains tabular explainer.

    Args:
        model (str): String identifying the model version. Passed into the
        selection of training data.
    """
    train_X, _, _, train_y, *_ = load_train_val_test_set(model)
    explainer = LimeTabularExplainer(
        training_data=train_X.values,
        feature_names=FEATURE_LIST,
        training_labels=train_y,
        class_names=['low', 'high']
    )
    return explainer

EXPLAINER = make_explainer()


def get_list_explanation(
    instance: str,
    vectorise: bool = False
    ) -> list:
    """Generates list of explanations.

    Args:
        instance (str): text to be explained

    Returns:
        list: list of explanations
    """
    vector = vectorise_data(instance)
    exp = EXPLAINER.explain_instance(
        vector.squeeze(),
        MODEL.predict_proba,
        num_features=vector.shape[1],
        labels=(1,)
        )
    if vectorise:
        return exp.as_list(), vector
    return exp.as_list()

def parse_explanations(exp_list: list) -> dict:
    """Parse explanation list into individual parts.

    Args:
        exp (dict): Dictionary with explanations for each feature.
    """
    exp_list = [[expl[0].split(' '), expl[1]] for expl in exp_list]
    parsed = dict()
    for feat in FEATURE_LIST:
        parsed[feat] = [
            [expl[0][1], float(expl[0][2]), expl[1]]
            for expl in exp_list
            if (expl[0][0] == feat and len(expl[0]) == 3) 
        ]
        parsed[feat] = [item for expl in parsed[feat] for item in expl]
        if parsed[feat] == []:
            del parsed[feat]
    parsed = dict(sorted(parsed.items(), key=lambda item: (item[1])[2]))
    return parsed


def recommend_what_to_do(unequal:str, impact: float) -> str:
    """Suggest what to do based on the inequality sign in parsed explanations.

    Args:
        less_more (str): Inequality sign as a string to be interpreted.

    Returns:
        str: recomendation
    """
    if unequal in ['>', '>='] and impact > 0:
        return 'pass'
    if unequal in ['<', '<='] and impact > 0:
        return 'pass'
    if unequal in ['>', '>='] and impact < 0:
        return 'decrease '
    if unequal in ['<', '<='] and impact < 0:
        return 'increase '
    


def recommend_from_expl(parsed_expl: dict) -> str:
    """Make recommendations based on parsed explanations.

    Args:
        parsed_expl (dict): parsed explanations

    Returns:
        str: string of recomnedations
    """
    result_string = ''
    for feat in parsed_expl:
        if feat in ['flesch', 'X']:
            continue
        exp = parsed_expl[feat]
        if exp == [] or recommend_what_to_do(exp[0], exp[2]) == 'pass':
            pass
        else:
            result_string += (
                f'{recommend_what_to_do(exp[0], exp[2])}'
                + f'{DISPLAY_NAMES[feat]}\n'
            )
    return result_string


def get_reading_level_from_flesch(flesch: float) -> str:
    """
    Assing standardised reading level to Flesch score.

    Parameters
    ----------
    flesch : float
        Flesch reading ease score.

    Returns
    -------
    str
        Standardised reading level.

    """
    # construct thresholds
    thresholds = [90, 80, 70, 60, 50, 30, 10, 0]
    levels = [
        '5th grader could read this',
        '6th grader could read this',
        '7th grader could read this',
        '8th & 9th graders could read this',
        '10th to 12th graders could read this',
        'College students could read this',
        'College graduates could read this',
        'Professionals could read this'
    ]
    # check conditions from highest threshold to lowest (same result as a
    # sequence of elif statements)
    for thresh, lev in zip(thresholds, levels):
        if flesch >= thresh:
            return lev
        elif flesch < 0:
            return 'Professionals could read this'


@cached(CACHE)
def get_recommendation_report(text: str) -> str:
    """Pipeline recommendations.

    Args:
        text (str): Text to get recommendations for.

    Returns:
        str: Full report including score.
    """
    exp, vector = get_list_explanation(text, vectorise=True)
    parsed = parse_explanations(exp)
    recom_string = recommend_from_expl(parsed)
    score = 100*MODEL.predict_proba(vector)[0][1]
    result_string = f'Overall score is {score:.3f}.\n'
    flesch = vector.loc[0, 'flesch']
    result_string += f'Flesch reading ease score is {flesch:.3f} – '
    result_string += get_reading_level_from_flesch(flesch) + '\n\n'
    result_string += recom_string
    return result_string
