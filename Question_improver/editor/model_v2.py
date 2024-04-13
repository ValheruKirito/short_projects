"""Stores functions related to the second verion of the editor."""
import os
from pathlib import Path

# import numpy as np
import pandas as pd
import joblib
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from spacy_syllables import SpacySyllables
from tqdm import tqdm

curr_path = Path(os.path.dirname(__file__))
tqdm.pandas()

FEATURE_LIST = [
    'freq_ques',
    'freq_peri',
    'freq_comm',
    'freq_excl',
    'polarity',
    'subjecti',
    'text_len',
    'num_words',
    'num_sylla',
    'num_sente',
    'num_diff_words',
    'avg_word_len',
    'num_stop_words',
    'flesch'
]

# names of position names for spacy model to count
POS_NAMES = {
    "ADJ": "adjective",
    "ADP": "adposition",
    "ADV": "adverb",
    "AUX": "auxiliary verb",
    "CONJ": "coordinating conjunction",
    "DET": "determiner",
    "INTJ": "interjection",
    "NOUN": "noun",
    "NUM": "numeral",
    "PART": "particle",
    "PRON": "pronoun",
    "PROPN": "proper noun",
    "PUNCT": "punctuation",
    "SCONJ": "subordinating conjunction",
    "SYM": "symbol",
    "VERB": "verb",
    "X": "other",
}

# EMB_LIST = [f'emb_{k}' for k in range(96)]

model_path = Path("../models/model_v2.pkl")
MODEL = joblib.load(curr_path / model_path)

NLP = spacy.load('en_core_web_sm')
NLP.add_pipe('syllables', after='tagger')
NLP.add_pipe('spacytextblob')


def count_syllables(doc) -> int:
    """Count number of syllables in the preprocessed spacy text.

    Args:
        spacy_text (doc): _description_
    """
    return sum(token._.syllables_count or 0 for token in doc)


def get_text_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds model version 2 features to the data.

    Args:
    df : pd.DataFrame
        Dataframe with generated v1 text features.

    Returns:
    pd.DataFrame :
        DataFrame with added text features from this version.
    """
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df, columns=['full_text'])
    ## numerical features
    df['text_len'] = df.full_text.str.len()
    df['freq_ques'] = df.full_text.str.count(r'\?') / df.text_len
    df['freq_peri'] = df.full_text.str.count(r'\.') / df.text_len
    df['freq_comm'] = df.full_text.str.count(',') / df.text_len
    df['freq_excl'] = df.full_text.str.count('!') / df.text_len
    ## text features
    text_spacy = df.full_text.progress_apply(NLP)
    df['polarity'] = text_spacy.apply(lambda text: text._.blob.polarity)
    df['subjecti'] = text_spacy.apply(lambda text: text._.blob.subjectivity)
    ## numerical text stats
    df['num_words'] = text_spacy.apply(len)
    df['num_sylla'] = text_spacy.apply(count_syllables)
    df['num_sente'] = text_spacy.apply(lambda x: len(list(x.sents)))
    df['num_diff_words'] = text_spacy.apply(lambda x: len(set(x)))
    df['avg_word_len'] = df.text_len / df.num_words
    df['num_stop_words'] = (
        text_spacy.apply(
            lambda x: 100 * len([stop for stop in x if stop.is_stop])
        )
        / df.text_len
    )
    df['flesch'] = (
        206.835
        - 1.015*(df.num_words/df.num_sente)
        - 84.6*df.num_sylla/df.num_words
    )
    ## frequency of various word types
    pos_list = text_spacy.apply(lambda doc: [token.pos_ for token in doc])
    for pos_name in POS_NAMES:
        df[pos_name] = (
            pos_list.apply(
                lambda x: len([match for match in x if match == pos_name])
            )
            / df.text_len
        )
    ###########################################################################
    # full text embedding doesn't improve model capabilities
    # embed full text
    # embed = text_spacy.progress_apply(lambda x: x.vector)
    # embed = pd.DataFrame(
    #     np.vstack(embed),
    #     columns=EMB_LIST,
    #     index=df.index
    #     )
    # df = pd.concat([df, embed], axis='columns')
    return df


def vectorise_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create feature vectorisation from a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with generated v0 and v1 text features.

    Returns
    -------
    pd.DataFrame
        Returns vectorised representation of the data

    """
    if isinstance(df, str):
        df = pd.DataFrame([df], columns=['full_text'])
    df = get_text_features_v2(df.copy())
    return df[
        FEATURE_LIST
        + list(POS_NAMES.keys())
        # + EMB_LIST
    ]


def get_proba_from_input(text) -> float:
    """
    Return probability that the selected input has good score.

    Parameters
    ----------
    text : str, 1d array-like
        Array of texts for prob pred.

    Returns
    -------
    float
        Probability that the selected output has good scores.

    """
    vector = vectorise_data(text)
    return MODEL.predict_proba(vector)
