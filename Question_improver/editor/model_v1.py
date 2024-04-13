"""Stores functions related to the first verion of the editor."""
import pandas as pd
import numpy as np
import joblib
import spacy
from tqdm import tqdm
from editor.project_heuristics import get_flesch_from_text

FEATURE_LIST = ['incl_ques',
                'action_verb',
                'lang_ques',
                'text_len',
                'flesch'
                ]
MODEL = joblib.load('../models/model_v1.pkl')


def get_text_features_v1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create basic text features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that has v0 features.

    Returns
    -------
    pd.DataFrame
        Returns DataFrame with added features. Adds:
            Full text - join Title and body text
            Text len - length of the FullText.

    """
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df, columns=['full_text'])
    # concat title to body text and compute its length
    df['incl_ques'] = df['full_text'].str.contains('?', regex=False)
    df['action_verb'] = df['full_text'].str.contains(
        r'(?i)\bcan\b|\bshould\b|\bwhat\b')
    df['lang_ques'] = df['full_text'].str.contains(
        r'(?i)\bpunctuate\b|\bcapitalize\b|\babbreviate\b')
    df['flesch'] = [get_flesch_from_text(text) for text in df['full_text']]
    # calculate text length
    df['text_len'] = df['full_text'].str.len()
    return df


def create_embeddings(text_series: pd.Series, tracker: bool = False) -> list:
    """
    Embed the input texts.

    Parameters
    ----------
    text_series : pd.Series
        Pandas series of texts to be embedded into a vector space.
        All positions must be filled, no NANs allowed.
    tracker : bool
        Default is False. If True enables for tracking progress of embeddings
        for long series (e.g. during embedding of training data).

    Returns
    -------
    list
        List of embeddings for each individual text in the series.

    """
    nlp = spacy.load(
        'en_core_web_lg',
        disable=['ner', 'textcat']
    )
    # loop instead of comprehension to track progress
    embeddings = []
    if tracker:
        for text in tqdm(text_series, desc='Embedding'):
            embeddings.append(nlp(text).vector)
    else:
        embeddings = [nlp(text).vector for text in text_series]
    return embeddings


def vectorise_data(df: pd.DataFrame, tracker: bool = False) -> pd.DataFrame:
    """
    Create feature vectorisation from a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with generated v0 and v1 text features.
    tracker : bool
        Default is False. If True enables for tracking progress of embeddings
        for long series (e.g. during embedding of training data).

    Returns
    -------
    pd.DataFrame
        Returns vectorised representation of the data

    """
    df = get_text_features_v1(df)
    embed = np.array(create_embeddings(df['full_text'], tracker=tracker))
    embed = np.append(embed, df[FEATURE_LIST], axis=1)
    columns = ['emb_'+str(i) for i in range(300)] + FEATURE_LIST
    return pd.DataFrame(embed,
                        columns=columns,
                        index=df.index
                        )


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
    if isinstance(text, str):
        text = [text]
    vector = vectorise_data(text)
    return MODEL.predict_proba(vector)
