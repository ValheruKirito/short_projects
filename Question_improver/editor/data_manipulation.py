"""Process and manipulate data, generate features."""
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import sklearn

curr_path = Path(os.path.dirname(__file__))

def get_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract parts of Creation date into separate columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that was at least raw-processed.

    Returns
    -------
    pd.DataFrame
        DataFrame with added columns for year, month, day and hour of creation.

    """
    # extract parts of dates
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    return df


def get_tags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Separate tags into individual columns.

    Parameters
    ----------
    df : pd.DataFrame
        Data DataFrame.

    Returns
    -------
    pd.DataFrame
        Data DataFrame with tags separated into indiviudal columns.

    """
    # one hot encoding of tags
    tags = df['Tags']
    clean_tags = tags.str.split('|')
    # remove empty strings produced by previous line
    for i, row in enumerate(clean_tags):
        if isinstance(row, float):
            pass
        else:
            clean_tags.iloc[i] = [tag for tag in row if tag != '']

    # append all tags to data
    clean_tags = clean_tags.apply(pd.Series)
    clean_tags.columns = ['tag_' + str(i) for i in range(1, 6)]
    return pd.concat([df, clean_tags], axis=1)


def encode_best_tags(df: pd.DataFrame, n_tags: int):
    """
    One-hot encode n most popular tags.

    Parameters
    ----------
    df : pd.DataFrame
        Data DataFrame.
    n_tags : int
        Number of most popular tags to be one-hot encoded.

    Returns
    -------
    pd.DataFrame
        Data DataFrame with one-hot encoded columns of most popular tags.
    list
        List of the names of the selected tags.

    """
    clean_tags = get_tags(df)[['tag_' + str(i) for i in range(1, 6)]]
    tag_columns = pd.get_dummies(clean_tags.stack()).groupby(level=0, axis=0
                                                             ).sum()
    all_tags = tag_columns.sum(axis=0).sort_values(ascending=False)
    # select requested n_tags
    top_tag_cols = tag_columns[all_tags.index[:n_tags]]
    # append one hot encoded top tags to data
    return pd.concat([df, top_tag_cols], axis=1), top_tag_cols



def standard_scale_col(df: pd.DataFrame, col_name: str) -> pd.Series:
    """
    Standardise DataFrame column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with data to be standardised
    col_name : str
        name of the column to be standardised

    Returns
    -------
    pd.Series
        Returns standardised DataFrame column.

    """
    return (df[col_name] - df[col_name].mean()) / df[col_name].std()


def create_train_test_split(df: pd.DataFrame):
    """
    Create train-test split controlling authors in different sets.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with or without generated features.

    Returns
    -------
    pd.DataFrame
        Returns train set
    pd.Series
        Returns test set.
    """
    # create train-test split
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=20)
    splits = splitter.split(df, groups=df['OwnerUserId'])
    train_idx, test_idx = next(splits)
    train_data = df.iloc[train_idx, :]
    test_data = df.iloc[test_idx, :]
    return train_data, test_data


def load_train_val_test_set(model: str):
    """
    Load data from files and return consistent train/val/test data splits.

    Parameters
    ----------
    model : str : 'v1'
        String identifying data to fetch based on the model version.

    Returns
    -------
    Returns consistent splits in this order:
        train_X, val_X, test_X, train_label, val_label, test_label

    """
    if model not in ['v1', 'v2', 'v3']:
        raise AttributeError(model, ' is not a valid model')
    vector_path = Path(f'../data/vectorised_features_{model}.csv')
    vectors = pd.read_csv(curr_path / vector_path)
    vectors = vectors.drop('Unnamed: 0', axis='columns')
    data_path = Path(f'../data/data.csv')
    data = pd.read_csv(
        curr_path / data_path,
        usecols=['is_question', 'OwnerUserId', 'Score']
    )
    vectors = pd.concat(
        [vectors, data.loc[data.is_question, ['OwnerUserId', 'Score']]],
        axis=1
    )
    # to clean up memory space
    del data

    # train-val-test split with proper author splitting
    train_set, test_set = create_train_test_split(vectors)
    train_set, val_set = create_train_test_split(train_set)

    # threshhold for labels based only on training set
    thresh = train_set['Score'].median()
    # X, y data
    train_label = (train_set['Score'] > thresh).astype(int)
    val_label = (val_set['Score'] > thresh).astype(int)
    test_label = (test_set['Score'] > thresh).astype(int)

    train_X = train_set.drop(['OwnerUserId', 'Score'], axis='columns')
    val_X = val_set.drop(['OwnerUserId', 'Score'], axis='columns')
    test_X = test_set.drop(['OwnerUserId', 'Score'], axis='columns')
    return train_X, val_X, test_X, train_label, val_label, test_label
