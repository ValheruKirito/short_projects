"""Download, recode, parse, clean and save data used for model training."""
import os
import sys
import xml.etree.ElementTree as ElT
import chardet
import pandas as pd
import py7zr
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
# local import
from editor.project_heuristics import clean_input

sys.path.append('../')


work_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(work_dir)


# =============================================================================
# %% Definitions
# =============================================================================
def edit_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Edit raw data.

    Parameters
    ----------
    df : pd.DataFrame
        Takes the raw data loaded from the data csv and makes some adjustments.

    Returns
    -------
    pd.DataFrame
        Returns the adjusted data in a csv.

    """
    # dropping unnecessary index column
    df = df.drop(['Unnamed: 0'], axis='columns')
    # dropping all posts without any text
    df = df.dropna(subset=['body_text'])
    # defining a question indication columns
    df['is_question'] = df['PostTypeId'] == 1
    # dropping everything that is not a question or an answer
    df = df[df['PostTypeId'] == 1]
    # adding no user id to owner
    df['OwnerUserId'] = df['OwnerUserId'].fillna(-1)
    df['OwnerUserId'] = df['OwnerUserId'].astype(int)
    df['PostTypeId'] = df['PostTypeId'].astype(int)
    # add indicator column whether the question is answered
    df['is_answered'] = pd.Series()
    df.loc[df.is_question, 'is_answered'] = \
        ~ df[df.is_question].AcceptedAnswerId.isnull()
    # add full text
    df['full_text'] = df['Title'].str.cat(df['body_text'], sep=' ', na_rep=',')
    df['full_text'] = df['full_text'].astype(str)
    # strip non ascii characters
    clean_text = []
    for text in tqdm(df['full_text'], desc='Full text cleaning'):
        clean_text.append(clean_input(text))
    df['full_text'] = clean_text
    #_________________________
    # droping bunch of columns just to get smaller data
    df = df.drop(
        [
            'CreationDate', 'ViewCount', 'FavoriteCount', 'LastEditorUserId',
            'LastEditDate',	'LastActivityDate', 'Title', 'Tags', 'AnswerCount',
            'CommentCount', 'ContentLicense', 'ClosedDate', 'OwnerDisplayName',
            'LastEditorDisplayName', 'CommunityOwnedDate', 'Body', 'body_text',
            'PostTypeId', 'Id', 'ParentId'
        ],
        axis='columns'
    )
    return df


def detect_encoding(source_path):
    """
    Return encoding of a file.

    Parameters
    ----------
    file_path : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    with open(source_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding']


def parse_xml_to_csv(path: str, save_path: str = None) -> pd.DataFrame:
    """
    Open .xml and convert the text to a csv, tokenizing it in the process.

    Parameters
    ----------
    path : str
        path to the xml document containing data
    save_path : str, optional
        path where to save parsed csv

    Returns
    -------
    pd.DataFrame, or None
        Pandas dataframe with the loaded parsed data. If parsing error occurs,
        error is raised and None is returned.

    """
    try:
        # Python' standard library to parse XML files
        doc = ElT.parse(path)
        root = doc.getroot()

        # Each row is a question
        all_rows = [row.attrib for row in root.findall('row')]

        # Using tqdm to display progress since preprocessing takes time
        for item in tqdm(all_rows, desc='Converting'):
            # Decode text from HTML
            if 'Body' in item.keys():
                soup = BeautifulSoup(item['Body'], features='html.parser')
                item['body_text'] = soup.get_text()
            else:
                item['body_text'] = ""
        # create a dataframe from our list of dictionaries
        df = pd.DataFrame.from_dict(all_rows)
        if save_path:
            df.to_csv(save_path)
        return df

    except ElT.ParseError as e:
        print(f'Parsing raised error: {e}')
        print('Document parsing stopped.')
        return None


# =============================================================================
# %% Code
# =============================================================================

if __name__ == '__main__':

    strain_names = ['writers', 'politics', 'coffee', 'astronomy', 'biology']
    os.mkdir('data/')

    for name in strain_names:
        archive_name = f'{name}.stackexchange.com.7z'

        # Download the file from repository
        response = requests.get('https://archive.org/download/stackexchange/' +
                                archive_name,
                                stream=True,
                                timeout=(30, None))

        # Get the total file size in bytes
        total_size = int(response.headers.get('content-length', 0))

        # Use tqdm to create a progress bar
        with open(archive_name, 'wb') as f:
            with tqdm(total=total_size,
                      unit='B',
                      unit_scale=True,
                      desc='Downloading',
                      ascii=True
                      ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    f.write(data)
                    pbar.update(len(data))

        # directory to unpack archive and file path
        data_dir = os.path.join(work_dir, f'data_stack_exchange_{name}')
        file_path = os.path.join(data_dir, 'Posts.xml')
        # Open the 7z archive
        with py7zr.SevenZipFile(archive_name) as archive:
            # Get the names of all files in the archive
            extracted_files = archive.getnames()
            archive.extract(data_dir, 'Posts.xml')

        # Detect the encoding of the extracted file
        encoding = detect_encoding(file_path)

        # Re-encode the file with the detected encoding into the right dir
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()
        # remove the temporary file
        os.remove(file_path)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        # convert POSTS files to csv
        result_path = os.path.join(work_dir, f'data/{name}.csv')
        data = parse_xml_to_csv(path=file_path,
                                save_path=result_path)

        # remove the archive, work xml file and directory
        os.remove(archive_name)
        os.remove(file_path)
        os.removedirs(data_dir)

    results_df = pd.read_csv(f'data/{strain_names[0]}.csv')
    for name in strain_names[1:]:
        work_df = pd.read_csv(f'data/{name}.csv')
        results_df = pd.concat([results_df, work_df], axis=0)

    # light editing
    results_df = edit_raw_data(results_df)
    # save data
    results_df.to_csv('data/data.csv', index=True)
