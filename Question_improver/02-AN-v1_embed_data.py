"""Look at the data from Stack Exchange."""
import os
import pandas as pd
# local imports
from editor.model_v1 import vectorise_data, get_text_features_v1


work_path = os.path.abspath(os.path.dirname(__file__))
os.chdir(work_path)

# =============================================================================
# data import
# =============================================================================
# throw the data into the data frame
FILE_PATH = 'data/data.csv'
# if the file exists
if not os.path.exists(FILE_PATH):
    with open('get_convert_data.py', 'rb') as script:
        exec(script.read())
    FILE_PATH = 'data/data.csv'
data = pd.read_csv(FILE_PATH)

# =============================================================================
# vectorised data
# =============================================================================
data = get_text_features_v1(data)
vect_feat = vectorise_data(
    data.loc[data.is_question, 'full_text'],
    tracker=True
)

# =============================================================================
# think about casting this to a database
# =============================================================================
# matched with data based on index (order of exported columns)
vect_feat.to_csv('data/vectorised_features_v1.csv')
