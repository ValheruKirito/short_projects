#!/bin/bash
conda install -r conda_requirements.txt
pip install -r pip_requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg