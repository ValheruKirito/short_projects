# Increasing chance of a question being answered

 The way you frame your questions can significantly influence the quality of the answers you receive. In today's information-rich environment, asking questions in the right way can lead to clearer insights and better outcomes. Our tool aims to assist you in crafting questions that are clear, relevant, and likely to elicit helpful responses, enhancing your chances of getting the information you need.

 This folder containes an app that provides a recommendations how to change formulation of a question to increase the likelihood of recieving a satisfying answer. The recommendations are based on a ML model leveraging data from [StackExchange Archive](https://archive.org/download/stackexchange/), more specifically individual posts asked in following categories: writers, politics, coffee, astronomy, biology. These threads are intentionally selected from various fields to increase the diversity of the training data. A questions score is chosen as a weak label describing questions quality during the training.

 The recommendations are based on a black-box explainer (LIME) leveraging the trained LM model. The current version of recommendations is very simple, this is more of a concept than fully polished product. The recommendations could definitely be improved in several ways:
 * creating more text-related features, that would be easier to implement by the writers,
 * improving the metrics of the used model, e.g. by further iterating on the data, possibly with those more meaningful features,
 * (after significant polishing and several rounds of such iterations) deploying the model in production and gathering feedback about what recommendations are actually usefull,
 * et cetera.
 
 Doing so should be fairly easy based on the project structure, a new model_v4 file has to be created, new vectorised data created and model versions updated in couple of locations through the script, but that is it.

 The current version of the app is not deployed online, it only provides a small, very simple GUI contained within the [app.py](app.py) script. The gui contains a text window, and window with recommendations. The recommendations can be evaluated as the text is typed (the model is fast enough for that), or switched to be evaluated once every 2000 ms.

 Depending on the goal of the deployment, either can have it's advantages. Should it be deployed on StackExchange as a guideline, how to improve likelihood of getting a good answer, a less frequent evaluations would be sufficient.

## Setup instructions

Instructions for setup are standard, install packages from [conda_requirements.txt](conda_requirements.txt) and [pip_requirements.txt](pip_requirements.txt) in your virtual environment. You also have to download data for the LLM used:

`python -m spacy download en_core_web_sm`

`python -m spacy download en_core_web_lg`

If you're looking only into the final model, it is basing it's predictions on `python -m spacy download en_core_web_sm`, the large model can be ignored.

## Notebooks and scripts

The folder contains majority of scripts used during the development. Part of these is written in a pure python (as e.g. [01-AN-get_convert_data.py](01-AN-get_convert_data.py) is called from within different script using exec, when the data files are not found in the data folder), and part is written in Jupyter notebooks for convenience.

The scripts reference core functions in the `editor` folder. Author was too lazy to develop set of thorough tests for a side project like this.

1. get_convert_data.py – downloads data from the StackExchange repository, parses XMLs, recodes them, provides raw data processing and saves the relevant data
2. v1_embed_data.py – provides feature creation and embedding of the data for the first version of the model
3. v1_train_model.py – trains the model and looks at its metrics. This version uses the large version of spaCy LLM with full embeddings, making the model slow, with individual features being hardly interpretable. It displays significatn overfittin, on the other it's calibrations on the validation set is surprisingly good.
4. v1_text_explainer.ipynb – using LIME text explainer to inspect what words and parts of speech does the model prioritise.
5. v2_train_model.ipynb – training the model on newly generated features, excludes the actual embeddings of the texts, since recommendations based on those features would be very hard to interpret.
6. v2_text_explainer.ipynb – using LIME text explainer to inspect focus of the second model.
7. v3_train_model.ipynb – just a simplified version of model 2, features with no importance or very highly correlated features are omitted. Yields comparable results, fairly good calibration, even though it clearly predicts a lot of positive cases very close to the default decision boundary.
8. v3_get_recomendations.ipynb – inspecting recommendations based on the v3 model.

# Description of the app

Firstly, actual deployment of the app could be significantly lighter, large parts of the scripts provided here are unnecesseary for the app itself, yet were used during the development.

The app can be launched running script in [app.py](app.py). It provides a very simple GUI, with a text input window, a button that allows for a switch between the evaluation modes (on key release vs. periodically) and recommendations on how to improve a questions likeness of being answered. For useres convenience a probability of being scored as a "good question" is also showed to serve as a metric of progress.

The author is definitely NOT a front-end developer and apps interface look according to it. The actual deployment would be ideal on a website.

## Showcase

Here we provide an example of improving the question while following the instructions. Evaluation mode was set to "on key release", providiing instantanous feedback.

[before advice](/images/before_advice.png)

[after 1st round](/images/after_1st_advice.png)

[after 2nd round](/images/after_2nd_advice.png)

The provided recommendations are definitely far from user friendly, as we have stated before, this is more of a concept than a polished final product.
