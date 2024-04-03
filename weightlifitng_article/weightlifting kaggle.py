# %% [markdown]
# # The importance of choosing the right metric, in other words, don't trust everything labeled as 'Machine learning'

# %% [markdown]
# ## Original article
# An article *Health challenges and acute sports injuries restrict weightlifting training of older athletes* published in *BMJ Open Sport & Exercise Medicine* in 2022 shows an interesting study of factors influencing injuries of master athletes who are dedicating their time to weightlifting.
# * Huebner M, Ma W Health challenges and acute sports injuries restrict weightlifting training of older athletes BMJ Open Sport & Exercise Medicine 2022;8:e001372. doi: 10.1136/bmjsem-2022-001372
# 
# Dataset based on an online survey amongst top weightlifters from various continents and various ages is made publicly available and is thus open to further investigations ([https://doi.org/10.5061/dryad.51c59zwb3](https://doi.org/10.5061/dryad.51c59zwb3)).
# 
# We are not going to delve deep into the original article, specific questions and considered injury predictors are not of importance for this notebook, we'll just provide a brief overview of its implementation of *machine learning models*. Several machine learning algorithms were used to "evaluate potentially complex interactions of age, sex, health-related and training-related predictors of injuries". Since the goal here is to predict injury, the task at hand is a classification problem. For this purpose the authors implemented:
# * support vector machines,
# * generalised boosted methods,
# * regularised logistic regression,
# * random forests, and
# * one layer neural network.
# 
# These classifiers were trained using 10-fold repeated cross-validation to tune hyperparameters on 80 % of the data as a training set. Remaining 20 % were used for model evaluation using *accuracy* as a metric.

# %% [markdown]
# ## Their results
# 
# All the classifiers the authors implemented yielded comaparable accuracy as shown in **Table 2** of the original article.

# %% [code] {"jupyter":{"outputs_hidden":false},"_kg_hide-input":true,"execution":{"iopub.status.busy":"2024-04-01T18:33:20.981618Z","iopub.execute_input":"2024-04-01T18:33:20.982052Z","iopub.status.idle":"2024-04-01T18:33:21.000824Z","shell.execute_reply.started":"2024-04-01T18:33:20.982024Z","shell.execute_reply":"2024-04-01T18:33:20.999737Z"}}
%matplotlib inline
from IPython.display import Image

# Read the image
Image('/kaggle/input/weightlifting-injuries-in-master-athletes/'
      'Table 2 orig article.png')

# %% [markdown]
# So they are able to predict injuries in master wightlifters with accuracy of about 75 % and for hip injuries with about 90 % accuracy! So the authors calculated feature importance metrics and made conclusions about variables that have the strongest influence on occurence of injury.
# 
# **This sounds fairly impressive, doesn't it?** Well, yes, except for one very significant detail. And this one is a biggie. But let's inspect the data ourselves.
# 
# # Exploratory data analysis
# 
# ## Importing libraries
# 
# We import standard data handling libraries (**Pandas**, **NumPy**), visualisation libraries (**PyPlot**, **Seaborn**) as well as some machine learning libraries (**SciKit-learn** and **PyTorch**). All the imports are handled at top of the file in accordance with PEP 8 standard. (Since this notebook is not meant as a learning tool for others we do not comply here with the more illustrative approach of importing things 'on the go' in respective cells.)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-04-01T18:33:21.002760Z","iopub.execute_input":"2024-04-01T18:33:21.003017Z","iopub.status.idle":"2024-04-01T18:33:32.910453Z","shell.execute_reply.started":"2024-04-01T18:33:21.002995Z","shell.execute_reply":"2024-04-01T18:33:32.909509Z"}}
# skorch allow for easy integration of PyTorch neural nets into SciKit-learn
# modules – pipelines, Grid/RandomisedSearchCV etc.
!pip install skorch
import pandas as pd  # data import and handling
import numpy as np
import seaborn as sns  # specific statistical plots, visual adjustments
import sklearn.pipeline as pp
import statsmodels.api as sm
import torch
import torch.optim as optim
import torch.nn as nn
import warnings
from IPython.display import Image
from matplotlib import pyplot as plt  # plotting the data
from scipy import stats as st
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, \
    f1_score, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split, GridSearchCV, \
    StratifiedKFold, KFold
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from skorch import NeuralNetBinaryClassifier
from skorch.callbacks import EarlyStopping
warnings.filterwarnings('ignore', category=FutureWarning)

# sets graphical style for plots
sns.set_theme(style='ticks', palette='bright')

# %% [markdown]
# ## Data import, data quality checks
# 
# **Point of this notebook is nowhere to be found in this part, feel free to skip to the next one.**
# 
# Let us load in the data and have a first look at specific columns/features.

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-04-01T18:33:32.911860Z","iopub.execute_input":"2024-04-01T18:33:32.912131Z","iopub.status.idle":"2024-04-01T18:33:32.943296Z","shell.execute_reply.started":"2024-04-01T18:33:32.912106Z","shell.execute_reply":"2024-04-01T18:33:32.942483Z"}}
# data import
data = pd.read_csv('/kaggle/input/weightlifting-injuries-in-master-athletes/'
                   'wlinj_dryad.csv',
                   index_col=0)
# display head for first look
data.head()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-04-01T18:33:32.944335Z","iopub.execute_input":"2024-04-01T18:33:32.944607Z","iopub.status.idle":"2024-04-01T18:33:32.968597Z","shell.execute_reply.started":"2024-04-01T18:33:32.944584Z","shell.execute_reply":"2024-04-01T18:33:32.967785Z"}}
# let's gather some info about the data
info = pd.DataFrame(index=data.columns)
# types of individual features
info['types'] = data.dtypes
# check for missing values
info['num_NA'] = data.isna().sum().to_frame()
# number of unique values per feature
info['num_uniques'] = data.nunique()
info

# %% [markdown]
# We have no missing data (this is not surprising – the dataset was pre-processed for the original study). This leaves out part of data preparation concerns.
# 
# Some categorical data (sex, agegrp3 and ) is still coded as strings instead of numbers. As those are either binomial (sex and hips) or ordinal (agegrp3), let's use simple label encoding.

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-04-01T18:33:32.971662Z","iopub.execute_input":"2024-04-01T18:33:32.971970Z","iopub.status.idle":"2024-04-01T18:33:32.982657Z","shell.execute_reply.started":"2024-04-01T18:33:32.971948Z","shell.execute_reply":"2024-04-01T18:33:32.981925Z"}}
# recoding string coded variables
data.hips = data.hips.replace({'yes': 1, 'no': 0}).astype(int)
data.sex = data.sex.replace({'m': 0, 'f': 1}).astype(int)
data.agegrp3 = data.agegrp3.replace({'35-44': 1, '45-59': 2, '60+': 3}
                                    ).astype(int)

# %% [markdown]
# ## Injuries columns
# 
# Majority of the columns here are not important for this notebook as they describe predictor features and the notebook aims to point out issues with the original analysis, not to identify injury predictors.
# 
# Columns of importance here are *shoulder*, *knees*, *back*, *wrist* and *hips* as these are 0-1 categorical variables indicating whether the subject was injured in the described location. These serve as **targets** for implemented classifiers.
# 
# Let's add a column describing injury occuring in any location at all, even though this has not been done in the original study.

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-04-01T18:33:32.983658Z","iopub.execute_input":"2024-04-01T18:33:32.983925Z","iopub.status.idle":"2024-04-01T18:33:32.993970Z","shell.execute_reply.started":"2024-04-01T18:33:32.983903Z","shell.execute_reply":"2024-04-01T18:33:32.993118Z"}}
# add agregated injury column
data['injury'] = ((data['shoulder'] == 1) ^ (data['knees'] == 1) ^
                  (data['back'] == 1) ^ (data['wrist'] == 1) ^
                  (data['hips'] == 1)).astype(int)

# %% [markdown]
# Let's create list of these injury locations for convenience and look at some statistics.

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-04-01T18:33:32.995007Z","iopub.execute_input":"2024-04-01T18:33:32.995239Z","iopub.status.idle":"2024-04-01T18:33:33.026515Z","shell.execute_reply.started":"2024-04-01T18:33:32.995219Z","shell.execute_reply":"2024-04-01T18:33:33.025673Z"}}
inj_loc = ['back', 'knees', 'shoulder', 'wrist', 'hips', 'injury']
# descriptives of the data – returns: N, mean, std, min, 25%, 50%, 75%, max
data[inj_loc].describe()

# %% [markdown]
# Compare the means (corresponding to frequency of injury occurence) to the accuracies reported by classifiers from the original article:

# %% [code] {"_kg_hide-input":true,"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-04-01T18:33:33.027633Z","iopub.execute_input":"2024-04-01T18:33:33.027899Z","iopub.status.idle":"2024-04-01T18:33:33.038085Z","shell.execute_reply.started":"2024-04-01T18:33:33.027878Z","shell.execute_reply":"2024-04-01T18:33:33.036772Z"}}
Image('/kaggle/input/weightlifting-injuries-in-master-athletes/'
      'Table 2 orig article.png')

# %% [markdown]
# Can you see the issue here?
# 
# ## So what's the issue with the original article?
# 
# Let's visualize the accuracies reported by the original article together with the distribution of the data.
# 
# Reminder: *Injury* was not part of the original study, hence no accuracies are reported.

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-04-01T18:33:33.039573Z","iopub.execute_input":"2024-04-01T18:33:33.039833Z","iopub.status.idle":"2024-04-01T18:33:34.426185Z","shell.execute_reply.started":"2024-04-01T18:33:33.039812Z","shell.execute_reply":"2024-04-01T18:33:34.425292Z"}}
# setting up the accuracies from the original study
orig_accur = {'back': [.773, .768, .768, .748],
              'knees': [.727, .742, .742, .713],
              'shoulder': [.644, .619, .619, .683],
              'wrist': [.774, .790, .790, .744],
              'hips': [.876, .876, .876, .901],
              'injury': []}
# setting up plot
fig, axs = plt.subplots(2, 3, figsize=(8, 5), dpi=150)
ax = axs.ravel()
for i, loc in enumerate(data[inj_loc]):
    # ploting frequency of the data
    sns.histplot(data, x=loc, stat='percent',  # data to display
                 discrete=True,  # adjusts bin-width
                 ax=ax[i])  # placement within subplots
    # plotting reported accuracies
    ax[i].hlines([k*100 for k in orig_accur[loc]],
                 -0.5, 1.5,
                 'r', linestyle=':', label='accuracy')
    ax[i].set_ylim(0, 100)

# visual adjustments
fig.legend(['Accuracies reported by the original article'],
           loc='upper center', bbox_to_anchor=(0.5, 1.08))
fig.tight_layout()
plt.show()

# cleaning up
plt.close(fig)
del loc, fig, ax, axs

# %% [markdown]
# **Accuracies reported by the original article correspond to the percentage of the majority category in the data. Why is this a problem?**
# 
# Imagine that you predict every single person as being negative regarding being prone ot injury. Accuracy of such predictions would be exactly the percentage of the negative category in the data (labeled 0 here) as the accuracy measures only correctly classified cases. It doesn't take into account mislabeled cases at all.
# 
# So even though you have performed some algorithms, which can be analysed for feature importance measuring how influential are individual features for the model predictions, the **model doesn't really predict anything**. Reporting such features and claiming they have some predictive power for the injury prediction is definitely far fetched.
# 
# Let's simulate the procedure of the original study and show that such classifiers do not have any reasonable predictive power whatsoever.

# %% [markdown]
# # Machine learning classifiers
# 
# Here we will replicate the procedure used in the original study, "training" various classifiying algorithms on the supplemented data with accuracies similar to the ones from the original study, while *looking at metrics that are actually relevant to this problem*. 
# 
# First we define structure of the further used neural network. This neural network is more complex than the one used in the study (yet we will see it yields similarly bad results).

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-04-01T18:33:34.427336Z","iopub.execute_input":"2024-04-01T18:33:34.427632Z","iopub.status.idle":"2024-04-01T18:33:34.434197Z","shell.execute_reply.started":"2024-04-01T18:33:34.427608Z","shell.execute_reply":"2024-04-01T18:33:34.433315Z"}}
class MyModel(nn.Module):
    """Feed-forward Neural Network used here."""

    def __init__(self, hid_feat, out_feat):
        super().__init__()
        # PyTorch 'nn.LazyLinear' layers can adapt the number of inputs to the
        # provided input tensor not knowing it's exact size (e.g. due to feature
        # selection in pipeline with grid/random search for good number of input
        # features).
        # They are currently under heavy development, so:
        warnings.filterwarnings('ignore', category=UserWarning, module='torch')
        self.layer1 = nn.LazyLinear(hid_feat)
        # normal linear hidden layer
        self.layer2 = nn.Linear(hid_feat, out_feat)
        self.drop = nn.Dropout(0.4)

    def forward(self, my_input):
        """
        Forward propagation of data in the NN.

        Args_
        input: The propagated data.

        Returns_
        Resulting output produced by the NN.
        """
        layer1a = torch.sigmoid(self.layer1(my_input))
        layer1a = self.drop(layer1a)
        layer2a = torch.sigmoid(self.layer2(layer1a))
        return layer2a

# %% [markdown]
# ## Simulating results of the original article.
# 
# Let's fit models they have used in the original article with similar procedure parameters:
# * using 80-20 train-test split (stratified),
# * using 10 fold (stratified) cross-validation,
# * searching for optimal hyper-parameters with Grid Search with accuracy as a target.
# 
# Since the original article is a bit foggy about the algortihms they've actually used (eg., referring to one only by 'ensemble methods'), we have decided to include following classifiers:
# * Dummy Classifier – doesn't actually classify, it is set up to classify everything as negative, to show that the other classfifiers do not behave well,
# * GradientBoostingClassifier,
# * RandomForestClassifier,
# * LogisticRegression, and
# * Forward-feed Neural Network.
# 
# Even though we are storing every trained model identified by GridSearchCV into a dictionary so they can be recalled later, we find it easier to simultaniously store their perfomance metrics into a Pandas dataframe. A plot of ROC and precision-recall curves of these classifiers is also created to inspect their parameters as classifiers later on.

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-04-01T18:33:34.435404Z","iopub.execute_input":"2024-04-01T18:33:34.435718Z","iopub.status.idle":"2024-04-01T18:36:10.038190Z","shell.execute_reply.started":"2024-04-01T18:33:34.435695Z","shell.execute_reply":"2024-04-01T18:36:10.037147Z"}}
# setting different plot theme
sns.set_theme(style='whitegrid', palette='bright')
# setting a random state to replicate randomised results
rnd = np.random.RandomState(10)

# creating resulting scores table to store all scores
cls_nm = ['dummy', 'GBC', 'RFC', 'Log', 'FNN']  # abbrev of classifiers
scores = ['accuracy', 'recall', 'precision', 'f1', 'AUC']
index = pd.MultiIndex.from_product([inj_loc, scores],
                                   names=['Injury location', 'Score'])
scores = pd.DataFrame(index=index,
                      dtype=float)
del index  # cleaning up

# set up for plots
# =============================================================================
fig = plt.figure(figsize=(10, 30), dpi=150)
subfigs = fig.subfigures(6, 1)
colors = ['r', 'g', 'b', 'orange', 'm']  # to ensure color consistency in plots
# set up to plot f1 score contours
array_0_1 = np.linspace(0.01, 1, 100)
f1_grid = np.empty((100, 100))
for i, a in enumerate(array_0_1):
    for ii, b in enumerate(array_0_1):
        f1_grid[i, ii] = 2*a*b/(a+b)
grid = np.meshgrid(array_0_1, array_0_1)
del array_0_1, i, a, ii, b

# actual models
# =============================================================================
# storage for all the best models
trained_best = {}

# folds for Cross Validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=rnd)

# define pipeline for GridSearch over different estimators
pipeline = pp.Pipeline([('scaler', MinMaxScaler()),
                        ('estim', LogisticRegression())])

# specify the neural network  as a sklearn estimator using Skorch library
# -----------------------------------------------------------------------
stopper = EarlyStopping()
net = NeuralNetBinaryClassifier(module=MyModel(hid_feat=1, out_feat=1),
                                optimizer=optim.Adam,
                                max_epochs=30,
                                device='cuda',
                                callbacks=[stopper],
                                verbose=0)
# parameter/estimator grid to search over
params = [
          # Dummy Classifier – assigns everything with 0
          {'estim': [DummyClassifier(strategy='most_frequent')]},
          # GBC
          {'estim': [GradientBoostingClassifier(max_features='sqrt',
                                                random_state=rnd)],
           'estim__learning_rate': [0.01, 0.1, 0.2, 0.5],
           'estim__n_estimators': [10, 20, 50],
           'estim__max_depth': [1, 2]
           },
          # Random Forest
          {'estim': [RandomForestClassifier(max_features='sqrt',
                                            random_state=rnd)],
           'estim__n_estimators': [30, 50, 100],
           'estim__max_depth': [2, 3]
           },
          # Logistic regression
          {'estim': [LogisticRegression(max_iter=1000,
                                        random_state=rnd)],
           'estim__C': [0.01, 0.1, 1, 10, 100]
           },
          # FNN
          {'estim': [net],
           'estim__module__hid_feat': range(500, 3001, 500),
           'estim__module__out_feat': [1]
           }
          ]

# set up work dataframe and selecting relevant data
df = data.copy().astype(np.float32)
X = df.copy()
X = X.drop(inj_loc, axis='columns')

# loops through different locations of injuries
# =============================================================================
for i, loc in enumerate(inj_loc):
    print('\n\n\n',
          '================================================================\n',
          loc, '\n',
          '================================================================'
          '\n')
    # select response variable
    y = df[loc]

    # split into train-test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        stratify=y,
                                                        random_state=rnd)

    # create key to store best estimated models
    trained_best[loc] = []
    # initialise plot for given injury location
    axs = subfigs[i].subplots(1, 2)
    subfigs[i].suptitle(loc)

    # loops through different estimators in parameter/estimator grid
    # =========================================================================
    # NOTE: if I was looking for the best classifier, I would ommit this
    # loop and pass the whole params as a param_grid into GridSearchCV
    for ii, param_grid in enumerate(params):
        # set up grid-search cross validation, score for accuracy
        model = GridSearchCV(pipeline,
                             param_grid,
                             cv=skf,
                             scoring='accuracy',
                             n_jobs=-1,
                             refit=True,
                             verbose=1)

        # fit the model
        model.fit(X_train, y_train)

        # select, display and store the best model
        best_model = model.best_estimator_  # select
        print(best_model.named_steps['estim'], '\n')  # display
        trained_best[loc].append(best_model)  # store

        # model prediction for TEST sets
        # =================================================================
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        # model  scoring for TEST set
        # =====================================================================
        # accuracy
        acc = accuracy_score(y_test, y_pred)
        scores.loc[(loc, 'accuracy'), cls_nm[ii]] = acc
        # recall
        recall = recall_score(y_test, y_pred)
        scores.loc[(loc, 'recall'), cls_nm[ii]] = recall
        # precision
        precision = precision_score(y_test, y_pred)
        scores.loc[(loc, 'precision'), cls_nm[ii]] = precision
        # f1 score
        f1 = f1_score(y_test, y_pred)
        scores.loc[(loc, 'f1'), cls_nm[ii]] = f1
        # compute AUC
        auc = roc_auc_score(y_test, y_prob)
        scores.loc[(loc, 'AUC'), cls_nm[ii]] = auc

        # plotting ROC curves
        # =====================================================================
        # calculating FPR and TPR for each threshold level
        fpr, tpr, thresh = roc_curve(y_test, y_prob)
        # plot ROC curve
        axs[0].plot(fpr, tpr,
                    label=f'{cls_nm[ii]}',
                    color=colors[ii])
        # find index of the threshold closest to 0.5
        thr_idx = np.argmin(np.abs(thresh - 0.5))
        # plot the defauls decision point onto ROC curve
        axs[0].plot(fpr[thr_idx], tpr[thr_idx],
                    color=colors[ii], marker='o',
                    zorder=10)
        # subplots visual adjustments
        axs[0].set_title('ROC curve')
        axs[0].set_xlabel('FPR')
        axs[0].set_ylabel('TPR')
        axs[0].legend(loc='lower right')

        # plotting precision-recall curves
        # =====================================================================
        # plotting f1 contours
        f1_score_plot = axs[1].contour(grid[0], grid[1], f1_grid,
                                       levels=np.arange(0, 1, 0.1),
                                       linestyles='dotted', colors='k')
        axs[1].clabel(f1_score_plot, inline=True, fontsize=8)

        # calculating precision and recall for each threshold level
        prec, rec, thresh = precision_recall_curve(y_test, y_prob)
        # plot ROC curve
        axs[1].plot(rec, prec,
                    label=f'{cls_nm[ii]}',
                    color=colors[ii])
        # find index of the threshold closest to 0.5
        thr_idx = np.argmin(np.abs(thresh - 0.5))
        # plot the defauls decision point onto ROC curve
        axs[1].plot(rec[thr_idx], prec[thr_idx],
                    color=colors[ii], marker='o',
                    zorder=10)
        # subplots visual adjustments
        axs[1].set_title('precision-recall')
        axs[1].set_xlabel('Recall')
        axs[1].set_ylabel('Precision')
        axs[1].legend(loc='upper right')

    for ax in axs.flat:
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

# save plot
plt.savefig('ROC unequal train.png', bbox_inches='tight')
plt.close(fig)

# cleaining up
del acc, auc, ax, axs, best_model, loc, params, skf

# extract accuracies from scores
accuracies = scores.drop(['recall', 'precision', 'f1', 'AUC'],
                         axis='index', level='Score').copy()
accuracies = accuracies.droplevel('Score', axis='index')

# %% [markdown]
# Even the fact that attempts to calculate precision result in an error suggest that these classifiers do not really classify at all, and simply assign everything with a negative label. But let us compare accuracies of our classifiers with the accuracies from the original study.
# 
# Let's create heatmaps for both, so we can do quick visual check on how we are doing.

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-04-01T18:36:10.039781Z","iopub.execute_input":"2024-04-01T18:36:10.040614Z","iopub.status.idle":"2024-04-01T18:36:10.818853Z","shell.execute_reply.started":"2024-04-01T18:36:10.040578Z","shell.execute_reply":"2024-04-01T18:36:10.818015Z"}}
# accuracies from original article
orig_accur['injury'] = ['nan' for k in range(4)]
orig_acc = pd.DataFrame(orig_accur,
                        index=['Ensemble', 'RFC', 'Log',
                               'RFC\nexpert'])
orig_acc.replace('nan', np.nan, inplace=True)
orig_acc = orig_acc.transpose()

fig, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=150)
# accuracy heatmaps
axs[0].set_title('Accuracies of our models')
sns.heatmap(accuracies,
            cmap='inferno', cbar=False, vmin=0.5, vmax=1.0,
            square=False, annot=True, fmt='.3f',
            ax=axs[0])
axs[1].set_title('Accuracies from the original article')
sns.heatmap(orig_acc,
            cmap='inferno', cbar=True, vmin=0.5, vmax=1.0,
            square=False, annot=True, fmt='.3f',
            ax=axs[1])
plt.show()
plt.close()

# %% [markdown]
# **So the accuracies of our classifiers are very comparable to the classifiers of the original study.**
# 
# To be fair here – it looks like are classifiers are really defaulting to assigning everything with the negative label, while their classifiers attempted to classify at least some cases. Yet the accuracy they've obtained doesn't actually provide any information about their performance – compare it to the *dummy classifier* that actually classifies every case with negative label. Any minor increase over this can be attributed purely to the luck with train-test split selection and cannot be called to be reasonably generalizable.
# 
# Let's point out that the original study includes features (specifically information about selected chronical illnesses of the respondents), that are not provided in the published dataset. Should these informations be provided, it is possible that the algorithms trained here would perform a better job at injury classification.
# 
# Let's look at the various metrics for our classifiers that are actually relevant to this case.

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-04-01T18:36:10.820090Z","iopub.execute_input":"2024-04-01T18:36:10.820386Z","iopub.status.idle":"2024-04-01T18:36:10.845904Z","shell.execute_reply.started":"2024-04-01T18:36:10.820361Z","shell.execute_reply":"2024-04-01T18:36:10.844982Z"}}
scores.round(3)

# %% [markdown]
# Simple look at precison and recall obtained from these classifiers shows that the vast majority of them just labels everything with a negative label. What is these classifiers are trained reasonably, only the default decision threshold of 0.5 does not work well for their predictions? Let's inspect the ROC and precision-recall curves we have created earlier.

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-04-01T18:36:10.849472Z","iopub.execute_input":"2024-04-01T18:36:10.849764Z","iopub.status.idle":"2024-04-01T18:36:10.873274Z","shell.execute_reply.started":"2024-04-01T18:36:10.849739Z","shell.execute_reply":"2024-04-01T18:36:10.872419Z"}}
Image('ROC unequal train.png')

# %% [markdown]
# **Inspecting these ROC curves shows one thing – these models do not reasonably work as classifiers no matter what decision threshold we would pick.**
# 
# # Conclusion
# 
# We have not discovered America here, the fact that choosing the right metric when dealing with imbalanced data is paramount is well known. But there are several things that are astounding here:
# * The accuracies of these classifiers for injuries of different bodyparts are very obviously corresponding to the majority category. It is very interesting that none of the authors noticed this.
# * The authors extracted feature importances and interpreted them as the strong predictors of injury occurence. Mathematicall speaking, the models were "trained", and feature importances do exist and can be extracted. The issue here is that the overall model does not really predict anything, so the feature importance is irrelevant.
# * First author is a **Director**  of **Center for Statistical Training and Consulting** at **Michigan State University**. Shouldn't such a person know better, and spot these errors from afar?
# * This article passed a peer review in a Q1 journal.
# 
# What to take from this? **Do not trust something just because it is signed by a real professional, published in a good journal, and contains the magic words *'machine learning'*.**
# 
# ## Can we do better?
# 
# Let us address the data imbalance issue by random undersampling of the majority category. We are going to do that by hand, even though we could use e.g. sklearn's RandomUnderSampler. The held-out test set has the same category ratio as the original dataset.
# 
# **Result:** This does not lead to interesting results. The resulting classifiers assign labels basically at random (achieving accuracies around 50 %). Implementing oversampling of the minority category might be an alternative approach (e.g. SMOTE), but there is also a possibility that the collected data simply does not contain information that would allow for reasonable injury classification.

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-04-01T18:49:56.868055Z","iopub.execute_input":"2024-04-01T18:49:56.868415Z","iopub.status.idle":"2024-04-01T18:51:32.629537Z","shell.execute_reply.started":"2024-04-01T18:49:56.868389Z","shell.execute_reply":"2024-04-01T18:51:32.628590Z"}}
# setting different plot theme
sns.set_theme(style='whitegrid', palette='bright')
# setting a random state to replicate randomised results
rnd = np.random.RandomState(10)

# creating resulting scores table to store all scores
cls_nm = ['dummy', 'GBC', 'RFC', 'Log', 'FNN']  # abbrev of classifiers
scores = ['accuracy', 'recall', 'precision', 'f1', 'AUC']
index = pd.MultiIndex.from_product([inj_loc, scores],
                                   names=['Injury location', 'Score'])
scores = pd.DataFrame(index=index,
                      dtype=float)
del index  # cleaning up

# set up for plots
# =============================================================================
fig = plt.figure(figsize=(10, 30), dpi=150)
subfigs = fig.subfigures(6, 1)
colors = ['r', 'g', 'b', 'orange', 'm']  # to ensure color consistency in plots
# set up to plot f1 score contours
array_0_1 = np.linspace(0.01, 1, 100)
f1_grid = np.empty((100, 100))
for i, a in enumerate(array_0_1):
    for ii, b in enumerate(array_0_1):
        f1_grid[i, ii] = 2*a*b/(a+b)
grid = np.meshgrid(array_0_1, array_0_1)
del array_0_1, i, a, ii, b

# run setups
# =============================================================================
# storage for all the best models
trained_best = {}

# folds for Cross Validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=rnd)

# define pipeline for GridSearch over different estimators
pipeline = pp.Pipeline([('scaler', MinMaxScaler()),
              #          ('engineer', PolynomialFeatures(degree=2)),
               #         ('selector', SelectPercentile(chi2, percentile=10)),
                        ('estim', LogisticRegression())])

# specify the neural network  as a sklearn estimator using Skorch library
# -----------------------------------------------------------------------
stopper = EarlyStopping()
net = NeuralNetBinaryClassifier(module=MyModel(hid_feat=1, out_feat=1),
                                optimizer=optim.Adam,
                                max_epochs=30,
                                device='cuda',
                                callbacks=[stopper],
                                verbose=0)
# parameter/estimator grid to search over
params = [
          # Dummy Classifier – assigns everything with 0
          {'estim': [DummyClassifier(strategy='most_frequent')]},
          # GBC
          {'estim': [GradientBoostingClassifier(max_features='sqrt',
                                                random_state=rnd)],
           'estim__learning_rate': [0.01, 0.1, 0.2, 0.5],
           'estim__n_estimators': [10, 20, 50],
           'estim__max_depth': [1, 2]
           },
          # Random Forest
          {'estim': [RandomForestClassifier(max_features='sqrt',
                                            random_state=rnd)],
           'estim__n_estimators': [30, 50, 100],
           'estim__max_depth': [2, 3]
           },
          # Logistic regression
          {'estim': [LogisticRegression(max_iter=1000,
                                        random_state=rnd)],
           'estim__C': [0.01, 0.1, 1, 10, 100]
           },
          # FNN
          {'estim': [net],
           'estim__module__hid_feat': range(500, 3001, 500),
           'estim__module__out_feat': [1]
           }
          ]

# set up work data frame
df = data.copy().astype(np.float32)

# loops through different locations of injuries
# =============================================================================
for i, loc in enumerate(inj_loc):
    print('\n\n\n',
          '================================================================\n',
          loc, '\n',
          '================================================================'
          '\n')
    # split original data into train-test to preserve distribution of classes
    # in test set
    df_train, df_test = train_test_split(df, train_size=0.8,
                                         stratify=data[loc],
                                         random_state=rnd)
    # test set can be split right away:
    X_test = df_test.drop(inj_loc, axis='columns').copy()
    y_test = df_test[loc].copy()

    # split the train data into positive and negative labels
    df_train_pos = df_train[df_train[loc] == 1]
    # take only the negative cases and create sample with the same size as the
    # positive dataset
    df_train_neg = df_train[df_train[loc] == 0].sample(n=df_train_pos.shape[0],
                                                       random_state=rnd)
    # join the positive and negative datasets and shuffle them
    df_train = pd.concat([df_train_pos, df_train_neg],
                         axis='index').sample(frac=1.0, random_state=rnd)
    # we can now separate target and predictors
    X_train = df_train.drop(inj_loc, axis='columns').copy()
    y_train = df_train[loc].copy()

    # create key to store best estimated models
    trained_best[loc] = []
    # initialise plot for given injury location
    axs = subfigs[i].subplots(1, 2)
    subfigs[i].suptitle(loc)

    # loops through different estimators in parameter/estimator grid
    # =========================================================================
    # NOTE: if I was looking for the best classifier, I would ommit this
    # loop and pass the whole params as a param_grid into GridSearchCV
    for ii, param_grid in enumerate(params):
        # set up grid-search cross validation, score for recall
        model = GridSearchCV(pipeline,
                             param_grid,
                             cv=skf,
                             scoring='recall',
                             n_jobs=-1,
                             refit=True,
                             verbose=1)

        # fit the model
        model.fit(X_train, y_train)

        # select, display and store the best model
        best_model = model.best_estimator_  # select
        print(best_model.named_steps['estim'])  # display
        trained_best[loc].append(best_model)  # store

        # model prediction for TEST sets
        # =====================================================================
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        # model  scoring for TEST set
        # =====================================================================
        # accuracy
        acc = accuracy_score(y_test, y_pred)
        scores.loc[(loc, 'accuracy'), cls_nm[ii]] = acc
        # recall
        recall = recall_score(y_test, y_pred)
        scores.loc[(loc, 'recall'), cls_nm[ii]] = recall
        # precision
        precision = precision_score(y_test, y_pred)
        scores.loc[(loc, 'precision'), cls_nm[ii]] = precision
        # f1 score
        f1 = f1_score(y_test, y_pred)
        scores.loc[(loc, 'f1'), cls_nm[ii]] = f1
        # compute AUC
        auc = roc_auc_score(y_test, y_prob)
        scores.loc[(loc, 'AUC'), cls_nm[ii]] = auc

        # plotting ROC curves
        # =====================================================================
        # calculating FPR and TPR for each threshold level
        fpr, tpr, thresh = roc_curve(y_test, y_prob)
        # plot ROC curve
        axs[0].plot(fpr, tpr,
                    label=f'{cls_nm[ii]}',
                    color=colors[ii])
        # find index of the threshold closest to 0.5
        thr_idx = np.argmin(np.abs(thresh - 0.5))
        # plot the defauls decision point onto ROC curve
        axs[0].plot(fpr[thr_idx], tpr[thr_idx],
                    color=colors[ii], marker='o',
                    zorder=10)
        # subplots visual adjustments
        axs[0].set_title('ROC curve')
        axs[0].set_xlabel('FPR')
        axs[0].set_ylabel('TPR')
        axs[0].legend(loc='lower right')

        # plotting precision-recall curves
        # =====================================================================
        # plotting f1 contours
        f1_score_plot = axs[1].contour(grid[0], grid[1], f1_grid,
                                       levels=np.arange(0, 1, 0.1),
                                       linestyles='dotted', colors='k')
        axs[1].clabel(f1_score_plot, inline=True, fontsize=8)

        # calculating precision and recall for each threshold level
        prec, rec, thresh = precision_recall_curve(y_test, y_prob)
        # plot ROC curve
        axs[1].plot(rec, prec,
                    label=f'{cls_nm[ii]}',
                    color=colors[ii])
        # find index of the threshold closest to 0.5
        thr_idx = np.argmin(np.abs(thresh - 0.5))
        # plot the defauls decision point onto ROC curve
        axs[1].plot(rec[thr_idx], prec[thr_idx],
                    color=colors[ii], marker='o',
                    zorder=10)
        # subplots visual adjustments
        axs[1].set_title('precision-recall')
        axs[1].set_xlabel('Recall')
        axs[1].set_ylabel('Precision')
        axs[1].legend(loc='upper right')

    for ax in axs.flat:
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

# save plot
plt.savefig('ROC unequal train.png', bbox_inches='tight')
plt.show()

# cleaining up
del acc, auc, ax, axs, best_model, loc, params

scores.round(3)