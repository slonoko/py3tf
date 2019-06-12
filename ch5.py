#%% IMDB movie dataset

import pyprind
import pandas as pd
import os

# change the `basepath` to the directory of the
# unzipped movie dataset

basepath = '/home/elie/Workspace/data/aclImdb'

labels = {'pos': 1, 'neg': 0}
pbar = pyprind.ProgBar(50000)
df = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join(basepath, s, l)
        for file in sorted(os.listdir(path)):
            with open(os.path.join(path, file),
                      'r', encoding='utf-8') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]],
                           ignore_index=True)
            pbar.update()
df.columns = ['review', 'sentiment']

#%% Dumping into csv
import numpy as np

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('movie_data.csv', index=False, encoding='utf-8')

df = pd.read_csv('movie_data.csv', encoding='utf-8')
df.head(3)

#%% Displaying
df.shape

#%%
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer()
docs = np.array([
       'The sun is shining',
       'The weather is sweet',
               'The sun is shining, the weather is sweet, '
        'and one and one is two'])
bag = count.fit_transform(docs)

#%%
print(count.vocabulary_)

#%%
print(bag.toarray())

#%%
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer(use_idf=True,
                         norm='l2',
                         smooth_idf=True)
np.set_printoptions(precision=2)
print(tfidf.fit_transform(count.fit_transform(docs))
      .toarray())

#%%
df.loc[0, 'review'][-50:]

#%%
import re
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text


#%%
preprocessor(df.loc[0, 'review'][-50:])


#%%

preprocessor("</a>This :) is :( a test :-)!")


#%%

df['review'] = df['review'].apply(preprocessor)

#%%
def tokenizer(text):
   return text.split()
tokenizer('runners like running and thus they run')

#%%
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
   return [porter.stem(word) for word in text.split()]

tokenizer_porter('runners like running and thus they run')

#%%
import nltk
nltk.download('stopwords')

#%%
from nltk.corpus import stopwords

stop = stopwords.words('english')
[w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:] if w not in stop]

#%%
X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

#%%
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)
param_grid = [{'vect__ngram_range': [(1,1)],
              'vect__stop_words': [stop, None],
              'vect__tokenizer': [tokenizer,
                                  tokenizer_porter],
              'clf__penalty': ['l1', 'l2'],
              'clf__C': [1.0, 10.0, 100.0]},
            {'vect__ngram_range': [(1,1)],
              'vect__stop_words': [stop, None],
              'vect__tokenizer': [tokenizer,
                                  tokenizer_porter],
              'vect__use_idf':[False],
              'vect__norm':[None],
              'clf__penalty': ['l1', 'l2'],
              'clf__C': [1.0, 10.0, 100.0]}
            ]
lr_tfidf = Pipeline([('vect', tfidf),
                    ('clf',
                     LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr'))])
gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                          scoring='accuracy',
                          cv=5, verbose=1,
                          n_jobs=-1)
gs_lr_tfidf.fit(X_train, y_train)

#%%
print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f'
       % gs_lr_tfidf.best_score_)

clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f'
    % clf.score(X_test, y_test))