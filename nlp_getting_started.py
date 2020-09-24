import gc
import re
import pandas as pd
from tqdm import tqdm
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from spacy.lang.en import stop_words
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import tensorflow as tf
from tensorflow.python.keras.api.keras.models import Sequential
from tensorflow.python.keras.api.keras.preprocessing import text, sequence
from tensorflow.python.keras.api.keras.callbacks import EarlyStopping
from tensorflow.python.keras.api.keras.layers import Bidirectional, Dense, Dropout, Embedding, Flatten, GRU

gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

train_df = pd.read_csv('train.csv')

stop_spacy = stop_words.STOP_WORDS
stop_nltk = stopwords.words('english')
stop_sklearn = ENGLISH_STOP_WORDS

stop_words = set(stop_spacy)|set(stop_nltk)|set(stop_sklearn)

del stop_spacy, stop_nltk, stop_sklearn

def get_wordnet_pos(token):
    tag = nltk.pos_tag([token])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN, 
        "V": wordnet.VERB, 
        "R": wordnet.ADV
        }
    return tag_dict.get(tag, wordnet.NOUN)

lemma = WordNetLemmatizer()

def clean_text(doc):
    doc = re.sub(r'https?://\S+|www\.\S+', '', doc)
    doc = re.sub(r'<.*?>', '', doc)
    doc = re.sub(r'[^a-zA-Z0-9]+', ' ', doc)
    doc = re.sub(r'[0-9]', '', doc)
    doc = doc.lower()
    doc = nltk.word_tokenize(doc)
    doc = [token for token in doc if token not in stop_words]
    return " ".join([lemma.lemmatize(token, get_wordnet_pos(token)) for token in doc])

corpus = []
for doc in tqdm(train_df['text']):
    doc = clean_text(doc)
    corpus.append(doc)

tokenizer = text.Tokenizer(filters="")
tokenizer.fit_on_texts(corpus)
sequences = tokenizer.texts_to_sequences(corpus)
X_train_data = sequence.pad_sequences(sequences, padding="post")

max_length = X_train_data.shape[1]

y_train_data = train_df.iloc[:, 4].values

del train_df, corpus, sequences

X_train, X_val, y_train, y_val = train_test_split(X_train_data, y_train_data)

del X_train_data, y_train_data

model = Sequential()
model.add(Embedding(len(tokenizer.word_index)+1, 64, input_length=max_length))
model.add(Bidirectional(GRU(64, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

es = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min')

history = model.fit(X_train, y_train, batch_size=32, 
                    epochs=10, verbose = 1, callbacks = [es], validation_data=(X_val, y_val))

del X_train, X_val, y_train, y_val

test_df = pd.read_csv('test.csv')

corpus = []
for doc in tqdm(test_df['text']):
    doc = clean_text(doc)
    corpus.append(doc)

sequences = tokenizer.texts_to_sequences(corpus)
X_test = sequence.pad_sequences(sequences,maxlen=max_length, padding="post")

del test_df, corpus, sequences

y_test = model.predict(X_test, batch_size=32, verbose=1)

y_test = [1 if prob>0.5 else 0 for prob in y_test]

sample_submission = pd.read_csv('sample_submission.csv')

sample_submission['target'] = y_test
sample_submission.to_csv('submission.csv', index=False)

del X_test, y_test
gc.collect()

