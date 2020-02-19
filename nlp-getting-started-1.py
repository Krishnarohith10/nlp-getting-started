import re
import nltk
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_len = 512
vocab_size = 10000
embedding_dims = 256
batch_size = 32
epochs = 5

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

def clean_text(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9]+', ' ', text)
    text = re.sub(r'[0-9]', '', text)
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if((word.isalpha()==1) and (word not in set(nltk.corpus.stopwords.words('english'))))]
    text = ' '.join(text)
    return text

corpus = []
for text in tqdm(train_df['text']):
    text = clean_text(text)
    corpus.append(text)

tokenizer = Tokenizer(num_words=vocab_size)

tokenizer.fit_on_texts(corpus)
sequences = tokenizer.texts_to_sequences(corpus)
x_train_data = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

y_train_data = train_df.iloc[:, 4].values
x_train, x_val, y_train, y_val = train_test_split(x_train_data, y_train_data)

model = Sequential()
model.add(Embedding(vocab_size, embedding_dims, input_length = max_len))
model.add(Bidirectional(LSTM(256, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
model.add(LSTM(128, dropout=0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer = 'Adam', metrics=['accuracy'])

model.summary()

history = model.fit(x_train, y_train, batch_size=batch_size, 
                    epochs=epochs, verbose = 1, validation_data=(x_val, y_val))

model.save('nlp-getting-started-trained-model.h5')

range_epochs = range(1, epochs+1)

plt.plot(range_epochs, history.history['accuracy'], 'r')
plt.plot(range_epochs, history.history['val_accuracy'], 'b')
plt.title('Train and Validation Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(range_epochs, history.history['loss'], 'r')
plt.plot(range_epochs, history.history['val_loss'], 'b')
plt.title('Train and Validation Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

corpus=[]
for text in tqdm(test_df['text']):
    text = clean_text(text)
    corpus.append(text)

tokenizer.fit_on_texts(corpus)
sequences = tokenizer.texts_to_sequences(corpus)
x_test = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

y_test = model.predict(x_test, batch_size=32)

sample_sub = pd.read_csv('sample_submission.csv')

pred_new=[]
for pred in y_test:
    if pred>=0.5:
        pred_new.append(1)
    else:
        pred_new.append(0)

sample_sub['target'] = pred_new

sample_sub.to_csv('submission.csv', index=False)
