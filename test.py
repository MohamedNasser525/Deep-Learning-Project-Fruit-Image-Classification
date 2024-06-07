#!pip install
import tensorflow as tf
from keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Embedding, Input, GlobalAveragePooling1D, Dense
from keras.models import Model
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

# Transformer Block
def transformer_block(inputs, embed_dim, num_heads, ff_dim, rate=0.1):
    att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    att = Dropout(rate)(att)
    att = LayerNormalization(epsilon=1e-6)(inputs + att)

    ffn_output = tf.keras.Sequential([
        Dense(ff_dim, activation="relu"),
        Dense(embed_dim),
    ])(att)
    ffn_output = Dropout(rate)(ffn_output)
    return LayerNormalization(epsilon=1e-6)(att + ffn_output)

# Token and Position Embedding
def token_and_position_embedding(x, maxlen, vocab_size, embed_dim):
    positions = tf.range(start=0, limit=maxlen, delta=1)
    positions = Embedding(input_dim=maxlen, output_dim=embed_dim)(positions)
    x = Embedding(input_dim=vocab_size, output_dim=embed_dim)(x)
    return x + positions

vocab_size = 50000
maxlen = 20

df = pd.read_csv('out.csv', encoding='utf-8')
x = df.drop('rating', axis=1)
y = df['rating']
df['rating'] = df['rating'].map({-1: 0, 0: 0, 1: 1})
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(train_df['review_description'])

x_train_sequences = tokenizer.texts_to_sequences(train_df['review_description'])
x_val_sequences = tokenizer.texts_to_sequences(val_df['review_description'])

x_train = pad_sequences(x_train_sequences, maxlen=maxlen, padding='post', truncating='post')
x_val = pad_sequences(x_val_sequences, maxlen=maxlen, padding='post', truncating='post')

y_train = train_df['rating']
y_val = val_df['rating']

embed_dim = 32
num_heads = 2
ff_dim = 32

inputs = Input(shape=(maxlen,))
x = token_and_position_embedding(inputs, maxlen, vocab_size, embed_dim)
x = transformer_block(x, embed_dim, num_heads, ff_dim)
x = GlobalAveragePooling1D()(x)
x = Dropout(0.1)(x)
x = Dense(20, activation="relu")(x)
x = Dropout(0.1)(x)
outputs = Dense(2, activation="softmax")(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(x_train, y_train,
                    batch_size=32, epochs=10,
                    validation_data=(x_val, y_val))

model.save_weights("predict_class.h5")
results = model.evaluate(x_val, y_val, verbose=2)