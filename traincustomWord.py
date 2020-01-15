import tensorflow as tf
import numpy as np
import pandas as pd
import re
# import nltk
import time

# nltk.download('stopwords')
# stopwordList = nltk.corpus.stopwords.words('english')
TAG_RE = re.compile(r'<[^>]+>')
def remove_tags(text):
    return TAG_RE.sub('', text)
def preprocessing_text(text):
    # Remove html tag
    sentence = remove_tags(text)
    # Remove link
    sentence = re.sub(r'https:\/\/[a-zA-Z]*\.com',' ',sentence)
    # Remove number
    sentence = re.sub(r'\d+',' ',sentence)
    # Remove white space
    sentence = re.sub(r'\s+',' ',sentence)
    # Remove single character
    sentence = re.sub(r"\b[a-zA-Z]\b", ' ', sentence)
    # Remove bracket
    sentence = re.sub(r'\W+',' ',sentence)
    # Make sentence lowercase
    sentence = sentence.lower()
    # sentence = " ".join([word for word in sentence.split(' ') if word not in stopwordList])
    return sentence
print('Loading data')
rawData = pd.read_csv("IMDB_Dataset.csv")
print('Preprocessing data')
sentences = [preprocessing_text(inputText) for inputText in rawData['review']]

print('Generating vocab')
vocabWords = ["".join(sentence).split(' ') for sentence in sentences]
extendedVocabWords = []
for words in vocabWords:
    extendedVocabWords.extend(words)
vocab = sorted(set(extendedVocabWords))
vocab_size = len(vocab)
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

print('Splitting to datasets')
sentiment = np.array([1.0 if senti == "positive" else 0.0 for senti in rawData['sentiment']])
sentenceVectored = [[char2idx[word] for word in sentence.split(' ')] for sentence in sentences]
dataset = [(sentenceVectored[count], sentiment[count]) for count in range(len(sentiment))]

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 256))
# model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, stateful=False)))
# model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, stateful=False)))
# model.add(tf.keras.layers.LSTM(512, return_sequences=True, stateful=False))
# model.add(tf.keras.layers.Conv1D(128, 5, activation='relu'))
# model.add(tf.keras.layers.Conv1D(256, 5, activation='relu'))
# model.add(tf.keras.layers.GlobalMaxPooling1D())
# model.add(tf.keras.layers.Dropout(0.2)),
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary()

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# def loss(labels, logits):
#     return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
# model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

# model.fit(sentenceVectored, sentiment, epochs=5, batch_size=50, validation_split=0.1, validation_steps=20)

optimizer = tf.keras.optimizers.Adam()
def train_step(inp, target):
    with tf.GradientTape() as tape:
        predictions = model(np.array([inp]))
        loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(
                np.array([target]), predictions))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, 1.0 if (predictions.numpy() > 0.5 and target > 0.5) or (predictions.numpy() < 0.5 and target < 0.5) else 0.0

def train(epochs=10, stepsToOutput=100):
    totalSteps = len(dataset)
    for epoch in range(epochs):
        start = time.time()
    
        # initializing the hidden state at the start of every epoch
        # initally hidden is None
        hidden = model.reset_states()
        
        totalLoss = 0.0
        totalAccuracy = 0.0
        for stepNumber, (inp, target) in enumerate(dataset):
            loss, accuracy = train_step(inp, target)
            totalLoss += loss
            totalAccuracy += accuracy
            if (stepNumber+1) % stepsToOutput == 0:
                template = 'Epoch {}  Step {} out of {}  Average Loss {:.4f}  Average Accuracy {}'
                print(template.format(epoch+1, stepNumber+1, totalSteps, totalLoss/stepsToOutput, totalAccuracy/stepsToOutput))
                totalLoss = 0.0
                totalAccuracy = 0.0
        
        # saving (checkpoint) the model every 5 epochs
        # if (epoch + 1) % 5 == 0:
        #     model.save_weights(checkpoint_prefix.format(epoch=epoch))
        
        print ('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
        print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
def saveModel(modelPath='model.h5', variablePath='vocab.npy'):
    model.save(modelPath)
    np.save(variablePath, vocab)
train()