import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
import numpy as np
import sys
import re

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
    return sentence

inputSentence = sys.argv[1]

vocab = np.load('vocab.npy', allow_pickle=True)
char2idx = {u:i for i, u in enumerate(vocab)}

doExit = False
for word in preprocessing_text(inputSentence).split(' '):
    if word not in char2idx:
        print('The word {} is not in dictionary'.format(word))
        doExit = True
if doExit:
    sys.exit(1)

model = tf.keras.models.load_model('model.h5', compile=False)
prediction = model.predict(np.array([[char2idx[word] for word in preprocessing_text(inputSentence).split(" ")]]))[0][0]

if prediction > 0.5:
    print('The neural network thinks that the sentence is a positive review with confidence of', prediction)
else:
    print('The neural network thinks that the sentence is a negative review with confidence of', 1-prediction)