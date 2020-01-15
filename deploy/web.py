import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import sys
import re

from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib

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


vocab = np.load('vocab.npy', allow_pickle=True)
char2idx = {u:i for i, u in enumerate(vocab)}

model = tf.keras.models.load_model('model.h5', compile=False)
model.summary()


class mainRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Send response status code
        self.send_response(200)
        
        self.send_header('Content-type', 'text/html')
        self.end_headers()
 
        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        if 'phrase' not in parsed:
            self.wfile.write(bytes(
"""
<pre>
Welcome to movie sentiment analysis built by tensorflow written by Simon Cheng
To get started, simply put /?phrase=yourphrase and the response are as follows:
If keyword is not in wordlist: [1, word1, word2, word3...]
If success: [0, probability, 0 for negative 1 for positive]
</pre>
""", "utf8"))
        else:
            phrase = parsed['phrase'][0]
            print(phrase)
            output = ""
            
            listOfUnknowns = [1]
            for word in preprocessing_text(phrase).split(' '):
                if word not in char2idx:
                    listOfUnknowns.append(word)
            if len(listOfUnknowns) != 1:
                output = str(listOfUnknowns)
            else:
                result = [0]
                prediction = model.predict(np.array([[char2idx[word] for word in preprocessing_text(phrase).split(" ")]]))[0][0]
                if prediction > 0.5:
                    result.extend([prediction, 1])
                else:
                    result.extend([1-prediction, 0])
                output = str(result)
            self.wfile.write(bytes(output, "utf8"))
        return

port = int(os.environ.get('PORT'))

server_address = ('0.0.0.0', port)
httpd = HTTPServer(server_address, mainRequestHandler)
print('Started sentiment analysis server on port', port)
httpd.serve_forever()
 