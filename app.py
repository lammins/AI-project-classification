import os
from flask import Flask, request, render_template
import numpy as np
from keras.models import load_model
import cv2
#import sys


app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static')

my_model = load_model("models/cat_dog_classifier.hdf5")

@app.route("/", methods=['GET', 'POST'])

def home():
    if request.method == "GET":
        return render_template("index.html")
    
    global my_model
    if request.method == 'POST':   
                f = request.files['file']
                image = cv2.imdecode(np.fromstring(f.read(), np.uint8), cv2.IMREAD_COLOR)    
                image = cv2.resize(image, dsize=(150, 150))
                image = np.expand_dims(image, axis=0)
                predict = my_model.predict(image)
                if predict  < 0.5 : 
                    output = "This is a cat"
                else: 
                    if predict >= 1:
                        output = "This is a dog"
                return output
    return "No file found"

if __name__ == '__main__':
   app.run(host='0.0.0.0',debug = True, port=8000)