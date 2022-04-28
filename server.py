from flask import Flask, request, jsonify
from flask_cors import cross_origin
from dexpression import Dexpression
from fastai.vision.all import load_learner
import os
import pathlib

# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath
learner = load_learner('64pctmodel.pkl')
# pathlib.PosixPath = temp

app = Flask(__name__)

# Default API route
@app.route("/", methods=['POST'])
@cross_origin()
def homepage():

    print(request.files)
    f = request.files['file']
    filename = f.filename
    f.save(filename)
    f.close()

    result = learner.predict(filename)
    os.remove(filename)
    return jsonify({'result': result[0].upper()})

if __name__ == "__main__":
    app.run(host='localhost', port=9874, debug=True)