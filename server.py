from flask import Flask, request, jsonify
from dexpression import Dexpression
from fastai.vision.all import load_learner
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
learner = load_learner('64pctmodel.pkl')
pathlib.PosixPath = temp

app = Flask(__name__)

# Default API route
@app.route("/", methods=['POST'])
def homepage():
    result = learner.predict(request.json['img_url'])
    return jsonify({'result': result[0]})

if __name__ == "__main__":
    app.run(host='localhost', port=9874, debug=True)