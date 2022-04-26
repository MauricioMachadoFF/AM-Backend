from flask import Flask
from dexpression import Dexpression
from fastai.vision.all import load_learner
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
learner = load_learner('64pctmodel.pkl')
pathlib.PosixPath = temp

result = learner.predict('./teste.jpg')
app = Flask(__name__)

# Default API route
@app.route("/")
def homepage():
    return result[0]

if __name__ == "__main__":
    app.run(host='localhost', port=9874)