from flask import Flask

app = Flask(__name__)

# Deafult API route
@app.route("/")
def homepage():
    return "Hello World"

if __name__ == "__main__":
    app.run(host='localhost', port=9874)