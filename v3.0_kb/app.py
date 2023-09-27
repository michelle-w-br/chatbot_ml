from flask import Flask, render_template, request, jsonify
from qa_llamaindex import get_response

app = Flask(__name__)

@app.get("/")
def index_get():
    return render_template("home.html")

@app.post("/predict")
def predict():
    text=request.get_json().get("message")
    response=get_response(text)
    
    print(response.response)
    message={"answer":response.response}
    print(message)
    return jsonify(message)

if __name__=="__main__":
    app.run(debug=True)