# app.py

from flask import Flask, render_template, request
from predict import predict_review, predict_csv
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    confidence = None
    summary = None

    # Single review prediction
    if request.method == "POST" and "review_text" in request.form:
        review = request.form["review_text"]
        sentiment, confidence = predict_review(review)

    # CSV upload prediction
    if request.method == "POST" and "file" in request.files:
        file = request.files["file"]
        if file.filename.endswith(".csv"):
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)
            _, summary = predict_csv(path)

    return render_template(
        "index.html",
        sentiment=sentiment,
        confidence=confidence,
        summary=summary
    )

if __name__ == "__main__":
    app.run(debug=True)
