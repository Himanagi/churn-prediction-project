from flask import Flask, request, jsonify
import pandas as pd
from scripts.churn_utils import load_churn_model, preprocess_input

app = Flask(__name__)
model = load_churn_model()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df_input = pd.DataFrame([data])
    X = preprocess_input(df_input)
    prob = model.predict(X)[0][0]
    return jsonify({'churn_probability': float(prob)})

if __name__ == '__main__':
    app.run(debug=True)
