from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load('model.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    user_input = request.args.get('user_input', '')
    result = model.predict([[float(x) for x in user_input.split(',')]])
    return jsonify({'result': result.tolist()})

if __name__ == '__main__':
    app.run()