import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('Sanskar.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
sc = StandardScaler()
X_train = sc.fit_transform(X)

app = Flask(__name__)
mp = pickle.load(open('model_pickle.pkl', 'rb'))


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def stock_analysis():
    df = pd.read_csv('Sanskar.csv')
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    sc = StandardScaler()
    X_train = sc.fit_transform(X)

    features = [x for x in request.form.values()]
    final_features = [np.array(features)]
    inp = sc.transform(final_features)
    
    output = mp.predict(inp)
    if output == -1:
        output = "LOSS"
    else:
        output = "PROFIT"

    return render_template('index.html', prediction_text='  {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
