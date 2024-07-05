from flask import Flask, render_template, request
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
import joblib

model = joblib.load("major_finalModel2.pkl")


app = Flask(__name__)


@app.route("/")
def Home():
    return render_template("index.html")


@app.route("/normal.html")
def nextPage():
    return render_template("normal.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        Fn = request.form["fname"]
        age = request.form["ag"]
        mof = "Male"
        getgen = request.form["ge"]
        if getgen[0] == "f" or getgen[0] == "F":
            mof = "Female"
        ctype = "non-anginal"
        chestpain = request.form["cp"]
        if chestpain[1] == "s" or chestpain[1] == "S":
            ctype = "asymptomatic"
        elif chestpain[0] == "a" or chestpain[0] == "A":
            ctype = "atypical angina"
        elif chestpain[0] == "t" or chestpain[0] == "T":
            ctype = "typical angina"

        trb = request.form["trest"]
        ch = request.form["chol"]
        bfb = 1
        fbs = request.form["fb"]
        if fbs[0] == "F" or fbs[0] == "f":
            bfb = 0
        re = 0
        res = request.form["rest"]
        if res[0] == "l" or res[0] == "L":
            re = 1
        elif res[0] == "s" or res[0] == "S":
            re = 2
        thalch = request.form["t"]
        exb = 1
        exhan = request.form["ex"]
        if exhan[0] == "F" or exhan[0] == "f":
            exb = 0
        dep = request.form["de"]
        slo = request.form["sl"]
        sa = 0
        if slo[0] == "u" or slo[0] == "U":
            sa = 1
        elif slo[0] == "d" or slo[0] == "D":
            sa = 2
        c = request.form["ca"]

        b = request.form["bm"]

        inputs = [
            [
                float(age),
                mof,
                ctype,
                float(trb),
                float(ch),
                bfb,
                float(re),
                float(thalch),
                exb,
                float(dep),
                float(sa),
                float(c),
                float(b),
            ]
        ]
        results = model.predict(inputs)
        print(results)
        if str(results[0]) == "0":
            return render_template("free.html", name=Fn)
        else:
            return render_template("results.html", name=Fn, stage=str(results[0]))
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
