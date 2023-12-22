from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import joblib
import re 
import string 


nlp = spacy.load("en_core_web_sm")
data_fake = pd.read_csv("D:/backup folder/Fake.csv")
data_true = pd.read_csv("D:/backup folder/True.csv")
data_fake["class"] = 0
data_true["class"] = 1

data_merge = pd.concat([data_fake, data_true], axis=0)
data = data_merge.drop(['subject'], axis=1)
data = data.sample(frac=1)
data.reset_index(inplace=True)
data.drop(['index', 'title'], axis=1, inplace=True)

x = data['text']
y = data['class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)
print(xv_train)

def word(text):
    text= text.lower()
    text=re.sub('\[.*?\]','', text)
    text=re.sub('\\W',' ',text)
    text=re.sub('https?://\S+|www\.\S+','',text)
    text=re.sub('<.*>+','',text)
    text=re.sub('[%s]' % re.escape(string.punctuation),'',text)
    text=re.sub('\n','',text)
    text=re.sub('\w*\d\w','',text)
    return text


def train_random_forest_classifier(x_train, y_train):
    rf = RandomForestClassifier(random_state=0)
    trained_rf_model = rf.fit(x_train, y_train)
    return trained_rf_model

rf_model = train_random_forest_classifier(xv_train, y_train)

joblib.dump(rf_model, 'trained_model.joblib')

def calculate_cosine_similarity(vec1, vec2):
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]

def preprocess_and_predict(rf_model, vectorization, text):
    user_text=word(text)
    doc = nlp(user_text)
    lemmatized_text = " ".join([f"{token.lemma_}_{token.pos_}_{token.vector}" for token in doc])

    avg_pos_vector = np.mean([token.vector for token in doc], axis=0)
    avg_pos_vector_str = " ".join(map(str, avg_pos_vector))
    avg_pos_doc = nlp(avg_pos_vector_str)

    combined_feature = f"{lemmatized_text} {calculate_cosine_similarity(doc.vector, avg_pos_doc.vector):.4f} User-provided analysis goes here."

    vectorized_input = vectorization.transform([combined_feature])

    print("Shape of vectorized input:", vectorized_input.shape)  

    prediction = rf_model.predict(vectorized_input)

    result = "Fake News" if prediction[0] == 1 else "True News"

    return result

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        user_input = request.form.get("user_input", "")

        loaded_model = joblib.load('trained_model.joblib')

        result = preprocess_and_predict(loaded_model, vectorization, user_input)

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
