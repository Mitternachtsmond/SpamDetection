import re
from flask import Flask,render_template,request
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer as tf
import pickle
import nltk
nltk.download('punkt')



app = Flask(__name__)
loaded_model = pickle.load(open('SVM.sav', 'rb'))
fin = pickle.load(open('vectorizer.pk', 'rb'))
@app.route('/', methods = ['GET', 'POST'])
def index():
    if(request.method == 'GET'):
        return render_template('index.html')
    else:
        text = request.form.get('text')
        X = fin.transform([text])
        boole = loaded_model.predict(X)[0]
        if boole == 0:
            result='not spam'
        else:
            result='spam'
        return render_template ('result.html', text= text, result = result )



if __name__ == 'main':
    app.run(debug=True)
