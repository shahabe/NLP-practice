
# coding: utf-8

# In[3]:

from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import re, os
import numpy as np

import torch
import json 
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "bert-large-cased-whole-word-masking-finetuned-squad"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)



def text_to_wordlist(text, remove_stopwords=False, stem_words=False, lemmatize_words = False):
    text = re.sub(r"[^A-Za-z0-9]", " ", text)

    text = text.lower().split()

    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]

    text = " ".join(text)

    if lemmatize_words:
        text = text.split()
        lemm = WordNetLemmatizer()
        text = [lemm.lemmatize(word) for word in text]
        text = " ".join(text)

    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    return (text)





# App config.
DEBUG = False
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
class ReusableForm(Form):
    name = StringField('Name:', validators=[validators.required()])
 
 
@app.route("/", methods=['GET', 'POST'])
def query_db():
    form = ReusableForm(request.form)
 
    if request.method == 'POST':
        

 
        if form.validate():
            test_phrase=request.form['name']
            question=request.form['name2']


            QA_input = {'question': question, 'context': test_phrase}
                    
            res = nlp(QA_input)

            display_answer = str(res['answer'])
            display_score = str(res['score'])




            flash(test_phrase)
            flash(question)
            flash(display_answer)
            flash(display_score)
            
        else:
            flash('Error: All the form fields are required. ')
    return render_template('query_db4.html', form=form)



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5003)

# %%
