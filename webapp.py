#! /usr/bin/python3
from flask import Flask,render_template,session,redirect,url_for
from flask_wtf import FlaskForm
from forms import NewsForm, FaceForm, DictForm
from wtforms.validators import DataRequired
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
import string
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from skimage import io
import cv2
from difflib import get_close_matches
import json


app=Flask(__name__)
app.config['SECRET_KEY']="mykey"

def text_process(mess):
    stemmer= PorterStemmer()
    nopunc = [char for char in mess if char not in string.punctuation]

    nopunc = ''.join(nopunc)

    no_stop=[word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

    return [stemmer.stem(x) for x in no_stop]

def detect_and_blur_plate(file_location):
    casc= cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt2.xml")
    try:
        img = io.imread(file_location)
    except :
        print("could not find pic")
    name=str(str(file_location).split("/")[-1])    #file name

    try:
        no_image=False
        img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rect_measures=casc.detectMultiScale(img,minNeighbors=5)
        number=str(len(rect_measures))
        for x,y,w,h in rect_measures:

            to_blur=img[y:y+h,x:x+w]
            font=cv2.FONT_HERSHEY_COMPLEX
            blurred=cv2.medianBlur(to_blur,51)
            try:
                img[y:y+h,x:x+w]=blurred
                cv2.putText(img,"Face",(img.shape[0],0),font,1,[255,0,0],1,cv2.LINE_AA)
            except UnboundLocalError:
                print('Sorry could not find a face in the image')
        path="static/"+name
        try:
            cv2.imwrite(path, img)
        except cv2.error:
            name='error'

    except UnboundLocalError:
        name='error'
        print("\n\n\n\n\n True\n\n\n\n\n\n\n")

    return name

@app.route("/face_blur",methods=["GET","POST"])
def face_blur():
    name=None
    form=FaceForm()
    y = None

    if form.validate_on_submit():
        y=None
        img=form.pic.data
        session["submit"] = form.submit.data

        y = str(detect_and_blur_plate(str(img)))

        session["name"] = ("./static/" + str(detect_and_blur_plate(str(img))))

        return redirect(url_for("thankyou"))

    return render_template("face_blur.html",form=form,name=name, y=y)


@app.route("/dict",methods=["GET","POST"])
def dict():
    form=DictForm()
    meaning=False
    no_meaning=False
    y=False
    other_word=False

    if form.validate_on_submit():

        word= form.word.data
        # session["submit"] = form.submit.data
        form.word.data=""

        data=json.load(open('./static/076 data.json'))
        word = word.lower().strip()

        if word in data:
            meaning = data[word]
        else:
            other_word = get_close_matches(word,data.keys(),n=1)
            if other_word == []:
                no_meaning=True
                meaning=True

            else:
                y = True
                no_meaning = True
                other_word = other_word[0]

    return render_template("dict.html", form=form, meaning=meaning, other_word=other_word, no_meaning=no_meaning, y=y)



@app.route("/news_form",methods=("POST","GET"))
def news_app():

    form=NewsForm()
    model=joblib.load("static/news_classifier.joblib")

    if form.validate_on_submit():
        news=[(form.text.data)]
        y=str(model.predict(news)[0])
        session["y"] = str(model.predict(news)[0])
        session["proba_fake"] = str(model.predict_proba(news)[0][0])
        session["proba_real"] = str(model.predict_proba(news)[0][1])

        if y == '1':
            session["y"] = "The news article is most probably showing real news"
        elif y == "0":
            session["y"]= "The news article is most probably False"

        else:
            session["y"]= "Sorry we could not get an answer"

        return redirect(url_for("news_prediction"))

    return render_template("news_form.html",form=form)

@app.route("/news_prediction")
def news_prediction():
    return render_template("news_prediction.html")


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/thankyou")
def thankyou():
    return render_template("thankyou.html")

@app.route("/spam_filter")
def spam_filter():
    return render_template("spam_filter.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")


if __name__=="__main__":
    app.run()
