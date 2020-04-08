from flask import Flask,render_template,session,redirect,url_for
from flask_wtf import FlaskForm
from wtforms import (StringField,BooleanField,SelectField,
                    SubmitField,RadioField,TextAreaField)
from wtforms.validators import DataRequired


class NewsForm(FlaskForm):
    text=TextAreaField("Please paste the news article below")
    submit=SubmitField("Submit")


class FaceForm(FlaskForm):
    pic = StringField("Please enter a url of an image file.",validators=[])
    # img_vid=RadioField("Select whether you uploaded an image or a video file",choices=[("img","Image file"),("vid","Video file")])
    submit=SubmitField('Submit')

class DictForm(FlaskForm):
    word=TextAreaField("Please enter a word below")
    submit=SubmitField("Submit")
