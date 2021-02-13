from wtforms import TextAreaField, SubmitField
from wtforms.validators import DataRequired
from flask_wtf import FlaskForm


class TextForm(FlaskForm):
    text = TextAreaField('Цель', validators=[DataRequired()])
    submit = SubmitField('Отправить')
