import os


class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'university2035'
    QLALCHEMY_TRACK_MODIFICATIONS = False
