from config import Config
from flask import Flask


def create_app(config=Config):
    app = Flask(__name__)
    app.config.from_object(config)

    from . models_api import models

    app.register_blueprint(models, url_prefix='/style_model')

    return app

