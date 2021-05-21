from flask import Flask, request
from flask_api import create_app
from pathlib import Path
import os

app = create_app()

if __name__ == '__main__':
    # print(app.url_map)
    app.run(debug=True)
