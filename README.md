# Program-for-Stylizing-Musical-Compositions-Based-on-Machine-Learning
Using CycleGAN for symbolic genre transfer.

# Web-service implementing symbolic music genre transfer with CycleGAN

This project is complete using further sources:
- https://github.com/sumuzhao/CycleGAN-Music-Style-Transfer
- https://github.com/sumuzhao/CycleGAN-Music-Style-Transfer-Refactorization
- https://salu133445.github.io/pypianoroll/index.html
- https://craffel.github.io/pretty-midi/
- https://flask.palletsprojects.com/en/1.1.x/

## Content

Project provides REST API for symbolic music style transfer and training pipeline to fit the model.

## Instructions
All commands are meant to be typed from "Program-for-Stylizing-Musical-Compositions-Based-on-Machine-Learning" directory.
To get all requirements type:
```
python -m venv project-env
project-env\Scripts\activate.bat
python -m pip install -r requirements.txt
```
Checkpoints for models can be obtained via link:

To run webserver type this to command prompt from the project directory:\
```
python flask_api\app.py
```
To run client unpack "Helio.zip" and run Helio.exe. 
For the stable operation of the program python 3.x and a PyQT5 library instalations are required. 
Stylisation options are available in selection menu in pattern editor.