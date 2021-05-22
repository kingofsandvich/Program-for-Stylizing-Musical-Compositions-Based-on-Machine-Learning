## Web-service implementing symbolic music genre transfer with CycleGAN

This project is complete using further sources:
- https://github.com/sumuzhao/CycleGAN-Music-Style-Transfer
- https://github.com/sumuzhao/CycleGAN-Music-Style-Transfer-Refactorization
- https://salu133445.github.io/pypianoroll/index.html
- https://craffel.github.io/pretty-midi/
- https://flask.palletsprojects.com/en/1.1.x/
- https://github.com/helio-fm/helio-workstation

## Content

Project provides REST API for symbolic music style transfer and training pipeline to fit the model.

## Deployment
All commands are meant to be typed from "Program-for-Stylizing-Musical-Compositions-Based-on-Machine-Learning" directory.
To get all requirements type:
```
python -m venv project-env
project-env\Scripts\activate.bat
python -m pip install -r requirements.txt
```
Checkpoints for models can be obtained via link: https://drive.google.com/drive/folders/1Y19U_o2kNziEEXXp5Qr2SJpFnGfzA5Ha?usp=sharing

To run webserver type this to command prompt from the project directory:\
```
python flask_api\app.py
```
To run client unpack "Helio.zip" and run Helio.exe. 
If your server is not on "http://127.0.0.1:5000/" you must specify it in "host.txt".
For the stable operation of the program python 3.x and a PyQT5 library instalations are required. 
Stylisation options are available in selection menu in pattern editor.