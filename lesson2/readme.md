`
git clone https://github.com/eleijonmarck/image-preprocessing.git
virtualenv env -p python3
source env/bin/activate
pip install -r image-preprocessing/cat_dog/requirements.txt 
export FLASK_APP=image-preprocessing/cat_dog/application.py
flask run --host=0.0.0.0
nohup flask run --host=0.0.0.0 & `
