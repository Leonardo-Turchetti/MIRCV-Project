from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import utils
import base64
from point import Point
import io
import pickle
from VPT import VPTree
from point import Point
from flask_ngrok import run_with_ngrok
import tensorflow as tf
from tensorflow import keras

PATH_HOME = '/content/drive/Shareddrives/MIRCV - Tonelli,Turchetti,Sirigu,Campilongo/'
PATH_DATASET = '/content/drive/Shareddrives/MIRCV - Tonelli,Turchetti,Sirigu,Campilongo/dataset/preprocessed/'
PATH_DISTRACTOR = '/content/drive/Shareddrives/MIRCV - Tonelli,Turchetti,Sirigu,Campilongo/dataset/mirflickr25k/mirflickr/'
IMAGE_HEIGHT = 299
IMAGE_WIDTH = 299

points_ft = []
#load feature point already extracted with fine tuned model
with open('/content/drive/Shareddrives/MIRCV - Tonelli,Turchetti,Sirigu,Campilongo/Performance and Evaluations/punti/points_ft1.pkl', 'rb') as ft:
    points_ft = pickle.load(ft)
#VPTree construction
vpt_fine_tuned = VPTree(points_ft, 200, "manhattan")

points_base = []
#load feature point already extracted with base model
with open('/content/drive/Shareddrives/MIRCV - Tonelli,Turchetti,Sirigu,Campilongo/Performance and Evaluations/punti/points_base.pkl', 'rb') as b:
    points_base = pickle.load(b)
#VPTree construction
vpt_base = VPTree(points_base, 200, "manhattan")

app = Flask(__name__, template_folder='/content/drive/Shareddrives/MIRCV - Tonelli,Turchetti,Sirigu,Campilongo/webapp/templates')
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
run_with_ngrok(app)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/', methods = ['POST'])
def handlePost():
	img = Image.open(request.files["query"])
	option = request.form["choice"]
	mod = request.form["model"]
	query = []
	if mod == 'fine_tuned':
		query = create_query_ft(img)
	else:
		query = create_query_base(img)
	images = {}
	if option == 'knn':
		if mod == 'fine_tuned':
			print("Model fine tuned, KNN search K = ", request.form["k"])
			images["images"] = knn_s(vpt_fine_tuned, query, request.form["k"])
		else:
			print("Model base, KNN search K = ", request.form["k"])
			images["images"] = knn_s(vpt_base, query, request.form["k"])
	else:
		if mod == 'fine_tuned':
			print("Model fine tuned, Range search with range = ", request.form["r"])
			images["images"] = range_s(vpt_fine_tuned, query, request.form["r"])
		else:
			print("Model base, Range search with range = ", request.form["r"])
			images["images"] = range_s(vpt_base, query, request.form["r"])
	if len(images) == 0:
		return "Search error!"
	return jsonify(images), 200

def create_query_ft(img):
	#Img preprocess
	im = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), resample=2)
	im.convert('RGB')
	image_np = tf.keras.preprocessing.image.img_to_array(im)
	img_np = np.expand_dims(image_np, axis=0)
	#Load the model
	print("Loading Fine Tuned Model")
	model = tf.keras.models.load_model(PATH_HOME+"models/model_ft_dense.h5")
	#Extract img feature
	features = model.predict(img_np, verbose=0)
	point = Point(features, "", "")
	return point

def create_query_base(img):
	print("Loading Base Model")
	model_base = keras.applications.inception_v3.InceptionV3(
    weights="imagenet",
    include_top=False,
    pooling = "avg",
    input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
	#Img preprocess
	im = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), resample=2)
	im.convert('RGB')
	image_np = tf.keras.preprocessing.image.img_to_array(im)
	ima_np = tf.keras.applications.inception_v3.preprocess_input(image_np)
	img_np = np.expand_dims(ima_np, axis=0)
	#Extract img feature
	features = model_base.predict(img_np, verbose=0)
	point = Point(features, "", "")
	return point

# Manage pictures returns a b64encoded picture
def manage_picture(filename):
    im = Image.open(filename)
    # Resizing the images without loosing aspect ratio
    MAX_SIZE = (150, 150)
    im.thumbnail(MAX_SIZE, Image.ANTIALIAS)
    buffered = io.BytesIO()
    im.save(buffered, format="PNG")
    buffered.seek(0)
    img = base64.b64encode(buffered.getvalue())
    return img

def retrieve_img(response):
	imgs = []
	filepath = ''
	for i in response:
		#build the path to the img
		if i[0] == "noise":
			filepath = PATH_DISTRACTOR + i[1]
		else:
			filepath = PATH_DATASET + i[0] + "/" + i[1]
		img = manage_picture(filepath)
		obj = ((i[0]),img.decode())
		imgs.append(obj)
	return imgs


def knn_s(index, query, k):
	response = utils.knn_search(index, query, k)
	#retrieve the img from the label and img_id
	result = retrieve_img(response)
	return result

def range_s(index, query, parameter):
	response = utils.range_search(index, query, parameter)
	#retrieve the img from the label and img_id
	result = retrieve_img(response)
	return result

if __name__ == '__main__':
    app.run()