from flask_cors import CORS, cross_origin
from flask import Flask, jsonify, request

from classifier import *
# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.

@app.route("/predict", methods=['POST', 'GET'])
@cross_origin()
def predict_api():
	output_dict = brand_influencer_classifier(request.json)
	response = jsonify(output_dict)
	return response

if __name__ == "__main__":
	app.run()
