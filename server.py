__author__ = 'tungtt'

from flask import Flask, request
import ner_vn


app = Flask('crf')

with open('crf.html', 'r', encoding='utf-8') as f:
	data = f.read()

@app.route('/',methods = ['GET'])
def homepage():
	return data


@app.route('/crf', methods=['POST'])
def process_request():
    data = request.get_data()
    crf = ner_vn.fit('crf02.pkl')
    return ner_vn.predict(crf, data)



if __name__ == '__main__':
    app.run(port=10000)