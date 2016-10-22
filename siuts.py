from flask import Flask
from bird import Bird
import jsonpickle
import json

app = Flask(__name__)
birds = []
with open('birds.json') as data_file:
    data = json.load(data_file)

for b in data['birds']:
    birds.append(Bird(b['name'],b['name_la'],b['name_en'],b['name_ee'],b['description']))

@app.route('/classify_bird', methods=['POST'])
def get_matching_birds():
    return jsonify({'birds': birds})

@app.route('/birds', methods=['GET'])
def get_all_birds():
    return jsonpickle.encode({'birds': birds}, unpicklable=False)

@app.route('/birds/<string:bird_name>', methods=['GET'])
def get_bird_by_name(bird_name):
    bird = next((b for b in birds if b.name == bird_name), {})
    return jsonpickle.encode(bird, unpicklable=False)

if __name__ == '__main__':
    app.run()
