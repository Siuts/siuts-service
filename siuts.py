from flask import Flask, request
from flask.ext.api import status
from flask.ext.redis import FlaskRedis
from bird import Bird
import jsonpickle
import json
import uuid

app = Flask(__name__)
redis_store = FlaskRedis(app)

birds = []
with open('birds.json') as data_file:
    data = json.load(data_file)

for b in data['birds']:
    birds.append(Bird(b['name'], b['name_la'], b['name_en'], b['name_ee'], b['description']))

@app.route('/classify', methods=['POST'])
def process_classification_request():
    audio_data = request.form['audio_data']
    if not audio_data:
        return 'Invalid request, audio data missing', status.HTTP_400_BAD_REQUEST
    request_id = str(uuid.uuid4())
    redis_store.set(request_id, None)
    classify(audio_data, request_id)
    return request_id

def classify(audio_data, request_id):
    ### Do magic here ###
    result = [{'name': 'phylloscopus_sibilatrix', 'match': 53.11}, {'name': 'parus_major', 'match': 72.15}]
    redis_store.set(request_id, jsonpickle.encode(result, unpicklable=False))

@app.route('/status/<string:request_id>', methods=['GET'])
def check_session_status(request_id):
    classifier_status = redis_store.get(request_id)
    if not classifier_status:
        return 'No status available yet', status.HTTP_204_NO_CONTENT
    redis_store.delete(request_id)
    return classifier_status

@app.route('/birds', methods=['GET'])
def get_all_birds():
    if not birds:
        return 'Something went horribly wrong', status.HTTP_500_INTERNAL_SERVER_ERROR
    return jsonpickle.encode({'birds': birds}, unpicklable=False)

@app.route('/birds/<string:bird_name>', methods=['GET'])
def get_bird_by_name(bird_name):
    bird = next((b for b in birds if b.name == bird_name), {})
    if not bird:
        return 'Sorry, the bird you are looking for is in another castle', status.HTTP_404_NOT_FOUND
    return jsonpickle.encode(bird, unpicklable=False)

if __name__ == '__main__':
    app.run()
