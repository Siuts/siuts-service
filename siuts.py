import os
from flask import Flask, request
from flask.ext.api import status
from flask.ext.redis import FlaskRedis
from flask_uploads import UploadSet, configure_uploads, patch_request_class
from threading import Thread
from bird import Bird
from pprint import pformat
import jsonpickle
import json
import uuid
import time

UPLOAD_FOLDER = '/Users/timo/projects/siuts-service/audiodata'

app = Flask(__name__)
redis_store = FlaskRedis(app)

audiodata = UploadSet('audiodata', default_dest='audiodata')
app.config['UPLOADED_AUDIODATA_DEST'] = UPLOAD_FOLDER
audiofiles = UploadSet('audiofiles', ('mp3', '3gpp', 'jpg'), lambda app: UPLOAD_FOLDER)
configure_uploads(app, (audiofiles,))

birds = []
with open('birds.json') as data_file:
    data = json.load(data_file)

for b in data['birds']:
    birds.append(Bird(b['name'], b['name_la'], b['name_en'], b['name_ee'], b['description']))

@app.route('/classify', methods=['POST'])
def process_classification_request():
    if 'audio_data' not in request.files:
        return 'Invalid request, audio_data missing', status.HTTP_400_BAD_REQUEST
        
    print ('======================================================')
    print (request.files['audio_data'])
    filename = audiofiles.save(request.files['audio_data'])
    fullfilename = os.path.join(UPLOAD_FOLDER, filename)
    print ('FILENAME', fullfilename)
    
    # TODO: stuff
        
    request_id = str(uuid.uuid4())
    redis_store.set(request_id, '')
    background_thread = Thread(target=classify, args=(audio_data,request_id))
    background_thread.start()
    return request_id

def classify(audio_data, request_id):
    ### Do magic here ###
    time.sleep(20)
    result = [{'name': 'phylloscopus_sibilatrix', 'match': 53.11}, {'name': 'parus_major', 'match': 72.15}]
    enriched_result = []
    for r in result:
        bird = get_bird_by_name(r['name'])
        if not bird:
            pass
        enriched_bird = eval(bird).copy()
        enriched_bird.update({'match': r['match']})
        enriched_result.append(enriched_bird)
    redis_store.set(request_id, jsonpickle.encode(enriched_result, unpicklable=False))

@app.route('/classify/<string:request_id>', methods=['GET'])
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

@app.route('/test_endpoint', methods=['POST'])
def test_endpoint():
    if 'audio_data' in request.files:
        print ('======================================================')
        print (request.files['audio_data'])
        filename = audiofiles.save(request.files['audio_data'])
        fullfilename = os.path.join(UPLOAD_FOLDER, filename)
        print ('FILENAME', fullfilename)
        return 'TEST'

if __name__ == '__main__':
    app.run()
