from flask import Flask
from bird import Bird
import jsonpickle

app = Flask(__name__)

birds = [
    Bird('parus_major', 'Parus Major', 'Great tit', 'Rasvatihane', 'The super awesome bird'),
    Bird('parus_minor', 'Parus Major', 'Great tit', 'Rasvatihane', 'This bird doesn\'t even exist')
]

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
