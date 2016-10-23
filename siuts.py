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
import os.path
from pydub import AudioSegment
import numpy as np
import pandas as pd
import scipy as sp
import pickle
from scipy import fft
from time import localtime, strftime
from skimage.morphology import disk, remove_small_objects
from skimage.filter import rank
from skimage.util import img_as_ubyte
import wave
import pylab
from numpy.lib import stride_tricks
import matplotlib.patches as patches
from os import listdir
from os.path import isfile, join
import scipy
import tensorflow as tf
from tensorflow.python.platform import gfile
import operator

UPLOAD_FOLDER = '/var/www/siuts/audiodata'
model_dir = "model/"

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



FFT_FRAME_SIZE = 512
FFT_FRAME_RES = 256
MIN_SEGMENT_SIZE = 400
P = 90  # percentange in binary
FRAME_RATE = 22050

SEGMENT_SIZE = 64

SAVE_PLOT = False
USE_LOG_SPECTOGRAMS = False


def convert_to_wav(filePath):
    wav_path_list = filePath.split(".")

    wav_path_list[-1] = "wav"
    wav_path = '.'.join(wav_path_list)
    sound = AudioSegment.from_file(filePath)
    sound = sound.set_frame_rate(22050).set_channels(1)
    sound.export(wav_path, format="wav")
    return wav_path


def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'Int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate


def stft(sig, frameSize, frameRes, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(np.floor(frameSize / 2.0)), sig)
    # cols for windowing
    cols = np.ceil((len(samples) - frameSize) / float(hopSize)) + 1

    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize),
                                      strides=(samples.strides[0] * hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.fft(frames, frameRes)


def segment_spectogram(mypic_rev, min_segment_size, p, sigma=3, grad_disk=3, ):
    mypic_rev_gauss = sp.ndimage.gaussian_filter(mypic_rev, sigma=sigma)
    mypic_rev_gauss_bin = mypic_rev_gauss > np.percentile(mypic_rev_gauss, p)
    mypic_rev_gauss_bin_close = sp.ndimage.binary_closing(sp.ndimage.binary_opening(mypic_rev_gauss_bin))
    mypic_rev_gauss_grad = rank.gradient(pic_to_ubyte(mypic_rev_gauss), disk(grad_disk))
    mypic_rev_gauss_grad_bin = mypic_rev_gauss_grad > np.percentile(mypic_rev_gauss_grad, p)
    mypic_rev_gauss_grad_bin_close = sp.ndimage.binary_closing(sp.ndimage.binary_opening(mypic_rev_gauss_grad_bin))
    bfh = sp.ndimage.binary_fill_holes(mypic_rev_gauss_grad_bin_close)
    bfh_rm = remove_small_objects(bfh, min_segment_size)
    return sp.ndimage.label(bfh_rm)


def pic_to_ubyte(pic):
    a = (pic - np.min(pic)) / (np.max(pic - np.min(pic)))
    a = img_as_ubyte(a)
    return a


def get_segments_t(filePath):
    wave_path = convert_to_wav(filePath)
    signal, fs = get_wav_info(wave_path)
    spectogram = abs(stft(signal, FFT_FRAME_SIZE, FFT_FRAME_RES))

    fname = "default"
    SPEC_SEGMENTS = []
    big_ROIs = 0

    mypic_rev = fft
    if not USE_LOG_SPECTOGRAMS:
        labeled_segments, num_seg = segment_spectogram(spectogram, MIN_SEGMENT_SIZE, P)

    if USE_LOG_SPECTOGRAMS:
        spectogram_log = np.log10(spectogram + 0.001)
        labeled_segments, num_seg = segment_spectogram(spectogram_log, MIN_SEGMENT_SIZE, P)

    if SAVE_PLOT:
        fig = plt.figure()
        plot = fig.add_subplot(111, aspect='equal')
        plot.imshow(spectogram)
        plot.axis('off')

    not_allowed_centers_list = []
    for current_segment_id in range(1, num_seg + 1):
        current_segment = (labeled_segments == current_segment_id) * 1
        xr = current_segment.max(axis=1)
        yr = current_segment.max(axis=0)
        xr_max = np.max(xr * np.arange(len(xr)))
        xr[xr == 0] = xr.shape[0]
        xr_min = np.argmin(xr)
        yr_max = np.max(yr * np.arange(len(yr)))
        yr[yr == 0] = yr.shape[0]
        yr_min = np.argmin(yr)
        xr_width = xr_max - xr_min
        if xr_width > FFT_FRAME_RES:
            big_ROIs += 1
            if SAVE_PLOT:
                plot.add_patch(patches.Rectangle((xr_min, yr_min), (xr_max - xr_min), (yr_max - yr_min),
                                                 fill=None, edgecolor="red", linewidth=0.1))
        else:
            if SAVE_PLOT:
                plot.add_patch(patches.Rectangle((xr_min, yr_min), (xr_max - xr_min), (yr_max - yr_min),
                                                 fill=None, edgecolor="green", linewidth=0.1))

        xr_center = xr_max - xr_width / 2
        xr_min = xr_center - FFT_FRAME_RES / 2
        xr_max = xr_center + FFT_FRAME_RES / 2

        if (xr_min >= 0 and xr_max <= len(spectogram) and xr_center not in not_allowed_centers_list):
            new_not_allowed_centers = range(xr_center - FFT_FRAME_RES / 4, xr_center + FFT_FRAME_RES / 4)
            not_allowed_centers_list = not_allowed_centers_list + new_not_allowed_centers
            yr_min = 0
            yr_max = FFT_FRAME_RES

            segment_frame = [xr_min, xr_max, yr_min, yr_max]
            subpic = np.array(spectogram[xr_min:xr_max, yr_min:yr_max])

            resized_subpic = scipy.misc.imresize(subpic, (SEGMENT_SIZE, SEGMENT_SIZE), interp='nearest')

            SPEC_SEGMENTS.append(np.array(resized_subpic))

    if SAVE_PLOT:
        fig.savefig(fname + '_segments.png', bbox_inches='tight', dpi=600)
        fig.clear()

    return SPEC_SEGMENTS


def denoise(filePath):
    wave_path = convert_to_wav(filePath)
    signal, fs = get_wav_info(wave_path)
    spectogram = abs(stft(signal, FFT_FRAME_SIZE, FFT_FRAME_RES))
    spectogram = np.array(spectogram)
    spectogram = spectogram.transpose()

    spec = spectogram[:128]

    filtered = np.zeros(np.shape(spec))
    i_i = 0
    for i in spec:
        r_mean = i.mean()
        if i_i >= np.shape(spec)[1]:
            return []
        c_mean = spec[:, i_i].mean()
        j_i = 0
        for j in i:
            if j > 2.5 * r_mean and j > 2.5 * c_mean:
                filtered[i_i, j_i] = 1
            else:
                filtered[i_i, j_i] = 0
            j_i += 1
        i_i += 1
    label = []
    label2 = np.zeros(np.shape(filtered)[1])
    for i in range(np.shape(filtered)[1]):
        if max(filtered[:, i]) == 1:
            # label.append(1)
            label2[i] = 1
            if i > 0:
                label2[i - 1] = 1
                # label2[i-2] = 1
                if i < np.shape(filtered)[1] - 1:
                    # print "{i}".format()
                    label2[i + 1] = 1
                    # label2[i+2] = 1

        else:
            # label.append(0)
            label2[i] = 0
    # for i in
    # cleaning time
    # sig_len = len(signal)
    # f_len = shape(filtered)[1]
    # win_size = sig_len/f_len + 1
    # last_win = sig_len - (f_len-1)*win_size
    cleaned_signal = np.zeros((np.shape(filtered)[0], 1))
    for i in range(len(label2) - 1):
        if label2[i] == 1:
            cleaned_signal = np.append(cleaned_signal, spec[:, i].reshape((np.shape(filtered)[0], 1)), axis=1)

    return cleaned_signal


def get_segments_a(wav_path):
    cleaned_signal = denoise(wav_path)
    segments = []
    if len(cleaned_signal) > 0:
        hop_size = cleaned_signal.shape[0] / 2
        for i in range(int(np.floor(cleaned_signal.shape[1] / hop_size - 1))):
            segment = cleaned_signal[:, i * hop_size:i * hop_size + cleaned_signal.shape[0]]
            segments.append(segment)
    return segments


def get_accuracies(filePath):
    unresized_segments = get_segments_t(filePath)
    # unresized_segments = get_segments_a(filePath)

    if (len(unresized_segments) <= 0):
        return []

    segments = np.resize(unresized_segments, (len(unresized_segments), SEGMENT_SIZE, SEGMENT_SIZE, 1))

    print "######### Successfully segmenteted!!! ##########"
    print np.array(segments).shape

    with tf.Session() as persisted_sess:
        with gfile.FastGFile(model_dir + "frozen_graph.pb", 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            persisted_sess.graph.as_default()
            tf_segment = tf.placeholder(tf.float32, shape=(1, 64, 64, 1))
            test_predictions_op = tf.import_graph_def(graph_def,
                                                      input_map={"tf_one_prediction:0": tf_segment},
                                                      return_elements=['sm_one:0'])

        testing_predictions = np.empty

        for i in range(len(segments)):
            current_predictions = test_predictions_op[0].eval(feed_dict={tf_segment: [segments[i]]})
            print current_predictions
            if i == 0:
                testing_predictions = test_predictions_op[0].eval(feed_dict={tf_segment: [segments[i]]})
            else:
                testing_predictions = np.concatenate((testing_predictions, current_predictions))

            print "Testing predictions: " + str(i) + " - " + str(testing_predictions.shape)

        averaged_predictions = np.mean(testing_predictions, axis=0)

        with open(model_dir + "labels.pickle", 'rb') as f:
            labels = pickle.load(f)

        predictions_dict = {}
        for i in range(10):
            predictions_dict[i] = averaged_predictions[i]
        sorted_preds = sorted(predictions_dict.items(), key=operator.itemgetter(1), reverse=True)

        result = []
        for i in range(3):
            result.append({'name': labels[sorted_preds[i][0]].lower(), 'match': round(sorted_preds[i][1] * 100, 2)})

    return result


@app.route('/classify', methods=['POST'])
def process_classification_request():
    if 'audio_data' not in request.files:
        return 'Invalid request, audio_data missing', status.HTTP_400_BAD_REQUEST
        
    print ('======================================================')
    print (request.files['audio_data'])
    filename = audiofiles.save(request.files['audio_data'])
    audio_file_name = os.path.join(UPLOAD_FOLDER, filename)
    print ('FILENAME', audio_file_name)

    request_id = str(uuid.uuid4())
    redis_store.set(request_id, '')
    background_thread = Thread(target=classify, args=(audio_file_name,request_id))
    background_thread.start()
    return request_id

def classify(audio_file_name, request_id):

    result = get_accuracies(audio_file_name)

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
