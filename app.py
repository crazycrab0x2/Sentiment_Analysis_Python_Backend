from flask import Flask, request, send_file
from flask_cors import CORS
import os
import whisper
import datetime
import subprocess
import torch
from pyannote.audio import Audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.core import Segment
import wave
import pydub
import contextlib
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from transformers import pipeline
# from werkzeug import secure_filename

app = Flask(__name__, static_folder='uploads', static_url_path='/uploads')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 1GB in bytes
app.config['TIMEOUT'] = 60

cors = CORS()
cors.init_app(app, resource={r"/api/*": {"origins": "*"}})

num_speakers = 2
model_size = 'base'
embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb")
model = whisper.load_model(model_size)
audio = Audio()
model_path = "cardiffnlp/twitter-roberta-base-sentiment"
sentiment_pipeline = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
sentiment = {
	"LABEL_2": "Positive",
	"LABEL_1": "Neutral",
	"LABEL_0": "Negative",
}

def get_file_names(directory):
    file_names = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_names.append(file)
    return file_names

def segment_embedding(segment, dur, filepath):
    start = segment["start"]
    # Whisper overshoots the end timestamp in the last segment
    end = min(dur, segment["end"])
    clip = Segment(start, end)
    waveform, sample_rate = audio.crop(filepath, clip)
    return embedding_model(waveform[None])

def time(secs):
    return datetime.timedelta(seconds=round(secs))


@app.route('/get_file_list', methods=['POST'])
def get_file_list():
    return get_file_names("./uploads")

@app.route('/api/upload', methods=['POST'])
def upload():
    for fname in request.files:
        f = request.files.get(fname)
        f.save('./uploads/'+fname)

    return 'Okay!'

@app.route('/get_sentiments', methods=['POST'])
def get_sentiments():
    body = request.get_json()
    result = []
    path = "uploads/" + body["filename"]
    transcription = model.transcribe(path)
    segments = transcription["segments"]
    embeddings = np.zeros(shape=(len(segments), 192))

    with contextlib.closing(wave.open(path,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        
    for i, segment in enumerate(segments):
        embeddings[i] = segment_embedding(segment, duration, path)

    embeddings = np.nan_to_num(embeddings)

    clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
    labels = clustering.labels_
    for i in range(len(segments)):
        segments[i]["speaker"] = 'Speaker ' + str(labels[i] + 1)

    for (i, segment) in enumerate(segments):
        if i==0 or segments[i - 1]["speaker"] != segment["speaker"]:  #different speaker
            result.append({"index": len(result), "speaker": segment["speaker"], "start": str(time(segment["start"])), "end": str(time(segment["end"])), "duration": str(time(segment["end"] - segment["start"])), "text": segment["text"][1:], "sentiment": sentiment[sentiment_pipeline(segment["text"][1:])[0]["label"]]})
            start = time(segment["start"])
        else:  #same speaker
            result[len(result)-1]["text"] += segment["text"]
            result[len(result)-1]["sentiment"] = sentiment[sentiment_pipeline(result[len(result)-1]["text"])[0]["label"]]
            result[len(result)-1]["duration"] = str(time(segment["end"]) - start)
            result[len(result)-1]["end"] = str(time(segment["end"]))

    return result

if __name__ == '__main__':
    app.run()
