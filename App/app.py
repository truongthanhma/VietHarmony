from flask import Flask, render_template, request, send_from_directory
from keras.models import load_model
import numpy as np
import librosa.display
import cv2
import matplotlib.pyplot as plt
import time
import soundfile as sf
import os
import statistics 
from owlready2 import *
import os
from borda_score import borda_score

os.environ["KERAS_BACKEND"] = "tensorflow"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

path = "./Spectrograms/"
categories = ['Ca_tru', 'Cai_luong', 'Cheo', 'Exception', 'Hat_ba_trao', 'Hat_chau_van', 'Hat_xoan', 'Ho',
              'Nhac_cung_dinh', 'Nhac_tai_tu', 'Quan_ho', 'Tuong', 'Xam']

# Load models
MODEL_PATHS = [
    './Model/DenseNet169_128x128_bs32_10s_ver2.h5',
    './Model/ResNet50_128x128_bs32_10s_ver2.h5',
    './Model/MobileNetV2_128x128_bs32_10s_ver2.h5',
]

models = [load_model(model_path, compile=True) for model_path in MODEL_PATHS]

# Read music information from text file
music_info = {}
try:
    with open('music_info.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()  
            if not line:  
                continue

            parts = line.split(':')
            if len(parts) >= 2:  
                category = parts[0].strip()
                info = ':'.join(parts[1:]).strip() 
                music_info[category] = info
            else:
                print(f"Warning: Skipping line with invalid format: {line}") 

except FileNotFoundError:
    print("Error: music_info.txt not found.")
except Exception as e:
    print(f"Error reading music_info.txt: {e}")

# Read YouTube links from text files
youtube_links = {}
for class_name in categories:
    link_file_path = f'./youtube_links/{class_name}.txt'
    with open(link_file_path, 'r', encoding='utf-8') as f:
        links = f.readlines()
        youtube_links[class_name] = [link.strip() for link in links]

# Read music information from text file
def read_info_link(class_name):
    info_path = f'./class_info/{class_name}.txt'
    with open(info_path, 'r') as file:
        link = file.read()
    return link


def read_info_music(class_name):
    info_path = f'./class_info_music/{class_name}.txt'
    with open(info_path, 'r', encoding='utf-8', errors='replace') as file:
        music = file.read()
    return music


def create_spectrogram(audio_file, image_file):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    y, sr = librosa.load(audio_file)
    ms = librosa.feature.melspectrogram(y=y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(log_ms, sr=sr)

    fig.savefig(image_file)
    plt.close(fig)


def cut_wav_file(input_file, output_directory, duration):
    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Read the input audio file
    data, samplerate = sf.read(input_file)

    # Calculate the number of parts to split the audio file into
    num_parts = int(len(data) / (samplerate * duration))

    count = 0
    segment_parts = []
    # Split the audio file into parts
    for i in range(num_parts):
        start = i * samplerate * duration
        end = min((i + 1) * samplerate * duration, len(data))
        part = data[start:end]

        # Create the output file path
        output_file = output_directory + '/test{}.wav'.format(count)
        count += 1

        # Write the part to the output file
        sf.write(output_file, part, samplerate)
        segment_parts.append(output_file)

    return segment_parts

@app.route('/')
def index():
    # Render trang HTML từ thư mục 'templates'
    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/get_audio/<int:segment_number>')
def get_audio(segment_number):
    audio_path = f'uploads/audio_cut/test{segment_number}.wav'
    return send_from_directory(app.config['UPLOAD_FOLDER'], audio_path)


@app.route('/upload', methods=['POST'])
def upload():
    if 'audio' not in request.files:
        return "Audio file not found!"

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return "No audio file selected!"

    # Delete all .wav files in the uploads directory
    cut_directory = 'uploads/audio_cut'
    for file in os.listdir(cut_directory):
        file_path = os.path.join(cut_directory, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Unable to delete file {file_path}: {e}")

    # Delete all .png files in the static directory
    static_directory = 'static'
    for file in os.listdir(static_directory):
        if file.endswith('.png'):
            file_path = os.path.join(static_directory, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Unable to delete file {file_path}: {e}")

    audio_file.save('uploads/audio.wav')
    segment_parts = cut_wav_file('uploads/audio.wav', 'uploads/audio_cut', 10)

    model_results = []
    computation_time = []
    for i, model in enumerate(models):
        print(f"Using model: {MODEL_PATHS[i]}")
        segment_predictions = []
        for i, segment_part in enumerate(segment_parts):
            start = time.time()
            create_spectrogram(segment_part, f'static/spectrogram{i}.png')
            img_path = f'static/spectrogram{i}.png'
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128)) 
            img_array = img / 255.0
            img_batch = np.expand_dims(img_array, axis=0)

            pred = model.predict(img_batch)
            class_name = categories[np.argmax(pred)]
            confidence = pred[0][np.argmax(pred)] * 100

            print(
                f"Segment {i + 1}: Label: {class_name}, Accuracy: {confidence}%")

            # print(f'Time: {time.time() - start}')
            computation_time.append(time.time() - start)

            segment_predictions.append(
                {'segment_number': i, 'class_name': class_name, 'confidence': confidence, 'img_path': img_path})
        # Save the predictions for the model
        model_results.append(segment_predictions)

    print("time: ", statistics.mean(computation_time))

    # Calculate Borda ranking
    borda_rankings = []
    for model_prediction in model_results:
        # Calculate the Borda ranking for each model
        class_counts = {}
        for i, segment_prediction in enumerate(model_prediction):
            class_name = segment_prediction['class_name']
            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1
        borda_rankings.append(class_counts)
        # Print the Borda ranking for the model
        print(f"Borda ranking for model i: {class_counts}")
    common_labels = set()
    for ranking in borda_rankings:
        common_labels.update(ranking.keys())
    borda_results = borda_score(borda_rankings, common_labels)
    print(f"Borda results: {borda_results}")

    most_common_label = max(borda_results, key=borda_results.get)
    print(f"Music genre: {most_common_label}")

    most_common_results = [
        result for result in segment_predictions if result['class_name'] == most_common_label]
    highest_accuracy_common = max(
        most_common_results, key=lambda x: x['confidence'])['confidence']

    total_accuracy_common = sum(
        result['confidence'] for result in most_common_results) / len(most_common_results)

    last_segment_confidences_common = [
        result['confidence'] for result in most_common_results]
    median_confidence_common = statistics.median(
        last_segment_confidences_common)

    info_link = read_info_link(most_common_label)
    info_music = read_info_music(most_common_label)
    music_description = music_info.get(most_common_label, "No information")

    audio_path = f'uploads/audio.wav?t={int(time.time())}'

    segment_img_paths = f'static/spectrogram{most_common_results[-1]["segment_number"]}.png'

    ontology_info = []

    # Load ontology
    ontology_path = "./ontology/VIPRIME_en.owl"
    onto = get_ontology(ontology_path).load()

    music_type = None

    if most_common_label == 'Ca_tru':
        music_type = onto.Ca_tru
    elif most_common_label == 'Cai_luong':
        music_type = onto.Cai_luong
    elif most_common_label == 'Cheo':
        music_type = onto.Cheo
    elif most_common_label == 'Hat_ba_trao':
        music_type = onto.Hat_ba_trao
    elif most_common_label == 'Hat_chau_van':
        music_type = onto.Hat_chau_van
    elif most_common_label == 'Ho':
        music_type = onto.Ho
    elif most_common_label == 'Quan_ho':
        music_type = onto.Quan_ho
    elif most_common_label == 'Hat_xoan':
        music_type = onto.Hat_xoan
    elif most_common_label == 'Xam':
        music_type = onto.Xam
    elif most_common_label == 'Tuong':
        music_type = onto.Tuong
    elif most_common_label == 'Dan_ca_tai_tu':
        music_type = onto.Dan_ca_tai_tu
    elif most_common_label == 'Nhac_cung_dinh':
        music_type = onto.Nhac_cung_dinh

    if most_common_label == 'Exception':
        ontology = {}
        ontology["Tên tác phẩm"] = ""
        ontology["Của tác giả"] = ""
        ontology["Link bài hát"] = ""
        ontology["Là của dân tộc"] = ""
        ontology["Có nguồn gốc từ"] = ""
        ontology["Là nhạc của"] = ""
        ontology["Thuộc dòng nhạc"] = ""
        ontology["Mô tả"] = ""
        ontology_info.append(ontology)
    else:
        label_list = list(onto.search(type=music_type))

        for individual in label_list:
            ontology = {}
            ontology["Tên tác phẩm"] = individual.co_ten_la[0]
            ontology["Của tác giả"] = individual.cua_tac_gia[0]
            ontology["Link bài hát"] = individual.co_URL_la[0]
            nhac_cua_dan_toc = individual.la_cua_dan_toc
            ontology["Là của dân tộc"] = [
                ndt.label[0] for ndt in nhac_cua_dan_toc]
            nguon_goc = individual.co_nguon_goc_tu
            ontology["Có nguồn gốc từ"] = [
                ng.label[0] for ng in nguon_goc]
            nhac_cua = individual.la_nhac_cua
            ontology["Là nhạc của"] = [nc.label[0] for nc in nhac_cua]
            ontology["Thuộc dòng nhạc"] = [
                nc.label[0] for nc in onto.get_parents_of(music_type)]
            ontology["Mô tả"] = music_type.duoc_mo_ta_la[0]
            ontology_info.append(ontology)

    return render_template(
        'index.html',
        category=info_music,
        confidence_common=total_accuracy_common,
        highest_accuracy_common=highest_accuracy_common,
        median_confidence_common=median_confidence_common,
        music_description=music_description,
        audio_path=audio_path,
        info_link=info_link,
        youtube_links=youtube_links.get(most_common_label, []),
        segment_img_paths=segment_img_paths,
        ontology_info=ontology_info
    )


if __name__ == '__main__':
    app.run(debug=True)
