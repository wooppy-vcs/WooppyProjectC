"""Main script to run host REST endpoints for categorizing images, pip install flask """
# !flask/bin/python
import base64
import operator
import collections
import os
import tensorflow as tf
import time
import datetime
import numpy as np

from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template, jsonify
from werkzeug.utils import secure_filename
from cnnTextClassifier import predictor
from inception.inception import inception_predict as inception
from tensorflow.python.platform import gfile
from pdf_manipulator import pdf_to_text
from pdf_manipulator import wand_pdf_to_image
from pdf_manipulator import image_to_text

IMAGE_UPLOAD_FOLDER = os.path.join('uploads', 'image_files')
TEXT_UPLOAD_FOLDER = os.path.join('uploads', 'texts')
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

classification_dict = {0: "Others",
                       1: "Storage & Memory Cards",
                       2: "Tablets",
                       3: "Screen Protectors",
                       4: "Cool Gadgets",
                       5: "Cables & Charges",
                       6: "Mobile Phones",
                       7: "Cases & Covers",
                       8: "Mobile Car Accessories",
                       9: "Wearables",
                       10: "Audio",
                       11: "Powerbanks & Batteries",
                       12: "Camera & Accessories",
                       13: "Selfie Accessories"}

app = Flask(__name__, static_folder=os.path.join("templates", "assets"))
app.secret_key = 'rem4lyfe'
app.config['IMAGE_UPLOAD_FOLDER'] = IMAGE_UPLOAD_FOLDER
app.config['TEXT_UPLOAD_FOLDER'] = TEXT_UPLOAD_FOLDER


def try_make_dir(try_path):
    if not gfile.Exists(try_path):
        os.makedirs(try_path)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def format_category_and_confidence(category, probability):
    return {"category": category, "probability": str("%.4f" % probability)}


def read_image(save_path):
    # img = Image.open(save_path)
    # print(Image.MIME[img.format])
    image = open(save_path, "rb")
    data = image.read()
    image.close()
    image_data = base64.b64encode(data)
    print(image_data)
    return image_data


# Example of how to use templates
@app.route('/index', methods=['GET'])
def index():
    print(os.path.join('templates', 'index.html'))
    return render_template(os.path.join('index.html'))


@app.route('/', methods=['GET'])
def upload_file():
    return render_template(os.path.join('upload.html'))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(os.path.join(app.config['IMAGE_UPLOAD_FOLDER'], 'original'), filename)


@app.route('/infer', methods=['POST'])
def infer_category():

    if request.method == 'POST':

        if 'file' not in request.files:
            flash('No file detected. Please upload a file!')
            return redirect(request.url)

        file = request.files['file']
        # mime = magic.Magic(mime=True)
        # print(file.content_type)
        if file.content_type != 'application/pdf':
            flash('File uploaded is not a PDF document. Please upload file in PDF format!')
            return redirect(request.url)

        if file.filename == '':
            flash('No selected file. Found empty filename.')
            return redirect(request.url)

        # Saving each interaction into folders with time as unique id
        folder_id = time.strftime("%Y%m%dT%H%M%S")
        pdf_save_path = os.path.join('uploads', folder_id, 'pdf')
        try_make_dir(pdf_save_path)
        pdf_path = os.path.join(pdf_save_path, folder_id + ".pdf")
        file.save(pdf_path)

        # Try to extract text. Removes trailing whitespaces
        extracted_text = pdf_to_text.extract_text(files=[pdf_path]).strip()
        # print(extracted_text)

        # Try to convert pdf to image
        image_save_path = os.path.join('uploads', folder_id, 'image')
        try_make_dir(image_save_path)
        num_pages, saved_image_path_list = wand_pdf_to_image.save_as_image(file=pdf_path, destination=image_save_path)

        # Join all pages into one vertical image for image recognition
        vertical_image_path = wand_pdf_to_image.make_vertical_image(saved_image_path_list=saved_image_path_list,
                                                                    save_image_folder=image_save_path)

        # If no text found
        if not extracted_text:
            # Run OCR SHIT
            extracted_text = ""
            for page in range(num_pages):
                extracted_text = extracted_text + image_to_text.extract_text_from_image(filepath=os.path.join(image_save_path, str(page) + '.png')) + "\n"
                # print(extracted_text)

        # Run text classification
        prediction, probabilities = predictor.predict(x_raw=extracted_text,
                                                      checkpoint_dir="models/text/cnn_text_bk_with04/checkpoints")

        idtoprobability_dict = collections.OrderedDict()
        classtoprobability_dict = collections.OrderedDict()
        for idx, probability in enumerate(probabilities[0:14]):
            idtoprobability_dict[int(idx)] = float(probability)
            classtoprobability_dict[classification_dict[int(idx)]] = float(probability)

        sorted_idtoprobability = sorted(idtoprobability_dict.items(), key=operator.itemgetter(1), reverse=True)
        sorted_classtoprobability = sorted(classtoprobability_dict.items(), key=operator.itemgetter(1),
                                           reverse=True)

        beautified_text = []

        for x in sorted_classtoprobability:
            beautified_text.append(format_category_and_confidence(x[0], x[1]))

        text_json = {"1st Prediction": format_category_and_confidence(
            classification_dict[sorted_idtoprobability[0][0]], sorted_idtoprobability[0][1]),
            "2nd Prediction": format_category_and_confidence(
                classification_dict[sorted_idtoprobability[1][0]], sorted_idtoprobability[1][1]),
            "3rd Prediction": format_category_and_confidence(
                classification_dict[sorted_idtoprobability[2][0]], sorted_idtoprobability[2][1]),
            "All classes": beautified_text}
        #
        # with open(os.path.join(app.config['TEXT_UPLOAD_FOLDER'], 'textlog_short.txt'), 'a') as logfileshort:
        #     logfileshort.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        #     logfileshort.write('Input: ' + text + '\n')
        #     logfileshort.write(classification_dict[sorted_idtoprobability[0][0]].ljust(25) + str("%.4f" % sorted_idtoprobability[0][1]) + '\n')
        #     logfileshort.write(classification_dict[sorted_idtoprobability[1][0]].ljust(25) + str("%.4f" % sorted_idtoprobability[1][1]) + '\n')
        #     logfileshort.write(classification_dict[sorted_idtoprobability[2][0]].ljust(25) + str("%.4f" % sorted_idtoprobability[2][1]) + '\n\n')
        #
        #     logfileshort.close()
        #
        # with open(os.path.join(app.config['TEXT_UPLOAD_FOLDER'], 'textlog_long.txt'), 'a') as logfilelong:
        #     logfilelong.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        #     logfilelong.write('Input: ' + text + '\n')
        #     for x in sorted_classtoprobability:
        #         logfilelong.write(x[0].ljust(25) + str("%.8f" % x[1]) + '\n')
        #     logfilelong.write('\n')
        #
        #     logfilelong.close()

        # Run image classification

        logits, probs = inception.predict(vertical_image_path)
        all_probabilities = probs[0][1:]

        # for class_predictions in prediction:
        #     predicted_class = int(class_predictions["classes"])
        #     all_probabilities = list(class_predictions["probabilities"])
        #     predicted_probability = float(all_probabilities[predicted_class])
        #     # print(class_predictions["classes"])
        #     # print(class_predictions["probabilities"])
        #     break

        # Below code redirects to uploaded image
        # return redirect(url_for('uploaded_file',
        #                         filename=filename))
        # all_classes_probabilities = {}
        all_classes_probabilities = collections.OrderedDict()

        i = 0
        for probabilities in all_probabilities:
            all_classes_probabilities[classification_dict[i]] = float(probabilities)
            i += 1

        sorted_all_classes_probabilities = sorted(all_classes_probabilities.items(), key=operator.itemgetter(1),
                                                  reverse=True)

        beautified_image = []

        for y in sorted_all_classes_probabilities:
            beautified_image.append(format_category_and_confidence(y[0], y[1]))

        # my_json = [{"predicted_class": classification_dict[predicted_class],
        #             "probability": predicted_probability,
        #             "all_probabilities": all_classes_probabilities}]

        image_json = {"1st Prediction": format_category_and_confidence(
            sorted_all_classes_probabilities[0][0], sorted_all_classes_probabilities[0][1]),
            "2nd Prediction": format_category_and_confidence(
                sorted_all_classes_probabilities[1][0], sorted_all_classes_probabilities[1][1]),
            "3rd Prediction": format_category_and_confidence(
                sorted_all_classes_probabilities[2][0], sorted_all_classes_probabilities[2][1]),
            "All classes": beautified_image}

        # log the interaction
        # with open(os.path.join(app.config['IMAGE_UPLOAD_FOLDER'], 'imagelog_short.txt'), 'a') as logfileshort:
        #     logfileshort.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        #     logfileshort.write('Filename: ' + filename + '\n')
        #     logfileshort.write(sorted_all_classes_probabilities[0][0].ljust(25) + str(
        #         "%.4f" % sorted_all_classes_probabilities[0][1]) + '\n')
        #     logfileshort.write(sorted_all_classes_probabilities[1][0].ljust(25) + str(
        #         "%.4f" % sorted_all_classes_probabilities[1][1]) + '\n')
        #     logfileshort.write(sorted_all_classes_probabilities[2][0].ljust(25) + str(
        #         "%.4f" % sorted_all_classes_probabilities[2][1]) + '\n\n')
        #
        #     logfileshort.close()
        #
        # with open(os.path.join(app.config['IMAGE_UPLOAD_FOLDER'], 'imagelog_long.txt'), 'a') as logfilelong:
        #     logfilelong.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        #     logfilelong.write('Filename: ' + filename + '\n')
        #     for x in sorted_all_classes_probabilities:
        #         logfilelong.write(x[0].ljust(25) + str("%.8f" % x[1]) + '\n')
        #     logfilelong.write('\n')
        #
        #     logfilelong.close()

        # LOGIC TO COMPARE RESULTS FROM IMAGE AND TEXT
        mean_dict = collections.OrderedDict()

        threshold = 0.9
        for key, value in all_classes_probabilities.items():
            if value > threshold and classtoprobability_dict[key] > threshold:
                mean_dict[key] = (value + classtoprobability_dict[key]) / 2
            elif value > threshold:
                mean_dict[key] = value
            elif classtoprobability_dict[key] > threshold:
                mean_dict[key] = classtoprobability_dict[key]
            else:
                mean_dict[key] = (value + classtoprobability_dict[key]) / 2

                # mean_dict[key] = (value + classtoprobability_dict[key]) / 2

        sorted_mean_dict = sorted(mean_dict.items(), key=operator.itemgetter(1), reverse=True)

        beautified_mean = []

        for z in sorted_mean_dict:
            beautified_mean.append(format_category_and_confidence(z[0], z[1]))

        mean_json = {"1st Prediction": format_category_and_confidence(
            sorted_mean_dict[0][0], sorted_mean_dict[0][1]),
            "2nd Prediction": format_category_and_confidence(
                sorted_mean_dict[1][0], sorted_mean_dict[1][1]),
            "3rd Prediction": format_category_and_confidence(
                sorted_mean_dict[2][0], sorted_mean_dict[2][1]),
            "All classes": beautified_mean}

        final_json = [{"model": "Overall", "data": mean_json}, {"model": "Text", "data": text_json},
                      {"model": "Image", "data": image_json}]

        return render_template(os.path.join('results.html'), result_json=final_json, original_string=text if has_text else None, file='uploads/'+filename if has_image else None)

    return render_template(os.path.join('upload.html'))


if __name__ == '__main__':
    # try_make_dir(app.config['IMAGE_UPLOAD_FOLDER'])
    # vggnet_b_classifier = tf.estimator.Estimator(
    #     model_fn=vggnet_b.vgg_net_b_dil,
    #     params=vggnet_b.DEFAULT_PARAMS,
    #     model_dir=vggnet_b.final_image_model_dir,
    #     config=tf.estimator.RunConfig().replace(keep_checkpoint_max=1000))

    app.run(debug=False, host='0.0.0.0', port="5000")
