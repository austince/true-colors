import cv2
import glob
import random
import numpy as np
import os
import sys

from util.constants import classifier_data_filename, emotions

classifier = cv2.face_FisherFaceRecognizer.create()  # Initialize fisher face classifier
dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, classifier_data_filename)
num_training_runs = 300


def get_files(emotion):  # Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("dataset/%s/*" % emotion)
    random.shuffle(files)
    training = files[:int(len(files) * 0.8)]  # get first 90% of file list
    prediction = files[-int(len(files) * 0.2):]  # get last 10% of file list
    return training, prediction


def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion_index, emotion in enumerate(emotions):
        training, prediction = get_files(emotion)
        # Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item)  # open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            training_data.append(gray)  # append image array to training data list
            training_labels.append(emotion_index)

        for item in prediction:  # repeat above process for prediction set
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emotion_index)

    return training_data, training_labels, prediction_data, prediction_labels


def train_classifier(training_data, training_labels):
    global classifier
    print "training fisher face classifier"
    print "size of training set is:", len(training_labels), "images"
    classifier.train(training_data, np.asarray(training_labels))


def run_classifier(prediction_data, prediction_labels):
    global classifier
    print "predicting classification set"
    correct = 0
    incorrect = 0
    for i, image in enumerate(prediction_data):
        pred, conf = classifier.predict(image)
        if pred == prediction_labels[i]:
            correct += 1
        else:
            print("Misclassified %s as %s" % (emotions[pred], emotions[prediction_labels[i]]))
            incorrect += 1

    return (100 * correct) / len(prediction_data)


def train(num_runs):
    meta_score = []
    for i in range(num_runs):
        print("Training run #%i/%i" % (i + 1, num_runs))
        training_data, training_labels, prediction_data, prediction_labels = make_sets()
        train_classifier(training_data, training_labels)
        num_correct = run_classifier(prediction_data, prediction_labels)
        print("got %i percent correct!" % (num_correct))
        meta_score.append(num_correct)

    print("end score: %i percent correct!" % (np.mean(meta_score)))


def save_classifier():
    classifier.write(data_path)


def get_classifier():
    global classifier, data_path
    if not os.path.exists(data_path):
        train(num_training_runs)
        save_classifier()
    else:
        classifier.read(data_path)

    return classifier


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        data_path = sys.argv[1]
    if len(sys.argv) == 3:
        num_training_runs = int(sys.argv[2])
    try:
        # Now run it
        print("Training classifier %s on %s emotions for %i runs." % (os.path.basename(data_path), ",".join(emotions), num_training_runs))
        train(num_training_runs)
        print("Saving classifier")
        save_classifier()
    except KeyboardInterrupt:
        print("Stopping early!")
        save_classifier()
