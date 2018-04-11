import glob
import os
from shutil import copyfile

from facial_emotion.util.constants import emotions

participants = glob.glob("CKFaces/Emotion/*")  # Returns a list of all folders with participant numbers

def make_dirs():
    if not os.path.exists("sorted_set"):
        os.mkdir("sorted_set")

    for emotion in emotions:
        emotion_dir = os.path.join("sorted_set", emotion)
        if not os.path.exists(emotion_dir):
            os.mkdir(emotion_dir)


def organize():
    for x in participants:
        part = "%s" % x[-4:]  # store current participant number
        for session_num, sessions in enumerate(glob.glob("%s/*" % x)):  # Store list of sessions for current participant
            print("Session %i" % (session_num))
            for i, files in enumerate(glob.glob("%s/*" % sessions)):
                current_session = files[20:-30]
                session_img_file = open(files, 'r')

                # emotions are encoded as a float, readline as float, then convert to integer.
                emotion = int(float(session_img_file.readline()))

                # get path for last image in sequence, which contains the emotion
                sourcefile_emotion = glob.glob("CKFaces/Images/%s/%s/*" % (part, current_session))[-1]
                # do same for neutral image
                sourcefile_neutral = glob.glob("CKFaces/Images/%s/%s/*" % (part, current_session))[0]

                dest_neut = "sorted_set/neutral/%s" % sourcefile_neutral[25:]  # Generate path to put neutral image
                # Do same for emotion containing image
                dest_emot = "sorted_set/%s/%s" % (emotions[emotion], sourcefile_emotion[25:])

                copyfile(sourcefile_neutral, dest_neut)
                copyfile(sourcefile_emotion, dest_emot)


if __name__ == "__main__":
    make_dirs()
    organize()
