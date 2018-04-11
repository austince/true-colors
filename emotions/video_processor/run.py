# From the root directory
# python -m emotions.video_processor.run {video-name} {function}
import ffmpeg
import sys
import subprocess
from string import Template
import os
from glob import glob
from ..facial_emotion import detector


def bw(src_file):
    # (ffmpeg
    #  .input(src_file)
    #  # .filter_()
    #  .hue(s=0)
    #  .output(src_file + "-bw.mp4")
    #  .run()
    #  )
    new_file = src_file + "-bw.mp4"
    cmd = "ffmpeg -i %s -vf hue=s=0 -pix_fmt yuv420p -c:a copy %s" % (src_file, new_file)
    subprocess.call(cmd, shell=True)
    return new_file


# Convert to black and white
# ffmpeg -i "leaders.mp4" -vf hue=s=0 -c:a copy "leaders-bw.mp4"

def strip_frames(src_file):
    if not os.path.exists("frames/" + src_file):
        os.mkdir("frames/" + src_file)
    if "-bw.mp4" not in src_file:
        src_file += "-bw.mp4"
    cmd_tmp = Template("ffmpeg -i ${filename} -vf fps=30 -qscale:v 2 frames/${filename}/frame%05d.jpg")
    cmd = cmd_tmp.substitute(filename=src_file)
    subprocess.call(cmd, shell=True)


# Strip out frames @ 30 fps
# ffmpeg -i "leaders-bw.mp4" -vf fps=30 -qscale:v 2 frames/frame%04d.jpg


def make_video(src_file):
    cmd_tmp = Template(
        "ffmpeg -r 30 -i outframes/${filename}/frame%04d.jpg \
        -i audio/${filename}/${filename}.wav -acodec aac -vcodec copy -shortest ${filename}-output.mp4")
    cmd = cmd_tmp.substitute(filename=src_file)
    subprocess.call(cmd, shell=True)


def strip_audio(src_file):
    if not os.path.exists("audio/" + src_file):
        os.mkdir("audio/" + src_file)
    cmd_tmp = Template("ffmpeg -i ${filename} -q:a 0 -map a audio/${filename}/${filename}.wav")
    cmd = cmd_tmp.substitute(filename=src_file)
    subprocess.call(cmd, shell=True)


def run_detection(src_file):
    files = glob("frames/%s/frame*.jpg" % src_file)
    for i, img_file in enumerate(files):
        print("Detecting %i/%i" % (i + 1, len(files)))
        detector.detect_and_save(img_file)


def run_processing():
    subprocess.call("./run.sh", shell=True)

# Create video from frames
# ffmpeg -r 30 -i outframes/frame%04d.jpg -crf 15 output.mp4

# Add audio to video file
# ffmpeg -i output.mp4 -i leaders.wav -codec copy -shortest output-with-sound.mp4

# Do them both at the same time?
# ffmpeg -r 30 -i outframes/frame%04d.jpg -i leaders.wav -codec copy -shortest output-with-sound.mp4

# Copy audio from video
# ffmpeg -i leaders.mp4 -q:a 0 -map a leaders.wav


if __name__ == "__main__":
    f = sys.argv[1]
    fun = sys.argv[2]
    if fun == "bw":
        bw(f)
    elif fun == "sf":
        strip_frames(f)
    elif fun == "sa":
        strip_audio(f)
    elif fun == "video":
        make_video(f)
    elif fun == "det":
        run_detection(f)
    elif fun == "p":
        run_processing()
    elif fun == "run":
        strip_audio(f)
        new_f = bw(f)
        strip_frames(new_f)
        run_detection(new_f)
        run_processing()  # must change name in .pyde file
        make_video(new_f)
