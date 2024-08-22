from flask import Flask, render_template, redirect, url_for
import subprocess
import os
import signal
import logging

app = Flask(__name__)

# Store the process ID for the OpenCV detection script
opencv_process = None

def start_opencv_process():
    global opencv_process
    # If the process is already running, do nothing
    if opencv_process is not None:
        return

    # Start the OpenCV face detection script
    try:
        opencv_process = subprocess.Popen(
            ['python', 'opencv_detection.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logging.info("Started OpenCV process.")
    except Exception as e:
        logging.error(f"Error starting OpenCV process: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    # Start the OpenCV detection process
    start_opencv_process()
    # Redirect back to the index page
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
