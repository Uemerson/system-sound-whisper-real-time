import threading
import time
import warnings
from queue import Queue

import librosa
import numpy as np
import soundcard as sc
import soundfile as sf
from faster_whisper import WhisperModel
from pydub import AudioSegment

warnings.filterwarnings("ignore")


def is_audio_silent(file_path="output.wav", threshold=0.01):
    """Check if audio is silent"""
    y, _ = librosa.load(file_path)
    return np.all(np.abs(y) < threshold)


class CallbackThread:

    def __init__(self, interval=5):
        self.interval = interval
        self.running = False
        self.thread = threading.Thread(target=self.run)
        self.data_queue = Queue()
        self.combined = AudioSegment.empty()

    def start(self):
        self.running = True
        self.thread.start()

    def run(self):
        start_time = time.time()
        while self.running:
            with sc.get_microphone(
                id=str(sc.default_speaker().name), include_loopback=True
            ).recorder(samplerate=48000) as mic:
                data = mic.record(numframes=48000)
                sf.write(
                    file="audio_segment.wav", data=data[:, 0], samplerate=48000
                )
                audio_segment = AudioSegment.from_wav("audio_segment.wav")
                self.combined += audio_segment

            current_time = time.time()
            elapsed_time = current_time - start_time

            if elapsed_time >= self.interval:
                self.data_queue.put(self.combined)
                self.combined = AudioSegment.empty()
                start_time = current_time

    def stop(self):
        self.running = False
        self.thread.join()


def main():
    model = "large-v3"
    audio_model = WhisperModel(model, device="cuda", compute_type="float16")
    print(f"load model: {model}")

    callback_thread = CallbackThread(2)
    callback_thread.start()

    try:
        while True:
            audio = callback_thread.data_queue.get()
            audio.export("output.wav", format="wav")
            if not is_audio_silent("output.wav"):
                result, _ = audio_model.transcribe("output.wav", language="pt")
                result = list(result)

                if len(result) > 0:
                    text = result[0].text.strip()
                    print(f"speaker: {text}")

            time.sleep(0.25)
    except KeyboardInterrupt:
        callback_thread.stop()
        print("Thread stopped.")


if __name__ == "__main__":
    main()
