import sounddevice as sd
import numpy as np
import speech_recognition as sr
import tempfile
import scipy.io.wavfile as wav

recognizer = sr.Recognizer()


def listen_question(duration=5, samplerate=16000):

    print("Listening...")

    audio = sd.rec(int(duration * samplerate),
                   samplerate=samplerate,
                   channels=1,
                   dtype='int16')

    sd.wait()

    # Save temporary WAV file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav.write(temp_file.name, samplerate, audio)

    with sr.AudioFile(temp_file.name) as source:

        audio_data = recognizer.record(source)

    try:

        text = recognizer.recognize_google(audio_data)

        print("User:", text)

        return text

    except sr.UnknownValueError:

        print("Speech not understood")

        return None