import pyttsx3
import threading

speech_lock = threading.Lock()

def speak(text):

    with speech_lock:

        try:
            print("AI:", text)

            engine = pyttsx3.init(driverName='sapi5')

            engine.setProperty("rate", 170)

            engine.say(text)

            engine.runAndWait()

            engine.stop()

        except Exception as e:
            print("Speech error:", e)