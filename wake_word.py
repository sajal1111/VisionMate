import pvporcupine
import sounddevice as sd
import numpy as np
import winsound

ACCESS_KEY = "PsyxMbAhz9EHxtDf7SFsziJ5L5OQpo5DvrtfrWMp8KzBttutPakopw=="

porcupine = pvporcupine.create(
    access_key=ACCESS_KEY,
    keyword_paths=["hello_camera_windows.ppn"]
)

def wait_wake_word():

    print("Waiting for wake word...")

    with sd.RawInputStream(
        samplerate=porcupine.sample_rate,
        blocksize=porcupine.frame_length,
        dtype="int16",
        channels=1
    ) as stream:

        while True:

            pcm, _ = stream.read(porcupine.frame_length)
            pcm = np.frombuffer(pcm, dtype=np.int16)

            result = porcupine.process(pcm)

            if result >= 0:

                print("Wake word detected!")

                # beep AFTER detection
                winsound.Beep(1000, 200)

                return