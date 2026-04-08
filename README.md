# VisionMate

> A real-time assistive vision system built for visually impaired people. Point a camera at the world, ask a question out loud — it answers back.

---

## What it does

You say *"Hello Camera"* — it wakes up, listens to your question, looks at the camera feed, and speaks an answer. Meanwhile, it's quietly watching in the background and warns you if something gets too close.

No buttons. No screen. Just voice and camera.

---

## How it works

There are three things running at the same time:

**1. Navigation (background)**
Every 3 seconds, YOLOv8 scans the frame for objects and MiDaS estimates how far they are. If something's close, it speaks a warning — *"chair very close on your left"*.

**2. Voice interaction (on wake word)**
Say *"Hello Camera"* → it beeps → you ask your question.
- Simple questions (*"is there a table?"*, *"how far is the chair?"*) → answered directly from YOLO + depth data
- Complex questions (*"what's in front of me?"*, *"describe the room"*) → sent to InternVL2, a vision-language model that actually understands the scene

**3. Camera loop**
Continuously reads frames and keeps the latest one ready for both systems above.

---

## Stack

| What | Tool |
|---|---|
| Vision Language Model | InternVL2-1B |
| Object Detection | YOLOv8n |
| Depth Estimation | MiDaS small |
| Wake Word | Porcupine (Picovoice) |
| Speech-to-Text | Google STT via `speech_recognition` |
| Text-to-Speech | pyttsx3 (Windows SAPI5) |

---

## Setup

**Requirements:** Python 3.10+, Windows, CUDA GPU (recommended), mic + webcam

```bash
pip install torch torchvision transformers ultralytics opencv-python \
            pvporcupine sounddevice scipy SpeechRecognition pyttsx3 Pillow numpy
```

Then just run:

```bash
python main.py
```

It'll load the models (takes ~30s first time), then wait for the wake word.

---

## Before you run

- Get a free API key from [picovoice.ai](https://console.picovoice.ai/) and put it in `wake_word.py`
- Don't commit that key — throw it in a `.env` file
- The `.ppn` wake word file is Windows-only. If you're on Linux/Mac, regenerate it on the Picovoice console

---

## File overview

```
main.py               ← starts everything, manages threads
live_internvl.py      ← handles the vision-language model
navigation.py         ← object detection + depth → obstacle info
depth_estimation.py   ← MiDaS wrapper
wake_word.py          ← listens for "Hello Camera"
voice_input.py        ← records mic + converts to text
speech_output.py      ← speaks the response out loud
yolov8n.pt            ← YOLOv8 weights
hello_camera_windows.ppn  ← custom wake word model
```

---

## Why I built this

Standard assistive tools are either too expensive or require constant phone interaction. This runs locally, responds in real time, and needs nothing but a camera and a voice.
