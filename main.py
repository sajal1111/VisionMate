import cv2
import threading
import time

from wake_word import wait_wake_word
from voice_input import listen_question
from speech_output import speak
from navigation import detect_objects_with_depth
from live_internvl import generate_caption


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera error")
    exit()

print("Camera connected")

latest_frame = None
last_warning = None


# -------------------------------
# Navigation Warning System
# -------------------------------

def navigation_warning(objects):

    warnings = []

    for obj in objects:

        label = obj["label"]
        distance = obj["distance"]
        direction = obj.get("direction", "ahead")

        if distance < 0.3:
            warnings.append(f"{label} very close on your {direction}")

        elif distance < 0.6:
            warnings.append(f"{label} ahead on your {direction}")

    return warnings

def navigation_loop():

    global latest_frame
    global last_warning

    while True:

        if latest_frame is None:
            time.sleep(0.1)
            continue

        objects = detect_objects_with_depth(latest_frame)

        warnings = navigation_warning(objects)

        if warnings:

            warning = warnings[0]

            if warning != last_warning:
                speak(warning)
                last_warning = warning

        time.sleep(3)


# -------------------------------
# Question Routing
# -------------------------------

def route_question(question):

    simple = ["is there", "how many", "do you see", "distance", "where"]

    q = question.lower()

    for word in simple:
        if word in q:
            return "simple"

    return "complex"


# -------------------------------
# Simple Question Processing
# -------------------------------

def process_simple(question, objects):

    q = question.lower()

    # -------------------------
    # Distance Question Handling
    # -------------------------
    if "distance" in q:

        for obj in objects:

            if obj["label"] in q:

                label = obj["label"]
                dist = obj["distance"]
                direction = obj.get("direction", "ahead")

                if dist < 0.3:
                    return f"The {label} is very close on your {direction}"

                elif dist < 0.6:
                    return f"The {label} is about one meter ahead on your {direction}"

                else:
                    return f"The {label} is farther away on your {direction}"

        return "I cannot estimate the distance"

    # -------------------------
    # How many objects
    # -------------------------
    if "how many" in q:

        target = q.split()[-1]

        count = sum(1 for obj in objects if obj["label"] == target)

        return f"I see {count} {target}"

    # -------------------------
    # Is there an object
    # -------------------------
    if "is there" in q:

        target = q.split()[-1]

        for obj in objects:
            if obj["label"] == target:
                direction = obj.get("direction", "ahead")
                return f"Yes, there is a {target} on your {direction}"

        return f"No, I do not see a {target}"

    # -------------------------
    # Where is the object
    # -------------------------
    if "where" in q:

        for obj in objects:
            if obj["label"] in q:
                direction = obj.get("direction", "ahead")
                return f"The {obj['label']} is on your {direction}"

        return "I cannot locate that object"

    return "I cannot answer that"


# -------------------------------
# Voice Interaction Loop
# -------------------------------

def interaction_loop():

    global latest_frame

    while True:

        wait_wake_word()

        speak("Listening")

        question = listen_question()

        if question is None:
            continue

        print("Question:", question)

        mode = route_question(question)

        frame = latest_frame

        if frame is None:
            speak("Camera not ready")
            continue

        objects = detect_objects_with_depth(frame)

        if mode == "simple":

            answer = process_simple(question, objects)

        else:

            answer = generate_caption(frame, question)

        speak(answer)


# -------------------------------
# Camera Loop
# -------------------------------

def camera_loop():

    global latest_frame

    while True:

        ret, frame = cap.read()

        if not ret:
            continue

        latest_frame = frame

        cv2.imshow("Assistive Camera", frame)

        if cv2.waitKey(1) == 27:
            break


# -------------------------------
# Start Threads
# -------------------------------

nav_thread = threading.Thread(target=navigation_loop, daemon=True)
nav_thread.start()

interaction_thread = threading.Thread(target=interaction_loop, daemon=True)
interaction_thread.start()

camera_loop()