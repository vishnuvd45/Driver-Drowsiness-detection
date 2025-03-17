import cv2
import dlib
import numpy as np
from sklearn.metrics import pairwise_distances
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import time
import pygame

# Load the pre-trained face and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize pygame mixer
pygame.mixer.init()

# Load the audio file
audio_file = "beep11.mp3"
pygame.mixer.music.load(audio_file)

# Function to compute eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    eye = eye.reshape(-1, 2)  
    A = pairwise_distances(eye[1].reshape(1, -1), eye[5].reshape(1, -1))[0][0]
    B = pairwise_distances(eye[2].reshape(1, -1), eye[4].reshape(1, -1))[0][0]
    C = pairwise_distances(eye[0].reshape(1, -1), eye[3].reshape(1, -1))[0][0]
    ear = (A + B) / (2.0 * C)
    return ear

# Function to check if the person is drowsy
def is_drowsy(landmarks):
    left_eye = landmarks[42:48]
    right_eye = landmarks[36:42]
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    avg_ear = (left_ear + right_ear) / 2.0
    return avg_ear < 0.25  

# Function to get the system volume
def get_system_volume():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    return volume

# Function to set the system volume to 100%
def set_max_volume(volume):
    volume.SetMasterVolumeLevelScalar(1.0, None)

# Function to restore the system volume to its previous state
def restore_volume(volume, prev_volume):
    volume.SetMasterVolumeLevelScalar(prev_volume, None)

# Open the webcam
cap = cv2.VideoCapture(0)

# Set the initial volume
current_volume = get_system_volume()
prev_volume = current_volume.GetMasterVolumeLevelScalar()

# Variables for tracking eye closure duration and audio playback state
eye_closed_start_time = None
audio_playing = False

while True:
    ret, frame = cap.read()

    # Detect faces in the frame
    faces = detector(frame)
    
    # Find the closest face to the camera
    closest_face = None
    min_distance = float('inf')
    for face in faces:
        landmarks = predictor(frame, face)
        landmarks = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(68)])
        
        # Calculate the distance from the camera to the face
        distance = np.linalg.norm(landmarks.mean(axis=0) - np.array([frame.shape[1] / 2, frame.shape[0] / 2]))
        
        if distance < min_distance:
            min_distance = distance
            closest_face = face

    if closest_face:
        landmarks = predictor(frame, closest_face)
        landmarks = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(68)])
        
        left_eye_rect = cv2.boundingRect(np.array([landmarks[42:48]]))
        right_eye_rect = cv2.boundingRect(np.array([landmarks[36:42]]))

        cv2.rectangle(frame, (left_eye_rect[0], left_eye_rect[1]),
                      (left_eye_rect[0] + left_eye_rect[2], left_eye_rect[1] + left_eye_rect[3]), (0, 255, 0), 2)
        cv2.rectangle(frame, (right_eye_rect[0], right_eye_rect[1]),
                      (right_eye_rect[0] + right_eye_rect[2], right_eye_rect[1] + right_eye_rect[3]), (0, 255, 0), 2)

        if is_drowsy(landmarks):
            if eye_closed_start_time is None:
                eye_closed_start_time = time.time()
            else:
                if time.time() - eye_closed_start_time >= 3 and not audio_playing:
                    set_max_volume(current_volume)
                    pygame.mixer.music.play(-1)
                    audio_playing = True
        else:
            if audio_playing:
                restore_volume(current_volume, prev_volume)
                pygame.mixer.music.stop()
                audio_playing = False
            eye_closed_start_time = None

    volume_bar_width = 20
    volume_bar_length = int(frame.shape[1] * 0.05)
    volume_bar_height = int(frame.shape[0] * current_volume.GetMasterVolumeLevelScalar())
    cv2.rectangle(frame, (10, frame.shape[0] - volume_bar_height), (10 + volume_bar_width, frame.shape[0]), (0, 255, 0), -1)
    cv2.rectangle(frame, (10, frame.shape[0] - volume_bar_height), (10 + volume_bar_width, frame.shape[0]), (255, 0, 0), 1)

    percentage = int(current_volume.GetMasterVolumeLevelScalar() * 100)
    cv2.putText(frame, f'Volume: {percentage}%', (15 + volume_bar_width, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('Drowsiness Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


