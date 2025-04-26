# Import necessary libraries
import cv2  # For video capture and drawing
import dlib  # For face detection and landmark detection
import numpy as np  # For numerical operations
from sklearn.metrics import pairwise_distances  # For distance calculation between eye landmarks
from ctypes import cast, POINTER  # For controlling system volume
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume  # For accessing system volume
import time  # For tracking how long eyes are closed
import pygame  # For playing alarm sounds

# Load pre-trained dlib face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize pygame mixer to play sound
pygame.mixer.init()

# Load alarm audio file
audio_file = "beep11.mp3"
pygame.mixer.music.load(audio_file)

# Function to compute Eye Aspect Ratio (EAR) for eye closure detection
def eye_aspect_ratio(eye):
    eye = eye.reshape(-1, 2)
    A = pairwise_distances(eye[1].reshape(1, -1), eye[5].reshape(1, -1))[0][0]
    B = pairwise_distances(eye[2].reshape(1, -1), eye[4].reshape(1, -1))[0][0]
    C = pairwise_distances(eye[0].reshape(1, -1), eye[3].reshape(1, -1))[0][0]
    ear = (A + B) / (2.0 * C)
    return ear

# Function to check if a person is drowsy based on EAR
def is_drowsy(landmarks):
    left_eye = landmarks[42:48]
    right_eye = landmarks[36:42]
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    avg_ear = (left_ear + right_ear) / 2.0
    return avg_ear < 0.25  # Threshold for drowsiness

# Function to get current system volume interface
def get_system_volume():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    return volume

# Function to set system volume to maximum (100%)
def set_max_volume(volume):
    volume.SetMasterVolumeLevelScalar(1.0, None)

# Function to restore system volume to previous level
def restore_volume(volume, prev_volume):
    volume.SetMasterVolumeLevelScalar(prev_volume, None)

# Open the webcam
cap = cv2.VideoCapture(0)

# Save the initial volume level to restore later
current_volume = get_system_volume()
prev_volume = current_volume.GetMasterVolumeLevelScalar()

# Variables to track eye closure time and whether alarm is playing
eye_closed_start_time = None
audio_playing = False

# Main loop: read video frames
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
        
        # Measure distance of face from center of frame
        distance = np.linalg.norm(landmarks.mean(axis=0) - np.array([frame.shape[1] / 2, frame.shape[0] / 2]))
        
        # Update the closest face
        if distance < min_distance:
            min_distance = distance
            closest_face = face

    if closest_face:
        # Predict landmarks for the closest face
        landmarks = predictor(frame, closest_face)
        landmarks = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(68)])
        
        # Draw rectangles around left and right eyes
        left_eye_rect = cv2.boundingRect(np.array([landmarks[42:48]]))
        right_eye_rect = cv2.boundingRect(np.array([landmarks[36:42]]))

        cv2.rectangle(frame, (left_eye_rect[0], left_eye_rect[1]),
                      (left_eye_rect[0] + left_eye_rect[2], left_eye_rect[1] + left_eye_rect[3]), (0, 255, 0), 2)
        cv2.rectangle(frame, (right_eye_rect[0], right_eye_rect[1]),
                      (right_eye_rect[0] + right_eye_rect[2], right_eye_rect[1] + right_eye_rect[3]), (0, 255, 0), 2)

        # Check for drowsiness based on eye closure
        if is_drowsy(landmarks):
            if eye_closed_start_time is None:
                eye_closed_start_time = time.time()
            else:
                # If eyes closed for more than 3 seconds, trigger alarm
                if time.time() - eye_closed_start_time >= 3 and not audio_playing:
                    set_max_volume(current_volume)
                    pygame.mixer.music.play(-1)  # Play alarm sound in loop
                    audio_playing = True
        else:
            # Eyes are open again: stop alarm and restore volume
            if audio_playing:
                restore_volume(current_volume, prev_volume)
                pygame.mixer.music.stop()
                audio_playing = False
            eye_closed_start_time = None

    # Draw volume bar on the frame
    volume_bar_width = 20
    volume_bar_length = int(frame.shape[1] * 0.05)
    volume_bar_height = int(frame.shape[0] * current_volume.GetMasterVolumeLevelScalar())
    cv2.rectangle(frame, (10, frame.shape[0] - volume_bar_height), (10 + volume_bar_width, frame.shape[0]), (0, 255, 0), -1)
    cv2.rectangle(frame, (10, frame.shape[0] - volume_bar_height), (10 + volume_bar_width, frame.shape[0]), (255, 0, 0), 1)

    # Display the current volume percentage
    percentage = int(current_volume.GetMasterVolumeLevelScalar() * 100)
    cv2.putText(frame, f'Volume: {percentage}%', (15 + volume_bar_width, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Show the frame with all drawings
    cv2.imshow('Drowsiness Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
