from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import cv2
import mediapipe as mp
import numpy as np
import requests

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates/user")

# Load face detection and landmark tracking models
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Fetch filters from the admin app
admin_app_url = "http://localhost:8000"  # Replace with the actual admin app URL
filters = requests.get(f"{admin_app_url}/api/filters").json()["filters"]

# Function to apply filter based on landmarks
def apply_filter(frame, filter_image, landmarks):
    # Get landmark points for positioning the filter (example: using eyes and nose)
    left_eye = (int(landmarks.landmark[133].x * frame.shape[1]), int(landmarks.landmark[133].y * frame.shape[0]))
    right_eye = (int(landmarks.landmark[362].x * frame.shape[1]), int(landmarks.landmark[362].y * frame.shape[0]))
    nose_tip = (int(landmarks.landmark[1].x * frame.shape[1]), int(landmarks.landmark[1].y * frame.shape[0]))

    # Calculate filter size and position based on landmarks
    filter_width = int(abs(left_eye[0] - right_eye[0]) * 2)  # Adjust scaling as needed
    filter_height = int(filter_width * filter_image.shape[0] / filter_image.shape[1])
    filter_x = int(nose_tip[0] - filter_width / 2)
    filter_y = int(nose_tip[1] - filter_height / 1.5)  # Adjust vertical position as needed

    # Resize the filter image
    resized_filter = cv2.resize(filter_image, (filter_width, filter_height))

    # Overlay the filter on the frame using transparency (if available)
    for i in range(filter_height):
        for j in range(filter_width):
            if filter_y + i < frame.shape[0] and filter_x + j < frame.shape[1] and filter_y + i >= 0 and filter_x + j >= 0:
                if resized_filter[i, j, 3] != 0:  # Check alpha channel for transparency
                    frame[filter_y + i, filter_x + j] = resized_filter[i, j, :3]

    return frame

async def video_feed():
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Apply the first filter for now (you can add filter selection later)
                if filters:
                    filter_image = cv2.imread(filters[0]["image_path"], cv2.IMREAD_UNCHANGED)
                    frame = apply_filter(frame, filter_image, face_landmarks)

        _, jpeg_frame = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpeg_frame.tobytes() + b"\r\n")

    video_capture.release()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video_feed")
async def video_feed_endpoint():
    return StreamingResponse(video_feed(), media_type="multipart/x-mixed-replace; boundary=frame")
