# ---------------------------------------------------------------- Python Script ----------------------------------------------------------------



import face_recognition
import cv2
import numpy as np
import os

# Directory to save registered faces
REGISTERED_FACES_DIR = "registered_faces"

# Ensure the directory exists
os.makedirs(REGISTERED_FACES_DIR, exist_ok=True)

# Load all registered faces and encodings
def load_known_faces():
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(REGISTERED_FACES_DIR):
        image_path = os.path.join(REGISTERED_FACES_DIR, filename)
        name = os.path.splitext(filename)[0]
       
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(name)

    return known_face_encodings, known_face_names

# Register a new face from the video stream
def register_new_face(frame, name):
    # Resize and process the frame for accurate face encoding
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if face_encodings:
        # Save the first detected face with the given name
        face_image = frame[face_locations[0][0]:face_locations[0][2], face_locations[0][3]:face_locations[0][1]]
        cv2.imwrite(f"{REGISTERED_FACES_DIR}/{name}.jpg", face_image)
        print(f"Face registered for {name}.")

# Real-time face recognition function with registration
def recognize_faces_in_video():
    # Load initial known faces
    known_face_encodings, known_face_names = load_known_faces()

    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture a frame from the webcam
        ret, frame = video_capture.read()
        if not ret:
            break

        # Resize and process the frame
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect and encode faces in the frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            name = "Unknown"

            if known_face_encodings:  # Only proceed if there are known faces
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            face_names.append(name)

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Video', frame)

        # Handle key presses for real-time face registration
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            name = input("Enter the name for the new face: ")
            register_new_face(frame, name)
            known_face_encodings, known_face_names = load_known_faces()  # Reload faces after registering

        elif key == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Start the face recognition system with registration capability
recognize_faces_in_video()





# ---------------------------------------------------------------- Flask App ----------------------------------------------------------------


# from flask import Flask, render_template, Response, request
# import face_recognition
# import cv2
# import numpy as np
# import os
# from io import BytesIO
# from PIL import Image

# app = Flask(__name__)

# # Directory to save registered faces
# REGISTERED_FACES_DIR = "registered_faces"
# os.makedirs(REGISTERED_FACES_DIR, exist_ok=True)

# # Load all registered faces and encodings
# def load_known_faces():
#     known_face_encodings = []
#     known_face_names = []
 
#     for filename in os.listdir(REGISTERED_FACES_DIR):
#         image_path = os.path.join(REGISTERED_FACES_DIR, filename)
#         name = os.path.splitext(filename)[0]
        
#         image = face_recognition.load_image_file(image_path)
#         encodings = face_recognition.face_encodings(image)
#         if encodings:
#             known_face_encodings.append(encodings[0])
#             known_face_names.append(name)

#     return known_face_encodings, known_face_names

# # Register a new face from the frame
# def register_new_face(frame, name):
#     # Resize and process the frame for accurate face encoding
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     face_locations = face_recognition.face_locations(rgb_frame)
#     face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

#     if face_encodings:
#         # Save the first detected face with the given name
#         face_image = frame[face_locations[0][0]:face_locations[0][2], face_locations[0][3]:face_locations[0][1]]
#         cv2.imwrite(f"{REGISTERED_FACES_DIR}/{name}.jpg", face_image)
#         print(f"Face registered for {name}.")

# # Real-time face recognition function with MJPEG streaming
# def generate_frames():
#     # Load initial known faces
#     known_face_encodings, known_face_names = load_known_faces()

#     video_capture = cv2.VideoCapture(0)

#     while True:
#         ret, frame = video_capture.read()
#         if not ret:
#             break

#         # Resize and process the frame
#         small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
#         rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

#         # Detect and encode faces in the frame
#         face_locations = face_recognition.face_locations(rgb_small_frame)
#         face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

#         face_names = []
#         for face_encoding in face_encodings:
#             name = "Unknown"

#             if known_face_encodings:  # Only proceed if there are known faces
#                 matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
#                 face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
#                 best_match_index = np.argmin(face_distances)

#                 if matches[best_match_index]:
#                     name = known_face_names[best_match_index]

#             face_names.append(name)

#         # Display the results
#         for (top, right, bottom, left), name in zip(face_locations, face_names):
#             top *= 4
#             right *= 4
#             bottom *= 4
#             left *= 4

#             cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#             cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

#         # Encode the frame as JPEG and yield it for MJPEG streaming
#         ret, jpeg = cv2.imencode('.jpg', frame)
#         if not ret:
#             continue

#         frame_bytes = jpeg.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

#     video_capture.release()

# # Route to display the live video stream
# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# # Route for face registration
# @app.route('/register_face', methods=['POST'])
# def register_face():
#     if 'file' not in request.files:
#         return "No file part", 400

#     file = request.files['file']
#     name = request.form['name']

#     if file.filename == '':
#         return "No selected file", 400

#     if file:
#         img = Image.open(file.stream)
#         frame = np.array(img)
#         register_new_face(frame, name)
#         return f"Face registered for {name}!", 200

# # Main route to render the HTML page
# @app.route('/')
# def index():
#     return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)
