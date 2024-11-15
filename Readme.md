------------------------------------------------------------------------------------------ face-recognition -----------------------------------------------------------------------------------------

Libraries Used:
face_recognition: Detects and encodes faces in images and video.
cv2 (OpenCV): Handles video input/output, frame processing, and displaying results.
numpy: Used for face distance calculations to find the best match.
os: Manages file paths and directories for saving registered faces.

Key Functionalities:
Loading Known Faces: Scans the registered_faces directory, loads images, extracts face encodings, and associates names with faces based on the image filenames.
Registering New Faces: Allows users to register a new face by capturing it from the live video stream, saving the image with a user-defined name, and updating the known faces for future recognition.
Real-Time Face Recognition: Captures video from the webcam, detects faces, compares them with known faces, and labels them with the corresponding name or "Unknown" if no match is found.
Interactive Registration: Users can press the r key to register new faces, input a name, save the face image, and update the list of known faces. Press q to exit the program.

How It Works:
1) Start the Program: The program initializes by loading previously registered faces from the registered_faces directory.
2) Video Capture: The webcam video is captured and processed frame by frame. In each frame, the faces are detected and encoded.
3) Face Matching: For each detected face, the system compares the face encoding with those of previously registered faces. If a match is found, the face is labeled with the associated name; otherwise, it is marked as "Unknown."
4) Face Registration: The user can press the r key to register a new face, which triggers the register_new_face() function to save the new face image with the user-provided name.
5) Exit: Press the q key to stop the program.

