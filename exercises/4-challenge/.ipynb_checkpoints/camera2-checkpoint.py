import cv2

# Define the GStreamer pipeline
# Adjust the width, height, and framerate as needed
gst_pipeline = ("nvarguscamerasrc ! "
                "video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! "
                "nvvidconv flip-method=2 ! "
                "video/x-raw, width=(int)1280, height=(int)720, format=(string)BGRx ! "
                "videoconvert ! "
                "video/x-raw, format=(string)BGR ! appsink")

# Open a connection to the camera
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Display the resulting frame
    cv2.imshow('Camera Feed', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()