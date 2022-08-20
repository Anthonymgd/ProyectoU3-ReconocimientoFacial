import time
import os
import numpy
import sys
import cv2
from flask import Flask, render_template, Response
from keras.models import load_model
from PIL import Image

app = Flask(__name__)

@app.route('/')
def index():
    # Video streaming home page
    return render_template('index.html')


size = 2  # change this to 4 to speed up processing trade off is the accuracy
classifier = 'haarcascade_frontalface_default.xml'
names = ['Anthony Goyes', 'Ariel Chabla', 'Bryan Solorzano', 'Davila Raymond', 'Fernando Masache', 'Genesis Heredia', 'Hector Cedenio', 'Jhon Zambrano', 'Joan Cevallos', 'Jordan Espinosa', 'Jorge Borrero', 'Kevin Paute', 'Leonardo Borja', 'LucioCarlos', 'Lucy Mosquera', 'Luis Olalla', 'Maria Jose Parraga', 'Melany Lopez', 'Mercy Arrobo', 'Nataly Acosta', 'RuizJose', 'SalazarJohana', 'Selena Enriquez', 'Selena Rivas', 'Solano Wilmer', 'Steven Barragan']
(im_width, im_height) = (100, 100)
count_max = 40
count = 0

# Create a Numpy array from the two lists above
model = load_model('cnn.h5', compile = True)
haar_cascade = cv2.CascadeClassifier(classifier)
webcam = cv2.VideoCapture()  # 0 to use webcam
def process():  
    while count < count_max:
        # Loop until the camera is working
        rval = False
        while(not rval):
            # Put the image from the webcam into 'frame'
            (rval, frame) = webcam.read()
            if(not rval):
                print("No se ha podido iniciar la cámara. Intentando de nuevo...")
        height, width, channels = frame.shape
        startTime = time.time()
        frame = cv2.flip(frame, 1) # 0 = horizontal ,1 = vertical , -1 = both
        # Convert to grayscalel
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize to speed up detection (optinal, change size above)
        mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))
        print(mini.shape)
        print('mini'+str(mini.shape))
        # Detect faces and loop through each one
        faces = haar_cascade.detectMultiScale(mini)
        for i in range(len(faces)):
            face_i = faces[i]
            # Coordinates of face after scaling back by size
            (x, y, w, h) = [v * size for v in face_i]
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (im_width, im_height))
            start =(x, y)
            end =(x + w, y + h)
            # Try to recognize the face
            prediction = model.predict(face_resize)
            # creating a bounding box for detected face
            cv2.rectangle(frame, start, end, (0, 255, 0), 3)
            # creating  rectangle on the upper part of bounding box
            cv2.rectangle(frame, (start[0], start[1]-20),
                        (start[0]+120, start[1]), (0, 255, 255), -3)
            # for i in prediction[1]
            # Remove false positives
            if(w * 6 < width or h * 6 < height):
                cv2.putText('El rostro no se encuentra en la base de datos.')
            else:
                # note: 0 is the perfect match  the higher the value the lower the accuracy
                if prediction[1] < 90:
                    cv2.putText(frame, '%s - %.0f' % (names[prediction[0]], prediction[1]),
                                (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), thickness=2)
                    print('%s - %.0f' % (names[prediction[0]], prediction[1]))
                else:
                    cv2.putText(frame, ("Unknown {} ".format(str(int(prediction[1])))), (
                        x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=2)
                    print("Unknown -", prediction[1])
        endTime = time.time()
        fps = 1/(endTime-startTime)
        cv2.rectangle(frame, (30, 48), (130, 70), (0, 0, 0), -1)
        cv2.putText(frame, "Fps : {} ".format(str(int(fps))), (34, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        # Show the image and check for "q" being pressed
        # compress and store image to memory buffer
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        # concat frame one by one and return frame
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    webcam.release()
    cv2.destroyAllWindows()
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host='localhost',port='5000', debug=False,threaded = True)