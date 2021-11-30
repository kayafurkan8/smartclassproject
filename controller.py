from re import A
import sys
from PyQt5.QtWidgets import * 
from smartClassUi_python import *
from PyQt5.QtGui import QImage, QMovie
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer,pyqtSignal,Qt

import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing.image import img_to_array


model = load_model("model/smartClassModel.hdf5", compile=False)

cv2.ocl.setUseOpenCL(False)

EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised",
            "neutral"]
class appUi(QMainWindow):
    
    changePixmap = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        

        self.timer = QTimer()
     
        self.timer.timeout.connect(self.startCamera)
        
        self.ui.startLesson.clicked.connect(self.controlTimer)
        self.ui.endLesson1.clicked.connect(self.controlTimerEnd)
        
        
    def startCamera(self):
        counter = 0
        sum = 0

        while True:
   
            ret, frame = self.cap.read()
            if not ret:
                break
            facecasc = cv2.CascadeClassifier(
                'face/haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = facecasc.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi = gray[y:y + h, x:x + w]
       
                roi = cv2.resize(roi, (64, 64))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = model.predict(roi)
                maxindex = int(np.argmax(prediction))
                preds = model.predict(roi)[0]
                maxPred = (np.max(preds))
                #for prob in maxPred:
                text = "{:.2f}%".format(maxPred * 100)
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
                cv2.putText(frame, EMOTIONS[maxindex], (x+20, y-60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 10, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, text, (x+50, y-90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 10, 255), 2, cv2.LINE_AA)
                results = {}
                results.update({EMOTIONS[maxindex]:maxPred})

                for i in results:
                    if(i == 'happy'):
                        counter += 1
                        sum = sum + results[i]
                        result = sum/counter
                        self.ui.happyRate.setText(str(result*100))
                    if(i == 'sad'):
                        counter += 1
                        sum = sum + results[i]
                        result = sum/counter
                        self.ui.sadRate.setText(str(result*100))
                    if(i == 'neutral'):
                        counter += 1
                        sum = sum + results[i]
                        result = sum/counter
                        self.ui.neutralRate.setText(str(result*100))
                    if(i == 'angry'):
                        counter += 1
                        sum = sum + results[i]
                        result = sum/counter
                        self.ui.angryRate.setText(str(result*100))
                    if(i == 'scared'):
                        counter += 1
                        sum = sum + results[i]
                        result = sum/counter
                        self.ui.scaredRate.setText(str(result*100))
                    if(i == 'surprised'):
                        counter += 1
                        sum = sum + results[i]
                        result = sum/counter
                        self.ui.suprisedRate.setText(str(result*100))

            qImg = QImage(frame.data,frame.shape[1],frame.shape[0],frame.strides[0], QImage.Format_RGB888)
            capture = QPixmap(qImg)
            self.ui.label.setScaledContents(1) 
            self.changePixmap.connect(self.setCapture(capture))
           
       
    def setCapture(self,capture):
        self.ui.label.setPixmap(capture)
       

    def controlTimer(self):
        if not self.timer.isActive():
            self.cap = cv2.VideoCapture(0)
            self.timer.start()
            
    def controlTimerEnd(self):
        self.timer.stop()
        self.cap.release()
        cv2.destroyAllWindows()
            

            


app = QApplication(sys.argv)
window = appUi()
window.show()
sys.exit(app.exec_())