from PyQt5.QtGui import QImage, QPixmap
from PyQt5.uic import loadUi
from PyQt5.QtCore import pyqtSlot, QTimer
from PyQt5.QtWidgets import QDialog, QDoubleSpinBox, QSpinBox
import cv2
import dlib
from scipy.spatial import distance
from win10toast import ToastNotifier


class Ui_OutputDialog(QDialog):
    def __init__(self):
        super(Ui_OutputDialog, self).__init__()
        loadUi("./outputwindow.ui", self)

        self.image = None
        self.setStyleSheet("background-image: url(./Images/gradblur.jpg)")
        self.imgLabel.setStyleSheet("border: 1px solid black")
        self.pushButton.clicked.connect(self.hideWindow)

        # finding children
        self.strainRatio = 0.24
        self.warningTime = 60
        self.ratioSpin = self.findChild(QDoubleSpinBox, "doubleSpinBox")
        self.ratioSpin.valueChanged.connect(self.spinSelected)
        self.warningSpin = self.findChild(QSpinBox, "spinBox")
        self.warningSpin.valueChanged.connect(self.spinSelected)

    def spinSelected(self):
        self.strainRatio = self.ratioSpin.value()
        self.warningTime = self.warningSpin.value()

    def hideWindow(self):
        self.showMinimized()

    @pyqtSlot()
    def startVideo(self, camera_name):

        if len(camera_name) == 1:
            self.capture = cv2.VideoCapture(int(camera_name))
        else:
            self.capture = cv2.VideoCapture(camera_name)
        self.timer = QTimer(self)

        self.timer.timeout.connect(self.updateFrame)
        self.timer.start(100)

    def calculateEAR(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        EAR = (A + B) / (2 * C)
        return EAR

    faceDetector = dlib.get_frontal_face_detector()
    landmarkDetector = dlib.shape_predictor("./Images/shapepredictor68.dat")
    strainCount = 0.0

    def updateFrame(self):
        ret, self.image = self.capture.read()
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        faces = self.faceDetector(gray)

        for face in faces:
            faceLandmarks = self.landmarkDetector(gray, face)
            leftEye = []
            rightEye = []

            for i in range(36, 42):
                x = faceLandmarks.part(i).x
                y = faceLandmarks.part(i).y
                leftEye.append((x, y))
                nextPoint = i + 1
                if i == 41:
                    nextPoint = 36
                x2 = faceLandmarks.part(nextPoint).x
                y2 = faceLandmarks.part(nextPoint).y
                cv2.line(self.image, (x, y), (x2, y2), (0, 255, 0), 1)

            for i in range(42, 48):
                x = faceLandmarks.part(i).x
                y = faceLandmarks.part(i).y
                rightEye.append((x, y))
                nextPoint = i + 1
                if i == 47:
                    nextPoint = 42
                x2 = faceLandmarks.part(nextPoint).x
                y2 = faceLandmarks.part(nextPoint).y
                cv2.line(self.image, (x, y), (x2, y2), (0, 255, 0), 1)

            leftEAR = self.calculateEAR(leftEye)
            rightEAR = self.calculateEAR(rightEye)

            averageEAR = round((leftEAR + rightEAR) / 2, 2)

            if averageEAR < self.strainRatio:
                self.strainCount += 0.1
                cv2.putText(self.image, "Eye Strain", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                cv2.putText(self.image, str(int(self.strainCount)), (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                if self.strainCount >= self.warningTime:
                    ToastNotifier().show_toast("Eye Strain Detected", "Maybe take a break?", threaded=True)
                    self.strainCount = 0
                # print("Eye Strain")
            # print(averageEAR)
            self.lcdNumber.display(averageEAR)
        self.displayImage(self.image, 1)

    def displayImage(self, image, window=1):

        image = cv2.resize(image, (640, 480))
        qformat = QImage.Format_Indexed8
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        outImage = QImage(image, image.shape[1], image.shape[0], image.strides[0], qformat)
        outImage = outImage.rgbSwapped()

        if window == 1:
            self.imgLabel.setPixmap(QPixmap.fromImage(outImage))
            self.imgLabel.setScaledContents(True)
