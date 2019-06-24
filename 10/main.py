import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor(img):
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces = face_classifier.detectMultiScale(gray, 1.3, 5)
	
	if faces is ():
		return None
		
	# crop faces
	for (x, y, w, h) in faces:
		cropped_face = img[y:y+h, x:x+w]
		
	return cropped_face
	
cam = cv2.VideoCapture(0)	
count = 0	
	
# collect 100 train data from cam
while True:
	ret, frame = cam.read()
	if face_extractor(frame) is not None:
		count += 1
		face = cv2.resize(face_extractor(frame), (200, 200))
		face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
	
		# save file
		try:
			file_path = './face/' + str(count) + '.jpg'
		except:
			os.mkdir('./face')
			file_path = './face/' + str(count) + '.jpg'
		cv2.imwrite(file_path,face)
	
		#curr count
		cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
		cv2.imshow('Count', face)
	else:
		print('Face not found')
		pass
		
	if cv2.waitKey(1) == 13 or count == 100:
		break

cam.release()
cv2.destroyAllWindows()
print('Collecting Samples Complete')

data_path = './face/'	
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]	
	
#array for training data and labels
train_data, label = [], []

for i, files in enumerate(onlyfiles):
	images_path = data_path + onlyfiles[i]
	images = cv2.imread(images_path, cv2.IMREAD_GRAYSCALE)
	train_data.append(np.asarray(images, dtype = np.uint8))
	label.append(i)
	
#NUMPY array for training data and labels
labels = np.asarray(label, dtype = np.int32)	
	
model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(train_data),np.asarray(labels))	

def face_detector(image, size = 0.5):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = face_classifier.detectMultiScale(gray, 1.3, 5)
	
	if faces is ():
		return image, []
	
	for (x, y, w, h) in faces:
		cv2.rectangle(image, (x, y), (x+w,y+h), (0, 255, 255), 2)
		roi = image[y:y+h, x:x+w]
		roi = cv2.resize(roi, (200, 200))
	
	return image,roi
	
cam = cv2.VideoCapture(0)	
	
while True:
	ret,frame = cam.read()
	image, face = face_detector(frame)
	
	try:
		face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
		
		#pass face to prediction model
		results = model.predict(face)
		
		#"results" comprises of a tuple containing the label and the confidence value
		if results[1] < 500:
			confidence = int(100 * (1 - (results[1])/300))
			conf_percent = str(confidence) + ' % confidence it is the user'
		cv2.putText(image, conf_percent, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 120, 150), 2)
		
		if confidence > 75:
			cv2.putText(image, 'Unlocked', (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
			cv2.imshow('Face Recognition Testing', image)
			
		else:
			cv2.putText(image, 'Locked', (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
			cv2.imshow('Face Recognition Testing', image)
	
	except:
		cv2.putText(image, 'No Face Found', (220, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
		cv2.putText(image, 'Locked', (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
		cv2.imshow('Face Recognition Testing', image)
	
	if cv2.waitKey(1) == 13:
		break

cam.release()
cv.destroyAllWindows()	