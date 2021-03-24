import json
import cv2
import numpy as np
import time
import requests
import pyrebase
import io
import zlib
from flask import Flask, request,jsonify    
import os


def compress_nparr(nparr):

    bytestream = io.BytesIO()
    np.save(bytestream, nparr)
    uncompressed = bytestream.getvalue()
    compressed = zlib.compress(uncompressed)
    return compressed, len(uncompressed), len(compressed)

def uncompress_nparr(bytestring):
  
    return np.load(io.BytesIO(zlib.decompress(bytestring)),allow_pickle=True)



config = {
    "apiKey": "AIzaSyC-JHZJZG85fvGZeDanuyhK4161WM5fPGU",
    "authDomain": "blind-vision-5bbf3.firebaseapp.com",
    "projectId": "blind-vision-5bbf3",
    "storageBucket": "blind-vision-5bbf3.appspot.com",
    "messagingSenderId": "227861990159",
    "appId": "1:227861990159:web:33d6c65a32d79050d54c7a",
    "measurementId": "G-D4QETNN03T",
    "databaseURL":"https://console.firebase.google.com/u/1/project/blind-vision-5bbf3/storage/blind-vision-5bbf3.appspot.com/files"
}

class YoloDetection:
  def __init__(self,weights_file,cfg_file,labels_file,firebase_config_file):
    self.net = cv2.dnn.readNet(weights_file, cfg_file)
    self.classes = []
    self.firebase = pyrebase.initialize_app(firebase_config_file)
    self.storage = self.firebase.storage()
    with open(labels_file, "r") as f:
      self.classes = [line.strip() for line in f.readlines()]
    self.layer_names = self.net.getLayerNames()
    self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]  
   
  def detectObjects(self,img):
    height, width, channels = img.shape
    #img = cv2.resize(img, None, fx=0.4, fy=0.4)
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    self.net.setInput(blob)
    outs = self.net.forward(self.output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.1:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w,h = int(detection[2] * width),int(detection[3] * height)
                x,y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indices =  cv2.dnn.NMSBoxes(boxes, confidences, 0.2,0.4)
    present_objects = []
    objects_coordinates = []
    confidences_values = []
    for i in indices:
      i = i[0]
      present_objects.append(self.classes[class_ids[i]])
      objects_coordinates.append(boxes[i])
      confidences_values.append(confidences[i])
    
    return present_objects,objects_coordinates,confidences_values
  
  
  def getdetails_from_firbase(self,camera_angle_of_view):
    start = time.time()
    self.storage.child("input_image.jpg").download(path="input_image.jpg",filename='input_image.jpg')
    img = cv2.imread("input_image.jpg")
    #print("download and read",time.time()-start)
    present_objects,objects_coordinates,confidences_values = self.detectObjects(img)
    height,width,_ = img.shape
    objects_angles = []
    for x,y,w,h in objects_coordinates:
      angle = (((x+w/2)-(width/2))/(width/2)) *(camera_angle_of_view/2)
      objects_angles.append(angle)
    output = {}
    for i,ob in enumerate(present_objects):
      output[str(i)]={"object":ob,"coordinates":objects_coordinates[i],"confidence":confidences_values[i],"angle":objects_angles[i]}
    with open('output.json', 'w') as fp:
        json.dump(output, fp)
    self.storage.child("output.json").put("output.json")
    return present_objects,objects_coordinates,confidences_values,objects_angles

  def getdetails(self,img,camera_angle_of_view=62.2):
    present_objects,objects_coordinates,confidences_values = self.detectObjects(img)
    height,width,_ = img.shape
    objects_angles = []
    for x,y,w,h in objects_coordinates:
      angle = (((x+w/2)-(width/2))/(width/2)) *(camera_angle_of_view/2)
      objects_angles.append(angle)
    output = {}
    for i,ob in enumerate(present_objects):
      output[str(i)]={"object":ob,"coordinates":objects_coordinates[i],"confidence":confidences_values[i],"angle":objects_angles[i]}
    with open('output.json', 'w') as fp:
        json.dump(output, fp)
    cv2.imwrite('input_image.jpg',img)
    self.storage.child('input_image.jpg').put('input_image.jpg')
    self.storage.child("output.json").put("output.json")
    
    return present_objects,objects_coordinates,confidences_values,objects_angles



yolodetect = YoloDetection('yolov4-tiny.weights','yolov4-tiny.cfg','coco.names',config)

#print(yolodetect.getdetails_from_firbase(60))

app = Flask(__name__)   


@app.route('/',methods=['GET'])                                 
def index():     
    return 'SERVER IS ON'     



@app.route('/detectimagefromfirebase',methods=['POST'])            
def detectimagefromfirebase():
    global yolodetect                                           
    result = yolodetect.getdetails_from_firbase(62.2)
    return '{}\n'.format(result) 


@app.route('/detectimage',methods=['POST'])            
def detectimage():
    global yolodetect     
    #data = request.json
    #img = np.array(eval(data['img']))
    #img = json.load(request.files['image_file'])
    #print(img) 
    img = uncompress_nparr( request.data)

    #img = cv2.resize(img, None, fx=0.4, fy=0.4)            
    result = yolodetect.getdetails(img,62.2)
    
    return '{}\n'.format(result) 



@app.route('/updatefirebase',methods=['POST'])            
def updatefirebase():                                           
    posted_data = request.json           
    global yolodetect                                          
    try:
        yolodetect.firebase = pyrebase.initialize_app(posted_data)
        return 'FIREBASE CONFIG FILE CHANGED'
    except Exception as e:
        return e  


if __name__ == '__main__':
    app.run(debug=True)
