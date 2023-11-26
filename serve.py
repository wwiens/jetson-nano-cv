
#******************************************************************
#*  ai3dge
#*  Last updated: 2023-11-25
#*  License: MIT License (see LICENSE.TXT)
#******************************************************************
from ultralytics import YOLO
from flask import Flask, jsonify, request, Response
from flask import render_template
from flask_socketio import SocketIO
from flask_cors import CORS, cross_origin
import json
import numbers
import os
from datetime import datetime
import cv2
import time
import uuid
import shutil
from pathlib import Path
import yoloutils
import math

#* Retrieve arguments and send back value or default if not found
#******************************************************************
def getargs(args, argname, defaultvalue):
    if argname in args:
        return int(args.get(argname))
    else:
        return defaultvalue


#* Load the COCO JSON file
#******************************************************************
if os.path.exists('coco.json'):
    with open("coco.json", "r") as f:
        coco_data = json.load(f)


#* Extract the header info and list of images and annotations
#******************************************************************
info = coco_data["info"]
licenses = coco_data["licenses"]
categories = coco_data["categories"]
images = coco_data["images"]
annotations = coco_data["annotations"]
camera = None

#* Create a list with the annotation IDs
#******************************************************************
imagelist = []
maximageid = 0
maxannotationid = 0
for annotation in annotations:
    imagelist.append(annotation["id"])
    if maximageid < annotation["image_id"]:
        maximageid = annotation["image_id"]
    if maxannotationid < annotation["id"]:
        maxannotationid = annotation["id"]


#* Define variables
#******************************************************************
cameraon = False   
motionon = False
currentepoch = 0
numberofepochs = 1
framecount = 0

model=YOLO("best.pt")

#= Function to save updated annotation information to the JSON file
#==================================================================
def on_train_epoch_end(trainer):
    global currentepoch
    currentepoch = currentepoch + 1
    sendmessage("Epoch " + str(currentepoch) + " of " + str(numberofepochs) + " completed")

def on_pretrain_routine_start(trainer):
    sendmessage("Pretrain routine started")
    
def on_pretrain_routine_end(trainer):
    sendmessage("Pretrain routine ended")

def on_train_start(trainer):
    sendmessage("Training started")

def on_train_epoch_start(trainer):
    sendmessage("Training epoch started")
def teardown(trainer):
    sendmessage("Full results in folder: "+ str(trainer.save_dir))
    filetocopy = str(trainer.save_dir) + "/results.png"
    filedest = "static/images/results.png"
    shutil.copy2(filetocopy, filedest)
    filetocopy = str(trainer.save_dir) + "/confusion_matrix.png"
    filedest = "static/images/confusion_matrix.png"
    shutil.copy2(filetocopy, filedest) 
    filetocopy = str(trainer.save_dir) + "/weights/best.pt"
    filedest = "best.pt"
    shutil.copy2(filetocopy, filedest)     
    sendmessage("Training complete")

#= Function to save updated annotation information to the JSON file
#==================================================================
def sendmessage(statusmsg):
    #? Sends the message to the browser
    socketio.emit('trainstatus', statusmsg)

#= Function to save updated annotation information to the JSON file
#==================================================================
def updateJSON():
    if os.path.exists('coco.json'):
        with open('coco.json', 'r+') as f:
            payload = {
                "info": info,
                "licenses": licenses,
                "categories": categories,
                "images": images,
                "annotations": annotations
            }
            f.seek(0)
            f.truncate()
            json.dump(payload, f)


#= Function to get the annotation image information based on the ID
#==================================================================
def getImage(id):

    # Check to see if ID is in the list
    if id <= len(imagelist)-1:
        imageid = imagelist[id]
    else:
        return False
    
    #? Find the annotation based on the ID we received
    annotation = yoloutils.annotation_search(imageid, annotations)
    
    #? Find the image based on the image_id in the annotation
    image = yoloutils.image_search(annotation["image_id"], images)
    
    #? Return a JSON object with the annotation and image information
    return {
            '[id': annotation["id"], 
            'image': image["file_name"], 
            'bbox': annotation["bbox"], 
            'lastupdated': annotation["last_update"], 
            'imagenumber': id,            
            'imagecount': len(imagelist)
            }


#= Function to get a video frame and send it to the browser
#==================================================================
def gen_frames():  
    while True:
        if cameraon:
            success, frame = camera.read()  # read the camera frame
            if not success:
                print("No success")
                break
            else:
                resize = cv2.resize(frame, (640, 360))
                ret, buffer = cv2.imencode('.jpg', resize)
                if ret:
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 


#= Function to detect motion and send image and result to browser
#==================================================================
def motion():  
    global framecount

    while True:

        if motionon:

            success, frame = camera.read()  # read the camera frame

            if not success:
                    break
            else:
                #resize = cv2.resize(frame, (640, 360))

                results = model.predict(source=frame, save=True, save_txt=True, verbose=False)
                for r in results:
                    boxes=r.boxes
                    for box in boxes:
                        x1,y1,x2,y2=box.xyxy[0]
                        x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
                        #print(x1,y1,x2,y2)
                        
                        conf=math.ceil((box.conf[0]*100))/100
                        if conf > 0.5:
                            cls=int(box.cls[0])
                            class_name = "object"
                            if 1 == 1:
                                cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,255),3)
                                label=f'{class_name}{conf}'
                                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                                cv2.rectangle(frame, (x1,y1), c2, [255,0,255], -1, cv2.LINE_AA)  # filled
                                cv2.putText(frame, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA) 
            

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 

#* Creates a Flask application
#******************************************************************
app = Flask(__name__)
#cors = CORS(app)
socketio = SocketIO(app, cors_allowed_origins='*')
#app.config['CORS_HEADERS'] = 'Content-Type'

#* Home page
#******************************************************************
@app.route("/")
@app.route("/index.html")
def home():
    return render_template('index.html')

#* Collect images page
#******************************************************************
@app.route("/collect.html")
def collect():
    print(annotations)
    return render_template('collect.html')

#* Annotate images page
#******************************************************************
@app.route("/annotate.html")
def annotate():
    return render_template('annotate.html')

#* Train model page
#******************************************************************
@app.route("/train.html")
def train():
    return render_template('train.html')

#* Detect  page
#******************************************************************
@app.route("/detect.html")
def tradetectin():
    return render_template('detect.html')


#* Help page
#******************************************************************
@app.route("/help.html")
def help():
    return render_template('help.html')


#= REST API that will return the information about an image
#==================================================================
@app.route("/getimage")
def getimageinfo():
    #? Get current image information from the web request
    if 'id' in request.args:
        id = int(request.args.get('id')) - 1
        print(id)
    else:
        id = 0

    #? Get image and annotation information to send to the client
    jsonimage = getImage(id)
    if jsonimage == False:
        return {
            'imagenumber': 0,            
            'imagecount': 0
        }
    else:
        return jsonify(jsonimage)


#= Web service that initializes the annotation process
#==================================================================
@app.route("/annotateinit")
def annotateinit():
    numberImages = len(imagelist)
    numberAnnotationsLeft = 0
    numberManualAnnotations = 0
    numberAutoAnnotations = 0
    for obj in annotations:
        if obj["update_method"] == "auto":
            numberAutoAnnotations += 1
        elif obj["update_method"] == "manual":
            numberManualAnnotations += 1
        else:
            numberAnnotationsLeft += 1

    numberAnnotations = numberManualAnnotations + numberAutoAnnotations
    return jsonify({'numberImages': numberImages, 'numberAnnotations': numberAnnotations, 'numberAnnotationsLeft': numberAnnotationsLeft, 'numberManualAnnotations': numberManualAnnotations, 'numberAutoAnnotations': numberAutoAnnotations})


#= Web service that saves annotation changes
#==================================================================
@app.route("/annotation")
def annotation():
    #? Get current image information from the web request
    if 'id' in request.args:
        id = int(request.args.get('id')) - 1
    else:
        return("Invalid ID")

    #? Get the annotation information from the web request
    if 'width' in request.args:
        width = int(request.args.get('width'))
    else:
        width = 0

    if 'height' in request.args:
        height = int(request.args.get('height'))
    else:
        height = 0

    if 'left' in request.args:
        left = int(request.args.get('left'))
    else:
        left = 0

    if 'top' in request.args:
        top = int(request.args.get('top'))
    else:
        top = 0                

    annotations[id]['bbox'] = [left, top, width, height]
    annotations[id]['last_update'] = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now())
    annotations[id]['update_method'] = "manual"
    annotations[id]['area'] = int(width * height)

    #? Call function to save annotation information to the JSON file
    updateJSON()

    #? Return success message to browser
    return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 


#= Web service that deletes an image and annotation
#==================================================================
@app.route("/deleteimage")
def deleteimage():
    #? Get current image information from the web request
    if 'id' in request.args:
        id = int(request.args.get('id')) - 1
    else:
        return("Invalid ID")
    print(id)
    #? Find the image based on the image_id in the annotation
    image = yoloutils.image_search(annotations[id]["image_id"], images)

    #? Delete the image and annotation
    filename = image["file_name"]
    imageindex = images.index(image)
    del images[imageindex]
    del annotations[id]
    del imagelist[id]

    os.remove('static/dataset/' + filename)

    #? Call function to save annotation information to the JSON file
    updateJSON()

    #? Return success message to browser
    return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 


#= Function to send video feed back to the web page to display
#==================================================================
@app.route('/video_feed')
def video_feed():
    global cameraon, camera
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


#= Function to turn off the video feed
#==================================================================
@app.route('/vidoff')
def vidoff():
    global cameraon, camera
    cameraon  = False
    if camera is not None:
        camera.release()
        camera = None


    return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 


#= Function to turn on the video feed
#==================================================================
@app.route('/vidon')
def vidon():
    global cameraon, camera
    #camera = cv2.VideoCapture(0)                   # Use this on the Jetson Nano
    camera = cv2.VideoCapture(1)                    # Use this on Windows for a USB Camera
    #camera = cv2.VideoCapture(cv2.CAP_DSHOW)       # Use this on Windows for a built-in camera
    time.sleep(2)
    cameraon = True
    return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 


#= Function to send video feed back to the web page to display
#==================================================================
@app.route('/motion_feed')
def motion_feed():
    global motionon, camera
    return Response(motion(), mimetype='multipart/x-mixed-replace; boundary=frame')

#= Function to turn on the video feed for motion detection
#==================================================================
@app.route('/startmotion')
def startmotion():
    global motionon, camera, model
    model=YOLO("best.pt")
    #camera = cv2.VideoCapture(0)                   # Use this on the Jetson Nano
    camera = cv2.VideoCapture(1)                    # Use this on Windows for a USB Camera
    #camera = cv2.VideoCapture(cv2.CAP_DSHOW)       # Use this on Windows for a built-in camera
    time.sleep(2)
    motionon = True
    return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 


#= Function to turn off the video feed
#==================================================================
@app.route('/stopmotion')
def stopmotion():
    global motionon, camera
    motionon  = False
    if camera is not None:
        camera.release()
        camera = None
    return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 


#= Function to save an image from the video feed to the dataset
#==================================================================
@app.route('/snapphoto')
def snapphoto():
    global maximageid, maxannotationid
    success, frame = camera.read()  # read the camera frame

    if not success:
        return json.dumps({'success':False}), 500, {'ContentType':'application/json'} 
    else:
        filedate = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        base_filename = filedate + str(uuid.uuid4())[:13] + '.png'
        unique_filename = 'static/dataset/' + base_filename
        resize = cv2.resize(frame, (640, 360))
        cv2.imwrite(unique_filename, resize)

        #? Increment the image ID by one to add a new image
        maximageid += 1

        #? Define all the JSON objects for the new image
        newimage = {
            "id": maximageid, 
            "license": 1, 
            "file_name": base_filename, 
            "height": 360, 
            "width": 640, 
            "date_captured": '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now())
        }

        #? Append the new image to the images list
        images.append(newimage)

        #? Increment the annotation ID by one to add a new image
        maxannotationid += 1

        #? Define all the JSON objects for the new annotation
        newannotation = {
            "id": maxannotationid, 
            "image_id": maximageid, 
            "category_id": 1, 
            "bbox": [200, 20, 250, 300],
            "area": 2500, 
            "segmentation": [], 
            "iscrowd": 0, 
            "last_update": '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()), 
            "update_method": "created"
        }

        #? Append the new annotation to the annotations list
        annotations.append(newannotation)
        imagelist.append(newannotation["id"])

        #? Call function to save annotation information to the JSON file
        updateJSON()


        #? Return success message to browser
        return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 

#= Web service that executes the training
#==================================================================
@app.route("/beginttraining")
def beginttraining():
    global currentepoch, numberofepochs
    currentepoch = 0
    print("DStart training")

    #? Get variables from request
    numberofepochs = getargs(request.args, 'epochs', 5)
    rng_train = getargs(request.args, 'train', 70)
    rng_valid = getargs(request.args, 'validate', 20)
    
    #? Determine number of files in each set
    trainsize = int(len(imagelist) * (rng_train/100))
    validsize = int(len(imagelist) * (rng_valid/100))

    #? Function the create yolo directories and files
    yoloutils.create_yolo_directories(imagelist, annotations, images, trainsize, validsize)
    print("Directories created")

    #? Load a pretrained YOLO model (recommended for training)
    sendmessage("Loading pretrained YOLO model")
    model = YOLO('yolov8n.pt')

    #? Add callbacks - these will let us send updates back to the browser
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_pretrain_routine_start", on_pretrain_routine_start)
    model.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)
    model.add_callback("on_train_start", on_train_start)
    model.add_callback("teardown", teardown)

    #? Train the model
    results = model.train(data='data.yaml', epochs=numberofepochs, verbose=False)

    #? Return a message to the browser
    return jsonify({'status': 'Training complete'})


#= Function to run inference
#==================================================================
@app.route("/trainme")
def trainme():
    print("DStart training")
    return jsonify({'status': 'Training complete'})


#= Function to run inference
#==================================================================
@app.route("/detect")
def detect():
    return render_template('detect.html')



#= Start up the application
#==================================================================
if __name__ == "__main__":
    app.run(debug=True)