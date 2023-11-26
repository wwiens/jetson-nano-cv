import os
import shutil
from pathlib import Path

#= Function to search for an annotation based on the ID
#==================================================================
def annotation_search(id, annotations):
    for annotation in annotations:
        if annotation['id'] == id:
            return annotation


#= Function to search for an image based on the ID
#==================================================================
def image_search(id, images):
    for image in images:
        if image['id'] == id:
            return image
        
def convert_coco_to_yolo(coco_coords, image_width, image_height):
    # CoCo coordinates [xmin, ymin, width, height]
    xmin, ymin, width, height = coco_coords
    
    # Calculate center coordinates 
    center_x = (xmin + width/2) / image_width
    center_y = (ymin + height/2) / image_height
    
    # Calculate relative width and height
    relative_width = width / image_width
    relative_height = height / image_height
    
    # YoLo coordinates [center_x, center_y, relative_width, relative_height]
    yolo_coords = [center_x, center_y, relative_width, relative_height]
    #print(yolo_coords)
    return yolo_coords

def create_yolo_directories(imagelist, annotations, images, trainsize, validsize):
    trainlist = imagelist[:trainsize]
    validlist = imagelist[trainsize:trainsize+validsize]
    testlist = imagelist[trainsize+validsize:]

    #? Create YOLO directories and/or make sure they are empty
    if os.path.exists("train"):
        shutil.rmtree("train")
    if os.path.exists("valid"):
        shutil.rmtree("valid")
    if os.path.exists("test"):
        shutil.rmtree("test")                
    Path("train/images").mkdir(parents=True, exist_ok=True)
    Path("train/labels").mkdir(parents=True, exist_ok=True)
    Path("valid/images").mkdir(parents=True, exist_ok=True)
    Path("valid/labels").mkdir(parents=True, exist_ok=True)
    Path("test/images").mkdir(parents=True, exist_ok=True)
    Path("test/labels").mkdir(parents=True, exist_ok=True)

    #? Copy images to YOLO directories
    for annotation in annotations:


        if annotation["id"] in trainlist:
            #? Copy images to the image folder
            imagetocopy = image_search(annotation["image_id"], images)            
            shutil.copy2("static/dataset/" + imagetocopy["file_name"], "train/images")

            #? Convert Coco coordinates to Yolo coordinates
            yolocoords = convert_coco_to_yolo(annotation["bbox"], imagetocopy["width"], imagetocopy["height"])
            yoloout = "0 " + str(yolocoords[0]) + " " + str(yolocoords[1]) + " " + str(yolocoords[2]) + " " + str(yolocoords[3])
            filename = "train/labels/" + imagetocopy["file_name"].replace(".jpg", ".txt")
            filename = "train/labels/" + imagetocopy["file_name"].replace(".png", ".txt")
            with open(filename, "w") as f:
                f.write(yoloout)

        elif annotation["id"] in validlist:
            #? Copy images to the image folder
            imagetocopy = image_search(annotation["image_id"], images)            
            shutil.copy2("static/dataset/" + imagetocopy["file_name"], "valid/images")
            
            #? Convert Coco coordinates to Yolo coordinates            
            yolocoords = convert_coco_to_yolo(annotation["bbox"], imagetocopy["width"], imagetocopy["height"])
            yoloout = "0 " + str(yolocoords[0]) + " " + str(yolocoords[1]) + " " + str(yolocoords[2]) + " " + str(yolocoords[3])
            filename = "valid/labels/" + imagetocopy["file_name"].replace(".jpg", ".txt")
            filename = "valid/labels/" + imagetocopy["file_name"].replace(".png", ".txt")
            with open(filename, "w") as f:
                f.write(yoloout)

        elif annotation["id"] in testlist:
            #? Copy images to the image folder
            imagetocopy = image_search(annotation["image_id"], images)            
            shutil.copy2("static/dataset/" + imagetocopy["file_name"], "test/images")
            
            #? Convert Coco coordinates to Yolo coordinates            
            yolocoords = convert_coco_to_yolo(annotation["bbox"], imagetocopy["width"], imagetocopy["height"])
            yoloout = "0 " + str(yolocoords[0]) + " " + str(yolocoords[1]) + " " + str(yolocoords[2]) + " " + str(yolocoords[3])
            filename = "test/labels/" + imagetocopy["file_name"].replace(".jpg", ".txt")
            filename = "test/labels/" + imagetocopy["file_name"].replace(".png", ".txt")
            with open(filename, "w") as f:
                f.write(yoloout)          