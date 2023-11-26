Overview

THe Squirrel Buster  is an attempt to incorporate the main elements
of a machine learning project from start to finish. Every step is
available through a simple web interface. While it was designed
to be a learning exercise, it could be used by someone to train
a model and run inference without writing any code.

There are four key components to the project. They are Collect,
Annotate, Train and Detect. In the Collect section, a user
can use a web camera to take photos of squirrels. The idea
here was to allow someone to train a model in their local 
environment, which should provide better accuracy.

The Annotation phase allows a user to draw bounding boxes
around the objects they want to detect. It is a primitive
approach that only allows for a single class, but it works
entirely in the browser and is effective.

In the Train section, the number of epochs can be defined
as well as the dataset split between train, validate and
test. I used the Ultralytics package train with Yolo8.
This is where the bulk of the code exists. The Collect
and Annotate sections save data in Coco format, so the
first task is to create the Yolo directory structure
and annotation files.

To track status, I use callbacks to get messages
through the process. When the model is complete,
two of the images from the run folder are presented.

Finally, the Detect phase will run inference on frames
from the web camera and highlight detected squirrels
with a bounding box.

The next version of the project will integrate a relay
that will trigger a water gun.