'''
Created on 22 sie 2020

@author: spasz
'''
from ObjectDetectors.yolov4 import darknet


class DetectorYOLOv4():
    '''
    classdocs
    '''

    def __init__(self, cfgPath, weightPath, metaPath):
        '''
        Constructor
        '''
        self.net, self.classes, self.colors = darknet.load_network(
            cfgPath, metaPath, weightPath)
        self.image = None

    def Detect(self, image, confidence=0.5, nms_thresh=0.45):
        ''' Detect objects in given image'''
        # Create image object we will use each time
        if (self.image is None):
            imheight, imwidth, channels = image.shape
            # Create an image we reuse for each detect
            self.image = darknet.make_image(imwidth, imheight, channels)

        # Detect objects
        darknet.copy_image_from_bytes(self.image, image.tobytes())
        detections = darknet.detect_image(
            self.net, self.classes, self.image, thresh=confidence, nms=nms_thresh)

        # Separate to 3 lists
        labels = []
        confs = []
        boxes = []
        for label, conf, box in detections:
            labels.append(label)
            confs.append(float(conf))
            boxes.append(darknet.bbox2points(box))

        return boxes, labels, confs

    def Draw(self, image, bboxes, labels, confidences):
        ''' Draw all boxes on image'''
        return darknet.draw_boxes(image, bboxes, labels, confidences, self.colors)

    def GetClassNames(self):
        ''' Returns all class names.'''
        return self.classes

    def GetAllowedClassNames(self):
        ''' Returns all interesing us class names.'''
        return self.classes
