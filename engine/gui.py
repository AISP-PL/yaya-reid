'''
Created on 16 lis 2020

@author: spasz
'''

from engine.annote import GetClasses, GetClassName
from helpers.images import ResizeToWidth, ResizeToHeight
import cv2
import logging
import helpers.boxes as boxes
import numpy as np
from Gui.drawing import DrawText


class Gui(object):
    def __init__(self, name):
        self.image = None
        self.width = None
        self.height = None
        self.winname = name
        self.annoter = None

        # Mouse
        self.lastPos = []
        self.coords = []
        self.dragging = False

    def SetAnnoter(self, annoter):
        ''' Set annoter .'''
        self.annoter = annoter

    def Start(self):
        ''' Start gui running.'''
        # Check - of existence of images.
        if (self.annoter.GetImagesCount() == 0):
            logging.error('(Gui) No images to process!')
            return False

        # Create CV2 window
        cv2.namedWindow(self.winname)
        # Images slider
        cv2.createTrackbar('Images', self.winname,
                           0, self.annoter.GetImagesCount(), self._images_trackbar_cb)
        # Classes slider
        cv2.createTrackbar('Classes', self.winname, 0,
                           len(GetClasses())-1, self._classes_trackbar_cb)
        # Mouse handling
        cv2.setMouseCallback(self.winname, self._mouse_cb)

        # GUI keyboard loop
        key = 0
        self.annoter.Process()
        while (key != 27):
            key = 0

            # Resize image
            self.image = ResizeToHeight(self.annoter.GetImage(), 900)
            self.height, self.width = self.image.shape[0:2]

            # Update window
            self._update()

            # Loop waiting for specific keys
            keyAction = False
            while (keyAction == False):
                key = cv2.waitKeyEx()
                keyAction = self._keyboard_cb(key)

        # Destroy gui
        cv2.destroyWindow(self.winname)
        return True

    def _classes_trackbar_cb(self, arg):
        ''' Callback from trackbar.'''
        self._update()

    def _images_trackbar_cb(self, arg):
        ''' Callback from trackbar.'''
        imageNumber = cv2.getTrackbarPos('Images', self.winname)
        self.annoter.SetImageNumber(imageNumber)
        self.image = ResizeToHeight(self.annoter.GetImage(), 900)
        self._update()

    def _keyboard_cb(self, key):
        '''
            Keyboard callback.
            - Return True - exit from keyboard loop.
        '''
        # Application exit
        if (key == 27):
            return True
        elif (key == ord('s')):
            self.annoter.Save()
            return True
        elif (key == ord('d')):
            self.annoter.Process(forceDetector=True)
            self._update()
        elif (key == ord('r')):
            annote = self.__getHoveredAnnotation(
                boxes.PointToRelative(self.lastPos, self.width, self.height))
            if (annote is not None):
                self.annoter.RemoveAnnotation(annote)
            return True
        elif (key >= ord('0')) and (key <= ord('9')):
            classNumber = key - ord('0')
            cv2.setTrackbarPos('Classes', self.winname, classNumber)
        # Clear all annotations
        elif (key == ord('c')):
            self.annoter.ClearAnnotations()
            self._update()
        # Flip whole image
        elif (key == ord('f')):
            self.annoter.TransformShape('Flip')
            return True
        # x - delete image
        elif (key == ord('x')):
            if (self.__popupYesNo(self.image, text='Delete?') == True):
                self.annoter.Delete()
                self.annoter.Process()
                return True
        # Next image
        elif (key == 65363) or (key == ord('.')):
            self.annoter.ProcessNext()
            return True
        # Prev image
        elif (key == 65361) or (key == ord(',')):
            self.annoter.ProcessPrev()
            return True

        return False

    def _mouse_cb(self, event, x, y, flags, parameters):
        self.lastPos = [x, y]

        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.coords[:] = [(x, y)]
            self.dragging = True

        elif event == 0 and self.dragging:
            self.coords[1:] = [(x, y)]

        # Record ending (x,y) coordintes on left mouse bottom release
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.coords[1:] = [(x, y)]
            xs, ys = list(zip(*self.coords))
            self.coords = (min(xs), min(ys),
                           max(xs), max(ys))
            height, width = self.image.shape[0:2]
            box = boxes.ToRelative(self.coords, width, height)
            classNumber = cv2.getTrackbarPos('Classes', self.winname)
            self.annoter.AddAnnotation(box, classNumber)
            self.coords = []

        # Clear drawing boxes on right mouse button click
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.coords = []
            self.dragging = False

        self._update()

    def __getHoveredAnnotation(self, point):
        ''' Finds currently hovered annotation.'''
        founded = None
        if (len(self.lastPos) == 2):
            annotations = self.annoter.GetAnnotations()
            for element in annotations:
                if (element.IsInside(point) == True):
                    if (founded is not None):
                        area1 = boxes.GetArea(element.GetBox())
                        area2 = boxes.GetArea(founded.GetBox())
                        if (area1 < area2):
                            founded = element
                    else:
                        founded = element

        return founded

    def _update(self):
        ''' Update image view.'''
        im = self.image.copy()
        h, w = im.shape[0:2]

        # Draw currrent rectangle
        if len(self.coords) == 2:
            cv2.rectangle(im, self.coords[0], self.coords[1], (0, 255, 0), 1)

        # Draw all annotations
        annotations = self.annoter.GetAnnotations()
        if (annotations is not None) and (len(annotations)):
            for annotate in annotations:
                annotate.Draw(im)

        # Get hovered annotation
        if (len(self.lastPos) == 2):
            annote = self.__getHoveredAnnotation(
                boxes.PointToRelative(self.lastPos, w, h))
            if (annote is not None):
                annote.Draw(im, highlight=True)

        # Draw status bar
        im = self.__drawStatusBar(im)

        # Draw crosshair
        im = self.__drawCrosshair(im)

        cv2.setTrackbarPos('Images', self.winname,
                           self.annoter.GetImageNumber())
        cv2.imshow(self.winname, im)

    def __popupYesNo(self, im, text=''):
        ''' Draw popup and wait in loop yes/no.'''
        # Show message
        h, w = im.shape[0:2]
        x, y = (int(w/2), int(h/2))
        DrawText(self.image, '%s Yes(y) / No(n)?' %
                 text, (x, y), bgColor=(0, 0, 0))
        # Wait in loop
        key = 0
        while ((key != ord('y')) and (key != ord('n'))):
            key = cv2.waitKeyEx()

        return (key == ord('y'))

    def __drawCrosshair(self, im):
        ''' Draws mouse cursor crosshair.'''
        if (len(self.lastPos) == 2):
            x, y = self.lastPos
            h, w = im.shape[0:2]
            color = (255, 255, 255)
            im = cv2.line(im, (x, 0), (x, h), color, 1)
            im = cv2.line(im, (0, y), (w, y), color, 1)

        return im

    def __drawStatusBar(self, im):
        ''' Draws status bar.'''
        h, w = im.shape[0:2]
        barHeight = min(40, int(h*0.10))
        bar = np.full([barHeight, w, 3], 255, dtype=np.uint8)

        # Text values
        textXmargin = 5
        textYmargin = int(barHeight/2)
        textThickness = 2

        # Write current class name
        label = '[Class : %s.] ' % GetClassName(
            cv2.getTrackbarPos('Classes', self.winname))
        cv2.putText(im, label,
                    (textXmargin-1, textYmargin-1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), textThickness)
        cv2.putText(im, label,
                    (textXmargin, textYmargin), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), textThickness)
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, textThickness)
        textXmargin += label_width

        # Write status of image
        if (self.annoter.IsSynchronized() == True):
            label = '[Synchronized.] '
            cv2.putText(im, label,
                        (textXmargin-1, textYmargin -
                         1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), textThickness)
            cv2.putText(im, label,
                        (textXmargin, textYmargin), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), textThickness)
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, textThickness)
        else:
            label = '[De-synchronized!] '
            cv2.putText(im, label,
                        (textXmargin-1, textYmargin -
                         1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), textThickness)
            cv2.putText(im, label,
                        (textXmargin, textYmargin), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), textThickness)
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, textThickness)
        textXmargin += label_width

        # Write errors status
        errors = self.annoter.GetErrors()
        if (len(errors) != 0):
            for error in errors:
                label = '[%s] ' % error
                cv2.putText(im, label,
                            (textXmargin-1, textYmargin -
                             1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), textThickness)
                cv2.putText(im, label,
                            (textXmargin, textYmargin), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), textThickness)
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, textThickness)
                textXmargin += label_width

        im[0:barHeight, 0:w] = cv2.addWeighted(
            im[0:barHeight, 0:w], 0.5, bar, 0.5, 0)
        return im
