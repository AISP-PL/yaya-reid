#!/usr/bin/python3
import cv2
import numpy as np
import logging
import argparse
import sys
import engine.annote as annote
from engine.gui import *
from engine.annoter import *
from helpers.files import *
from helpers.textAnnotations import *
from ObjectDetectors import IsCuda, IsDarknet, CreateDetector, GetDetectorLabels
from MainWindow import MainWindowGui

# Arguments and config
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str,
                    required=True, help='Input path')
parser.add_argument('-c', '--config', type=str,
                    required=False, help='Config path')
parser.add_argument('-dc', '--drawConfidence', type=int, nargs='?', const=1, default=1,
                    required=False, help='Draw annotations confidence (=1). No drawing (=0).')
parser.add_argument('-sb', '--sortBy', type=int, nargs='?', const=2, default=2,
                    required=False, help='Sort by method number (0 None, 1 Datetime, 2 Inv. Datetime, 3 Alphabet)')
parser.add_argument('-oc', '--onlyClass', type=int,
                    required=False, help='Only specific class number')
parser.add_argument('-odc', '--onlyDetectedClass', type=int,
                    required=False, help='Only specific detected class number(Procces all detections)')
parser.add_argument('-on', '--onlyNewFiles', action='store_true',
                    required=False, help='Process only files without detections file.')
parser.add_argument('-oo', '--onlyOldFiles', action='store_true',
                    required=False, help='Process only files with detections file.')
parser.add_argument('-nd', '--noDetector', action='store_true',
                    required=False, help='Disable detector pre processing of files.')
parser.add_argument('-oe', '--onlyFilesWithErrors', action='store_true',
                    required=False, help='Process only files with errors.')
parser.add_argument('-d', '--detector', type=int, nargs='?', const=0, default=0,
                    required=False, help='Detector type - default 0')
parser.add_argument('-v', '--verbose', action='store_true',
                    required=False, help='Show verbose finded and processed data')
args = parser.parse_args()

# Check - input argument
if (args.input is None):
    print('Error! No arguments!')
    sys.exit(-1)

# Check - files filter
isOnlyNewFiles = False
if (args.onlyNewFiles):
    isOnlyNewFiles = True

# Check - files filter
isOnlyOldFiles = False
if (args.onlyOldFiles):
    isOnlyOldFiles = True

# Check - files filter
isOnlyErrorFiles = False
if (args.onlyFilesWithErrors):
    isOnlyErrorFiles = True

# Check - files filter
isOnlySpecificClass = None
if (args.onlyClass is not None):
    isOnlySpecificClass = args.onlyClass

# Check - files filter
isOnlyDetectedClass = None
if (args.onlyDetectedClass is not None):
    isOnlyDetectedClass = args.onlyDetectedClass

# Check - detector
noDetector = False
if (args.noDetector is not None):
    noDetector = args.noDetector

# Enabled logging
if (__debug__ is True):
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
else:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logging.debug('Logging enabled!')

# Create detector
detector = None
if (IsDarknet() and (noDetector is False)):
    detector = CreateDetector(args.detector)
    detector.Init()
    annote.Init(detector.GetClassNames())
# CUDA not installed
else:
    noDetector = True
    annote.Init(GetDetectorLabels(args.detector))

# Create annoter
annoter = Annoter(FixPath(GetFileLocation(args.input)),
                  detector,
                  noDetector,
                  args.sortBy,
                  isOnlyNewFiles,
                  isOnlyOldFiles,
                  isOnlyErrorFiles,
                  isOnlyDetectedClass,
                  isOnlySpecificClass)

# Start QtGui
gui = MainWindowGui(args, detector, annoter)
sys.exit(gui.Run())


# Old OpenCV GUI
# # Start Gui
# gui = Gui('YoloAnnotate', args)
# gui.SetAnnoter(annoter)
# gui.Start()
# # Clean after all
# gui.Close()
