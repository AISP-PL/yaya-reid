#!/usr/bin/python3
import cv2
import numpy as np
import logging
import argparse
import sys
import engine.annote as annote
from engine.annoter import *
from helpers.files import *
from helpers.textAnnotations import *
from ObjectDetectors import IsCuda, IsDarknet, CreateDetector, GetDetectorLabels
from MainWindow import MainWindowGui


def main():
    ''' Main code method.'''
    # Arguments and config
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        required=True, help='Input path')
    parser.add_argument('-nd', '--noDetector', action='store_true',
                        required=False, help='Disable detector pre processing of files.')
    parser.add_argument('-f', '--forceDetector', action='store_true',
                        required=False, help='Force detector for every file.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        required=False, help='Show verbose finded and processed data')
    args = parser.parse_args()

    # Check - input argument
    if (args.input is None):
        print('Error! No arguments!')
        sys.exit(-1)


    # Enabled logging
    if (__debug__ is True):
        logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    else:
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    logging.debug('Logging enabled!')

    # Create detector

    scriptPath = os.path.dirname(os.path.realpath(__file__))
    detector = None
    if (IsDarknet() and (noDetector is False)):
        detector = CreateDetector(args.detector, path=scriptPath)
        if (detector is None):
            logging.error('Wrong detector!')
            return

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
                      detectorConfidence=args.detectorConfidence,
                      detectorNms=args.detectorNms,
                      )

    # Start QtGui
    gui = MainWindowGui(args, detector, annoter)
    sys.exit(gui.Run())


if __name__ == '__main__':
    main()
