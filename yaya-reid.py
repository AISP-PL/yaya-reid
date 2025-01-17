#!/usr/bin/python3
import logging
import argparse
import sys
from ReID.ReidClassifier import GetReidClassifier, ModelCreate, ModelsList, ModelsPrint
from engine.AnnoterReid import AnnoterReid
from helpers.files import *
from MainWindow import MainWindowGui


def main():
    ''' Main code method.'''
    # Arguments and config
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        required=True, help='Input path')
    parser.add_argument('-d', '--detectorNumber', type=int, nargs='?', const=0, default=0,
                        required=False, help='''Detector number from list''')
    parser.add_argument('-nd', '--noDetector', action='store_true',
                        required=False, help='Disable detector pre processing of files.')
    parser.add_argument('-f', '--force', action='store_true',
                        required=False, help='Force visuals + reid classifier for every file.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        required=False, help='Show verbose finded and processed data')
    args = parser.parse_args()

    # Enabled logging
    if (__debug__ is True):
        logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    else:
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    # ReID Classifier : Create
    models = ModelsList()
    ModelsPrint(models)

    # ReID Classifier : Get number of classifier to create
    if (args.detectorNumber >= len(models)):
        logging.error('Invalid REID classifier number %d!',
                      args.detectorNumber)
        return

    ModelCreate(models, args.detectorNumber)

    # Create annoter
    annoter = AnnoterReid(dirpath=FixPath(GetFileLocation(args.input)),
                          args=args,
                          features_classifier=GetReidClassifier()
                          )

    # Start QtGui
    gui = MainWindowGui(args=args,
                        reid_classifier=GetReidClassifier(),
                        annoter=annoter)
    sys.exit(gui.Run())


if __name__ == '__main__':
    main()
