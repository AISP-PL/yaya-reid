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
    parser.add_argument('-nd', '--noDetector', action='store_true',
                        required=False, help='Disable detector pre processing of files.')
    parser.add_argument('-f', '--forceDetector', action='store_true',
                        required=False, help='Force detector for every file.')
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
    ModelCreate(models, 0)

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
