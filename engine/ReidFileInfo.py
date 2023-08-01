'''
Created on 17 lis 2020

@author: spasz
'''
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import re


class ReidDataset(str, Enum):
    ''' Enum with reid datasets.'''
    AispReid = 'aispreid'
    Market1501 = 'market1501'


@dataclass
class ReidFileInfo:
    ''' Informations stored in name of reid image file.'''
    # Identity number
    identity: int = field(init=True, default=None)
    # Camera number
    camera: int = field(init=True, default=None)
    # Frame number
    frame: int = field(init=True, default=None)
    # Reid dataset type
    dataset: ReidDataset = field(init=True, default=ReidDataset.AispReid)

    @staticmethod
    def toPath(identity_number: int,
               camera_number: int,
               frame_number: int,
               dataset: ReidDataset) -> str:
        ''' According to dataset type return path to image.'''
        # AISP Reid
        if dataset == ReidDataset.AispReid:
            return f'ID{identity_number}_CAM{camera_number}_FRAME{frame_number}.jpeg'

        # Market1501
        if dataset == ReidDataset.Market1501:
            return None

        return None

    @staticmethod
    def PatternAispReid(text: str) -> ReidFileInfo:
        ''' Parse AISP reid filename.'''
        # Filename pattern
        pattern = re.compile(r'ID([-\d]+)_CAM(\d)_FRAME(\d)')
        # Regular expression : Get results
        regexResults = pattern.search(text)
        if regexResults is None:
            return None

        # Get pid, camid, frame
        pid, camid, frame = map(int, regexResults.groups())

        return ReidFileInfo(pid, camid, frame)

    @staticmethod
    def FromFilename(filename: str) -> ReidFileInfo:
        ''' Create fileinfo from filename.'''
        # Patter : AISP
        result = ReidFileInfo.PatternAispReid(filename)
        if result is not None:
            return result

        # Pattern : Market1501
        # @TODO

        return None
