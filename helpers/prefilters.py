'''
Created on 18 lis 2020

@author: spasz
'''
import numpy as np
import logging
from . import boxes
from . import metrics


def FilterIOUbyConfidence(annotations1,
                          annotations2,
                          maxIOU=0.70):
    '''
        Filter annotation1 with annotations2
        if has bigger IOU > maxIOU.
        Annotation with bigger confidence stays!
    '''
    # Create IOU results matrix
    results = np.zeros([len(annotations1), len(annotations2)],
                       dtype=np.float32)
    for i, annote1 in enumerate(annotations1):
        for j, annote2 in enumerate(annotations2):
            if(annote1 != annote2):
                results[i, j] = metrics.MetricIOU(
                    annote1.GetBox(), annote2.GetBox())

    # Parse each row of IOU matrix
    passed = []
    for i, annote in enumerate(annotations1):
        isFiltered = False
        for j in range(len(annotations2)):
            # If IOU >= maxIOU
            if (results[i, j] >= maxIOU):
                # If confidence is smaller
                if (annote.GetConfidence() <= annotations2[j].GetConfidence()):
                    isFiltered = True
                    break

        if (isFiltered == True):
            logging.info('Filtered out (%s, %2.2f).',
                         annote.GetClassName(), annote.GetConfidence())
        else:
            passed.append(annote)

    return passed
