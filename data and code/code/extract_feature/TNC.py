#!/usr/bin/env python
#_*_coding:utf-8_*_

import re
import sys, os
import numpy as np
import pandas as pd
import itertools
from collections import Counter
import numpy as np
def TNC(fastas, **kw):

    AA1 = 'ACGT'
    AA2 = 'ACGU'
    encodings = []

    AADict1 = {}
    for i in range(len(AA1)):
        AADict1[AA1[i]] = i
    AADict2 = {}
    for i in range(len(AA2)):
        AADict2[AA2[i]] = i


    for i in fastas:
        sequence = i.strip()
        code = []
        tmpCode = [0] * 64
        for j in range(len(sequence) - 3 + 1):
            if 'U' not in sequence:
                tmpCode[AADict1[sequence[j]] * 16 + AADict1[sequence[j+1]]*4 + AADict1[sequence[j+2]]] = \
                    tmpCode[AADict1[sequence[j]] * 16 + AADict1[sequence[j+1]]*4 + AADict1[sequence[j+2]]] +1
            else:
                tmpCode[AADict2[sequence[j]] * 16 + AADict2[sequence[j + 1]] * 4 + AADict2[sequence[j + 2]]] = \
                    tmpCode[AADict2[sequence[j]] * 16 + AADict2[sequence[j + 1]] * 4 + AADict2[sequence[j + 2]]] + 1

        if sum(tmpCode) != 0:
            tmpCode = [i/sum(tmpCode) for i in tmpCode]
        code = code + tmpCode
        encodings.append(code)
    return np.array(encodings)
