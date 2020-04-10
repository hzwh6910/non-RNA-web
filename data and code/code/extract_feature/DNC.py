import re
import numpy as np
def DNC(fastas, **kw):

    base1 = 'ACGT'
    base2 = 'ACGU'
    encodings = []

    AADict1 = {}
    for i in range(len(base1)):
        AADict1[base1[i]] = i
    AADict2 = {}
    for i in range(len(base2)):
        AADict2[base2[i]] = i

    for i in fastas:
        sequence= i.strip()
        code = []
        tmpCode = [0] * 16
        for j in range(len(sequence) - 2 + 1):
            if 'U' in sequence:
                tmpCode[AADict2[sequence[j]] * 4 + AADict2[sequence[j+1]]] = tmpCode[AADict2[sequence[j]] * 4 + AADict2[sequence[j+1]]] +1
            else:
                tmpCode[AADict1[sequence[j]] * 4 + AADict1[sequence[j + 1]]] = tmpCode[
                                                                                   AADict1[sequence[j]] * 4 + AADict1[
                                                                                       sequence[j + 1]]] + 1
        if sum(tmpCode) != 0:
            tmpCode = [i/sum(tmpCode) for i in tmpCode]
        code = code + tmpCode
        encodings.append(code)
    return np.array(encodings)