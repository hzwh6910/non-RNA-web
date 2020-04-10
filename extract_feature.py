#!/usr/bin/env python
# _*_coding:utf-8_*_
import sys



sys.path.extend(["../../", "../", "./"])
import sys, os
import pandas as pd
import numpy as np
from CKSNAP import CKSNAP
from DNC import DNC
from NAC import NAC
from TNC import TNC
from RCKmer import RCKmer
from kmer import Kmer
from Kmer1234 import Kmer1234


def read_fasta(file):
    f = open(file)
    docs = f.readlines()
    fasta = []
    for seq in docs:
        if seq.startswith(">"):
            continue
        else:
            fasta.append(seq)

    return np.array(fasta)


def extract_feature(fasta):

    t1 = CKSNAP(fasta)
    df1 = pd.DataFrame(t1)


    t2 = Kmer(fasta, 4)
    df2 = pd.DataFrame(t2)


    t3 = Kmer1234(fasta)
    df3 = pd.DataFrame(t3)



    t4 = NAC(fasta)
    df4 = pd.DataFrame(t4)


    t5 = RCKmer(fasta)
    df5 = pd.DataFrame(t5)


    t6 = DNC(fasta)
    df6 = pd.DataFrame(t6)


    t7 = TNC(fasta)
    df7 = pd.DataFrame(t7)
    return t1,t2,t3,t4,t5,t6,t7




