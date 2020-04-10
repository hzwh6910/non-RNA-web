import pandas as pd
import numpy as np
from kmer import Kmer


def Kmer1234(fasta):
    t4 = Kmer(fasta, 4)
    df4 = pd.DataFrame(t4)
    t1 = Kmer(fasta, 1)
    df1 = pd.DataFrame(t1)
    t2 = Kmer(fasta, 2)
    df2 = pd.DataFrame(t2)
    t3 = Kmer(fasta, 3)
    df3 = pd.DataFrame(t3)
    res_heng = pd.concat([df1, df2, df3, df4], axis=1).values
    return res_heng