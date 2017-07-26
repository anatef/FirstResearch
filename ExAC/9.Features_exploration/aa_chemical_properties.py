from enum import Enum

class aa_functional_group(Enum):
    ALIPHATIC = 0
    AROMATIC = 1
    NEGATIVE = 2
    POSITIVE = 3
    POLAR = 4
    STOP = 5

class aa_charge(Enum):
    NEUTRAL = 0
    POSITIVE = 1
    NEGATIVE = -1

class aa_propensity(Enum):
    ALPHA_HELIX = 0
    BETA_SHEET = 1
    TURN = 2

aa_functional_group_dict = {'A': aa_functional_group.ALIPHATIC,
                       'C': aa_functional_group.POLAR,
                       'D': aa_functional_group.NEGATIVE,
                       'E': aa_functional_group.NEGATIVE,
                       'F': aa_functional_group.AROMATIC,
                       'G': aa_functional_group.ALIPHATIC,
                       'H': aa_functional_group.POSITIVE,
                       'I': aa_functional_group.ALIPHATIC,
                       'K': aa_functional_group.POSITIVE,
                       'L': aa_functional_group.ALIPHATIC,
                       'M': aa_functional_group.ALIPHATIC,
                       'N': aa_functional_group.POLAR,
                       'P': aa_functional_group.POLAR,
                       'Q': aa_functional_group.POLAR,
                       'R': aa_functional_group.POSITIVE,
                       'S': aa_functional_group.POLAR,
                       'T': aa_functional_group.POLAR,
                       'V': aa_functional_group.ALIPHATIC,
                       'W': aa_functional_group.AROMATIC,
                       'Y': aa_functional_group.AROMATIC,
                       '*': aa_functional_group.STOP}

aa_charge_dict = {'A': aa_charge.NEUTRAL,
             'C': aa_charge.NEUTRAL,
             'D': aa_charge.NEGATIVE,
             'E': aa_charge.NEGATIVE,
             'F': aa_charge.NEUTRAL,
             'G': aa_charge.NEUTRAL,
             'H': aa_charge.NEUTRAL,
             'I': aa_charge.NEUTRAL,
             'K': aa_charge.POSITIVE,
             'L': aa_charge.NEUTRAL,
             'M': aa_charge.NEUTRAL,
             'N': aa_charge.NEUTRAL,
             'P': aa_charge.NEUTRAL,
             'Q': aa_charge.NEUTRAL,
             'R': aa_charge.POSITIVE,
             'S': aa_charge.NEUTRAL,
             'T': aa_charge.NEUTRAL,
             'V': aa_charge.NEUTRAL,
             'W': aa_charge.NEUTRAL,
             'Y': aa_charge.NEUTRAL,
             '*': aa_charge.NEUTRAL}

#Based on "A simple method for displaying the hydropatic character of a protein, Kyte & Doolittle, 1982
hindex_Kyte_Doolitle = {'A': 1.8,
                        'C': 2.5,
                        'D': -3.5,
                        'E': -3.5,
                        'F': 2.8,
                        'G': -0.4,
                        'H': -3.2,
                        'I': 4.5,
                        'K': -3.9,
                        'L': 3.8,
                        'M': 1.9,
                        'N': -3.5,
                        'P': -1.6,
                        'Q': -3.5,
                        'R': -4.5,
                        'S': -0.8,
                        'T': -0.7,
                        'V': 4.2,
                        'W': -0.9,
                        'Y': -1.3,
                        '*': 0}

#Based on: "EMPIRICAL PREDICTIONS OF PROTEIN CONFORMATION", Chou & Fasman, 1978
propensity_chou_fasman = {'A': [1.42, 0.83, 0.66],
                          'C': [0.70, 1.19, 1.19],
                          'D': [1.01, 0.54, 1.46],
                          'E': [1.51, 0.37, 0.74],
                          'F': [1.13, 1.38, 0.60],
                          'G': [0.57, 0.75, 1.56],
                          'H': [1.00, 0.87, 0.95],
                          'I': [1.08, 1.60, 0.47],
                          'K': [1.14, 0.74, 1.01],
                          'L': [1.21, 1.30, 0.59],
                          'M': [1.45, 1.05, 0.60],
                          'N': [0.67, 0.89, 1.56],
                          'P': [0.57, 0.55, 1.52],
                          'Q': [1.11, 1.11, 0.98],
                          'R': [0.98, 0.93, 0.95],
                          'S': [0.77, 0.75, 1.43],
                          'T': [0.83, 1.19, 0.96],
                          'V': [1.06, 1.70, 0.50],
                          'W': [0.83, 1.19, 0.96],
                          'Y': [0.69, 1.47, 1.14],
                          '*': [0, 0, 0]}

#Amino acid volume (A^3) from "Protein volume in solution", Zamyatnin, 1972
volume = {'A': 88.6,
          'C': 108.5,
          'D': 111.1,
          'E': 138.4,
          'F': 189.9,
          'G': 60.1,
          'H': 153.2,
          'I': 166.7,
          'K': 168.6,
          'L': 166.7,
          'M': 162.9,
          'N': 114.1,
          'P': 112.7,
          'Q': 143.8,
          'R': 173.4,
          'S': 89.0,
          'T': 116.1,
          'V': 140.0,
          'W': 227.8,
          'Y': 193.6,
          '*': 0.0

}
