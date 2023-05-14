import math

KAPPA = 2.41798793102
PI = 3.14159265358
CONSTANT_P = (4 / 3) * PI * (KAPPA / PI) ** 1.5
EXP = math.exp(1)

FEATURES = {
    "Donor": [1.0, True],
    "Acceptor": [1.0, True],
    "PosIonizable": [1.0, False],
    "NegIonizable": [1.0, False],
    "Aromatic": [0.7, True],
    "Hydrophobe": [1.0, False],
}
