from typing import List

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# String stuff

def startswith_any(filename: str, prefs: List[str]):
    for pref in prefs:
        if filename.startswith(pref):
            return True
    return False

# Implementing utility functions

