import pandas as pd
import numpy as np


class Strategy:
    """
    This is the parent class of any strategy class. Each of the subclasses
    will have to implement the method generate_signals.
    """
    def generate_signals(self, data):
        raise NotImplementedError("Subclasses must implement generate_signals method")

    def generate_random_signals(self, data):
        return pd.Series(np.random.choice([-1, 0, 1], size=len(data)), index=data.index)

    def plot_signals(self, prices):
        raise NotImplementedError("Subclasses must implement plot_signals method")

