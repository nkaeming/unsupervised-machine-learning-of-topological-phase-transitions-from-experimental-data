"""
This file contains all the handling of the theory data.
"""
import h5py
import numpy as np

class TheoryPhaseDiagram:

    def __init__(self, theory_file='data/phase_diagram_theory.h5'):
        """
        Initialize the class. Here we load the data from the calculatet theory file.
        """
        with h5py.File(theory_file, 'r') as h5f:
            self.freq = h5f['freq'][:].astype(np.float32)
            self.phase = h5f['phase'][:].astype(np.float32)
            self.chern_number = h5f['chern_number'][:].astype(np.float32)

    def get_theory_predictions(self, freqs, phases):
        """
        Function to get theory predictions for a list of freqs and phases.
        Params:
            freqs: A list of freqs with a length of n
            phases: A list of phases with length n
        """
        number_of_samples = len(freqs)

        chern_numbers = np.zeros(number_of_samples)
        for idx in range(number_of_samples):
            freq_idx = np.where(freqs[idx] == self.freq)[0]
            phase_idx = np.where(phases[idx] == self.phase)[0]
            chern_numbers[idx] = self.chern_number[phase_idx, freq_idx]
        return chern_numbers

    def get_theory_transitions(self, phase):
        """
        Function to get the two phase transitions for a given shaking phase.
        """
        number_of_samples = len(self.freq)
        phases = np.ones(number_of_samples) * phase
        chern_numbers = np.round(self.get_theory_predictions(self.freq, phases))
        non_trivial = np.where(chern_numbers != 0)[0]
        lower_freq = self.freq[non_trivial[0]]
        upper_freq = self.freq[non_trivial[-1]]
        return lower_freq, upper_freq



