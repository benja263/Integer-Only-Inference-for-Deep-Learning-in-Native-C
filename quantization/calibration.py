from collections import Counter

import numpy as np
from scipy.stats import entropy


class Histogram:
    def __init__(self, hist=None, skip_zeros=True, num_bins=2048, num_bits=8):

        self.hist = hist
        self.bin_edges = None
        self.skip_zeros = skip_zeros
        self.num_bins = num_bins
        self.num_bits = num_bits

    def fill_hist(self, data: np.array):
        x = np.abs(data.copy())
        if self.skip_zeros:
            x = x[np.where(x != 0)]

        if self.bin_edges is None and self.hist is None:
            # first time it uses num_bins to compute histogram.
            self.hist, self.bin_edges = np.histogram(x, bins=self.num_bins)
        else:
            temp_amax = np.max(x)
            if temp_amax > self.bin_edges[-1]:
                # increase the number of bins
                width = self.bin_edges[1] - self.bin_edges[0]
                new_bin_edges = np.arange(self.bin_edges[-1] + width, temp_amax + width, width)
                self.bin_edges = np.hstack((self.bin_edges, new_bin_edges))
            hist, self.bin_edges = np.histogram(x, bins=self.bin_edges)
            # update histogram
            hist[:len(self.hist)] += self.hist
            self.hist = hist

    def compute_amax(self):
        hist = self.hist
        bin_edges = self.bin_edges
        amax = _compute_amax_entropy(hist, bin_edges, self.num_bits)
        return amax
        

def compute_amax_entropy(hist, bin_edges, num_bits, stride=1):
    """Returns amax that minimizes KL-Divergence of the collected histogram"""

    # If calibrator hasn't collected any data, return none
    if bin_edges is None and hist is None:
        return None

    bins = hist[:]
    bins[0] = bins[1]

    total_data = np.sum(bins)

    divergences = []
    arguments = []

    # only take signed case
    nbins = 1 << (num_bits - 1)

    starting = 128
    stop = len(bins)

    new_density_counts = np.zeros(nbins, dtype=np.float64)

    for i in range(starting, stop + 1, stride):
        new_density_counts.fill(0)
        space = np.linspace(0, i, num=nbins + 1)
        digitized_space = np.digitize(range(i), space) - 1

        digitized_space[bins[:i] == 0] = -1

        for idx, digitized in enumerate(digitized_space):
            if digitized != -1:
                new_density_counts[digitized] += bins[idx]

        counter = Counter(digitized_space)
        for key, val in counter.items():
            if key != -1:
                new_density_counts[key] = new_density_counts[key] / val

        new_density = np.zeros(i, dtype=np.float64)
        for idx, digitized in enumerate(digitized_space):
            if digitized != -1:
                new_density[idx] = new_density_counts[digitized]

        total_counts_new = np.sum(new_density) + np.sum(bins[i:])
        # normalize
        if (new_density != 0).any():
            new_density = new_density / np.sum(new_density)

        reference_density = np.array(bins[:len(digitized_space)])
        reference_density[-1] += np.sum(bins[i:])

        total_counts_old = np.sum(reference_density)
        if round(total_counts_new) != total_data or round(total_counts_old) != total_data:
            raise RuntimeError("Count mismatch! total_counts_new={}, total_counts_old={}, total_data={}".format(
                total_counts_new, total_counts_old, total_data))

        if (reference_density != 0).any():
            reference_density = reference_density / np.sum(reference_density)

        ent = entropy(reference_density, new_density)
        divergences.append(ent)
        arguments.append(i)

    divergences = np.array(divergences)
    last_argmin = len(divergences) - 1 - np.argmin(divergences[::-1])
    calib_amax = bin_edges[last_argmin * stride + starting]
    return calib_amax




