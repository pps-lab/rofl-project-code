import unittest
import numpy as np
import time

def clip(updates, num_bits, num_frac, random=True):
    """Clips an update as if we had fixed precision `num_bits` with `num_frac` fractional bits"""
    assert num_bits >= num_frac
    # shifted = np.array(update, dtype=np.float) * pow(2, num_frac)
    # print(shifted)
    # for s in shifted:
    #     min = np.binary_repr(int(s), width=num_bits)
    #     max = np.binary_repr(int(round(s)), width=num_bits)
    #     print(min, max)
    # start = time.time()
    num_base = num_bits - num_frac
    # for b in range(int(pow(2, num_base - 1)), 0, -1):
    #     fra = 1.0 / pow(2, num_frac)
    #     for f in np.arange(1, 0, -fra):
    #         val = -b - f
    #         bins.append(val)
    # for b in range(0, int(pow(2, num_base - 1))):
    #     fra = 1.0 / pow(2, num_frac)
    #     for f in np.arange(0, 1, fra):
    #         val = b + f
    #         bins.append(val)
    fra = 1.0 / pow(2, num_frac)
    lim = int(pow(2, num_base - 1)) - fra

    bins = np.arange(-lim, lim, fra)

    clipped_all = []
    for update in updates:
        update = np.clip(update, bins[0], bins[-1] - 0.000001) # Clip to ensure value is within range
        shape = None
        if len(update.shape) > 1:
            shape = update.shape
            update = update.flatten()
        digits = np.digitize(update, bins)
        # clipped = [clip_binary(x, bins[maxIndice - 1], bins[maxIndice], random) for maxIndice, x in zip(digits, update)]
        r = np.random.rand(len(digits)) if random else 0.5
        min, max = bins[digits - 1], bins[digits]
        assert np.all(min <= update) and np.all(update <= max)
        translated = (update - min) / (max - min)
        clipped = np.where(translated < r, min, max)

        if shape is not None:
            clipped = np.array(clipped, dtype=np.float32).reshape(shape)
        clipped_all.append(clipped)
    # end = time.time()
    # print(f"Done in {end-start} seconds!")
    return clipped_all



def clip_binary(x, min, max, random):
    assert min <= x <= max, f"{min} <= {x} <= {max}"
    if x == min:
        return min
    elif x == max:
        return max

    norm = max - min
    val = (x - min) / norm
    if random:
        r = np.random.rand(1)
        if val < r:
            return min
        else:
            return max
    else:
        if val >= 0.5:
            return max
        else:
            return min

class TestClip(unittest.TestCase):

    def test_it_clips(self):
        update = [0.5, 0.001223, 0.12, 8.4]
        num_bits = 8
        num_frac = 4

        clip(update, num_bits, num_frac)

    def test_it_clips_large(self):
        D = 32000
        update = np.random.random(D) * 15
        num_bits = 8
        num_frac = 4

        clip(update, num_bits, num_frac)

    def test_it_clips_large_average(self):
        D = 32000
        update = np.random.random(D) * 15
        num_bits = 8
        num_frac = 4

        clipped = clip(update, num_bits, num_frac)
        avg_c, avg_o = np.average(clipped), np.average(update)
        print(avg_c, avg_o)