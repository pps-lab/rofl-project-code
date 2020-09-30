import unittest
import numpy as np

# def clip(update, num_bits, num_frac, random=True):
#     """Clips an update as if we had fixed precision `num_bits` with `num_frac` fractional bits"""
#     assert num_bits > num_frac
#     num_base = num_bits - num_frac
#     fra = 1.0 / pow(2, num_frac)
#     lim = int(pow(2, num_base - 1))
#     bins = np.arange(-lim, lim, fra)
#
#     shape = None
#     if len(update.shape) > 1:
#         shape = update.shape
#         update = update.flatten()
#     digits = np.digitize(update, bins)
#
#     r = np.random.rand(len(digits)) if random else 0.5
#     min, max = bins[digits - 1], bins[digits]
#     assert np.all(min <= update) and np.all(update <= max)
#     translated = (update - min) / (max - min)
#     clipped = np.where(translated < r, min, max)
#
#     if shape is not None:
#         clipped = np.array(clipped, dtype=np.float32).reshape(shape)
#
#     return clipped
def clip(updates, num_bits, num_frac, random=True):
    """Clips an update as if we had fixed precision `num_bits` with `num_frac` fractional bits"""
    assert num_bits >= num_frac
    num_base = num_bits - num_frac
    fra = 1.0 / pow(2, num_frac)
    lim = int(pow(2, num_base - 1))
    bins = np.arange(-lim, lim, fra)

    clipped_all = []
    for update in updates:
        update = np.clip(update, bins[0], bins[-1] - 0.000001) # Clip to ensure value is within range
        shape = None
        if len(update.shape) > 1:
            shape = update.shape
            update = update.flatten()
        digits = np.digitize(update, bins)
        # print(digits, update, bins)
        # clipped = [clip_binary(x, bins[maxIndice - 1], bins[maxIndice], random) for maxIndice, x in zip(digits, update)]
        r = np.random.rand(len(digits)) if random else 0.5
        min, max = bins[digits - 1], bins[digits]
        assert np.all(min <= update) and np.all(update <= max)
        translated = (update - min) / (max - min)
        clipped = np.where(translated < r, min, max).astype(np.float32)

        if shape is not None:
            clipped = clipped.reshape(shape)
        clipped_all.append(clipped)
    # end = time.time()
    # print(f"Done in {end-start} seconds!")
    return clipped_all


class TestClip(unittest.TestCase):

    def test_it_clips(self):
        update = np.array([0.5, 0.001223, 0.12, 7.4])
        num_bits = 8
        num_frac = 4

        print(clip([update], num_bits, num_frac))

    def test_it_clips_complex_shape(self):
        update = np.array([[[1, 2, 3, 4], [4, 5, 6, 7], [1.5, 2.5, 3.5, 4.5], [5.5, 6.5, 7.5, 7.0625]], [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]], dtype=np.float32)
        num_bits = 8
        num_frac = 4

        res = clip([update], num_bits, num_frac)
        print([update])
        print(res)
        assert np.array_equal([update], res)

    def test_it_clips_large(self):
        D = 32000
        update = np.random.random(D) * 7
        num_bits = 8
        num_frac = 4

        clip([update], num_bits, num_frac)

    def test_it_clips_large_average(self):
        D = 32000
        update = np.random.random(D) * 7
        num_bits = 8
        num_frac = 4

        clipped = clip([update], num_bits, num_frac)
        avg_c, avg_o = np.average(clipped), np.average(update)
        print(avg_c, avg_o)