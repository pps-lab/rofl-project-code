from rig.type_casts import float_to_fp


def get_fp_max(n_bits, n_frac):
    n_int = n_bits - n_frac - 1
    return 2**n_int - 1/(2**n_frac)

def get_fp_min(n_bits, n_frac):
    return -get_fp_max(n_bits, n_frac)