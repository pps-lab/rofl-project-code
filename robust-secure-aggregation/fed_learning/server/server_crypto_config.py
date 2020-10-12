

class ServerCryptoConfig(object):
    def __init__(self, 
                 fp_bits,
                 fp_frac,
                 value_range,
                 n_partition,
                 l2_value_range,
                 optimistic_starting):
        self.fp_bits = fp_bits
        self.fp_frac = fp_frac
        self.value_range = value_range
        self.n_partition = n_partition
        self.l2_value_range = l2_value_range
        self.optimistic_starting = optimistic_starting
