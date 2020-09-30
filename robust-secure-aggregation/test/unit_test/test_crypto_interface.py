import os
import sys
from os.path import join, dirname, abspath

module_loc = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,os.path.dirname(module_loc))

import unittest
import numpy as np
import binascii
from rig.type_casts import validate_fp_params
import subprocess

from fed_learning.crypto.crypto_interface import CryptoInterface
from fed_learning.crypto.crypto_interface.exception import *
from test.config_loader import ConfigLoader
from test.fp_utils import get_fp_max, get_fp_min
import logging

CURRENT_FOLDER = dirname(__file__)
ROOT = dirname(dirname(CURRENT_FOLDER))
RUST_PROJECT_PATH= join(ROOT, 'fed_learning/crypto/crypto_interface/rust_crypto')

class TestCryptoInterface(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        self.ci = CryptoInterface()
        super(TestCryptoInterface, self).__init__(*args, **kwargs)
        self.config = ConfigLoader()
        _, fp_max64 = validate_fp_params(False, self.config.fp_bits, self.config.fp_frac)
        self.fp_max = np.float32(fp_max64)
        self.fp_min = np.float32(-fp_max64)
        logging.disable(logging.ERROR)
        #subprocess.call(['cd %s' % '/home'], shell=True)
        #subprocess.call(['cd %s ; cargo build --release --features "fp%s frac%s"' % (RUST_PROJECT_PATH, self.config.fp_bits, self.config.fp_frac)], shell=True)

    def test_commit_no_blinding_roundtrip_lossless(self):
        x_vec = np.array([0.5, -1.25], dtype='float32')
        x_vec_enc = self.ci.commit_no_blinding(x_vec)
        y_vec = self.ci.extract_values(x_vec_enc)
        for (x, y) in zip(x_vec, y_vec):
            self.assertEqual(x, y)

    def test_commit_no_blinding_roundtrip_lossy_rounded(self):
        x_vec = np.array([21-(1.0/3.0), 12+0.1], dtype='float32')
        x_vec_enc = self.ci.commit_no_blinding(x_vec)
        y_vec = self.ci.extract_values(x_vec_enc)
        self.assertEqual(len(x_vec), len(y_vec))
        for (x, y) in zip(x_vec, y_vec):
            diff = abs(x-y)
            self.assertLessEqual(diff, 2**(-(self.config.fp_frac+1.0)))

    def test_commit_no_blinding_roundtrip_lossy_saturated(self):
        x_vec = np.array([float(self.fp_max)+float(12.9), self.fp_min-5.0], dtype='float32')
        x_vec_enc = self.ci.commit_no_blinding(x_vec)
        y_vec = self.ci.extract_values(x_vec_enc)

        self.assertEqual(len(x_vec), len(y_vec))
        self.assertEqual(y_vec[1], self.fp_min)
        self.assertEqual(y_vec[0], self.fp_max)

    def test_add_commitments(self):
        x_vec = np.array([0.25, 1.25, -1.5], dtype='float32')
        y_vec = np.array([-0.75, 1.25, -2.0], dtype='float32')
        z_vec = np.array([0.5, 1.25, -3.0], dtype='float32')
        target_vec = np.array([0.0, 3.75, -6.5], dtype='float32')
        
        x_vec_enc = self.ci.commit_no_blinding(x_vec)
        y_vec_enc = self.ci.commit_no_blinding(y_vec)
        z_vec_enc = self.ci.commit_no_blinding(z_vec)
        sum_vec_enc = self.ci.add_commitments([x_vec_enc, y_vec_enc, z_vec_enc])
        sum_vec = self.ci.extract_values(sum_vec_enc)
        
        for t, s in zip(target_vec, sum_vec):
            self.assertEqual(t, s)

    def test_commit_with_cancelling_blindings_roundtrip(self):
        x_vec = np.array([0.25, 1.25, -1.5], dtype='float32')
        y_vec = np.array([-0.75, 1.25, -2.0], dtype='float32')
        z_vec = np.array([0.5, 1.25, -3.0], dtype='float32')
        target_vec = np.array([0.0, 3.75, -6.5], dtype='float32')

        blindings = self.ci.generate_cancelling_blindings(3, 3)
        commitments = [self.ci.commit(v, b) for v, b in zip([x_vec, y_vec, z_vec], blindings)]
        sum_vec_enc = self.ci.add_commitments(commitments)
        sum_vec = self.ci.extract_values(sum_vec_enc)

        for t, s in zip(target_vec, sum_vec):
            self.assertEqual(t, s)

    def test_commit_quantized(self):
        # x_vec = np.array([-0.140625,  -0.1015625, -0.0625, 0.0390625, -0.0546875, -0.0625], dtype='float32')
        x_vec = np.array([-0.140625,  -0.1015625, -0.0625], dtype='float32')

        x_vec_commit = self.ci.commit_no_blinding(x_vec)
        x_res = self.ci.extract_values(x_vec_commit)
        print(x_res)
        np.testing.assert_array_equal(x_vec, x_res)
        # self.ci.extract_values(sum_vec_enc)
        y_vec = np.array([-0.75, 1.25, -2.0], dtype='float32')
        z_vec = np.array([0.5, 1.25, -3.0], dtype='float32')
        target_vec = np.array([-0.390625, 2.3984375, -5.0625], dtype='float32')

        blindings = self.ci.generate_cancelling_blindings(3, 3)
        commitments = [self.ci.commit(v, b) for v, b in zip([x_vec, y_vec, z_vec], blindings)]
        sum_vec_enc = self.ci.add_commitments(commitments)
        sum_vec = self.ci.extract_values(sum_vec_enc)

        for t, s in zip(target_vec, sum_vec):
            self.assertEqual(t, s)

    def test_range_proof_roundtrip(self):
        range_exp = self.config.fp_bits
        x_vec = np.array([-0.75, 1.25, -2.0], dtype='float32')
        blindings = self.ci.create_random_blinding_vector(len(x_vec))
        (range_proof, commitments) = self.ci.create_rangeproof(x_vec, blindings, range_exp, self.config.n_partition)
        assert(self.ci.verify_rangeproof(commitments, range_proof, range_exp))

    def test_rangeproof_generation_exception_wrong_bitsize(self):
        range_exp = 15
        x_vec = np.array([-0.75, 1.25, -2.0], dtype='float32')
        blindings = self.ci.create_random_blinding_vector(len(x_vec))
        exp_msg = "b'Internal error during proof creation: Invalid bitsize, must have n = 8,16,32,64.'"
        self.assertRaisesRegex(ProvingException,           \
                               exp_msg,                    \
                               self.ci.create_rangeproof,  \
                               x_vec,                      \
                               blindings,                  \
                               range_exp,                  \
                               self.config.n_partition)

    def test_rangeproof_generation_exception_wrong_blinding_len(self):
        range_exp = self.config.fp_bits
        x_vec = np.array([-0.75, 1.25, -2.0], dtype='float32')
        blindings = self.ci.create_random_blinding_vector(len(x_vec)+1)
        exp_msg = "b'Internal error during proof creation: Wrong number of blinding factors supplied.'"
        self.assertRaisesRegex(ProvingException,           \
                               exp_msg,                    \
                               self.ci.create_rangeproof,  \
                               x_vec,                      \
                               blindings,                  \
                               range_exp,                  \
                               self.config.n_partition)


    def test_rangeproof_verification_exception_wrong_range(self):
        range_exp = self.config.fp_bits
        wrong_range = range_exp-1 
        x_vec = np.array([-0.75, 1.25, -2.0], dtype='float32')
        blindings_x = self.ci.create_random_blinding_vector(len(x_vec))
        (range_proof, commitments) = self.ci.create_rangeproof(x_vec, blindings_x, range_exp, self.config.n_partition)
        exp_msg = "b'Invalid bitsize, must have n = 8,16,32,64.'"
        self.assertRaisesRegex(VerificationException,       \
                                exp_msg,                    \
                                self.ci.verify_rangeproof,  \
                                commitments,                \
                                range_proof ,               \
                                wrong_range)

    def test_rangeproof_with_cancelling_blindings(self):
        range_exp = self.config.fp_bits
        x_vec = np.array([0.25, 1.25, -1.5], dtype='float32')
        y_vec = np.array([-0.75, 1.25, -2.0], dtype='float32')
        z_vec = np.array([0.5, 1.25, -3.0], dtype='float32')
        target_vec = np.array([0.0, 3.75, -6.5], dtype='float32')

        blindings = self.ci.generate_cancelling_blindings(3, len(x_vec))
        (range_proof_x, commitments_x) = self.ci.create_rangeproof(x_vec, blindings[0], range_exp, self.config.n_partition)
        (range_proof_y, commitments_y) = self.ci.create_rangeproof(y_vec, blindings[1], range_exp, self.config.n_partition)
        (range_proof_z, commitments_z) = self.ci.create_rangeproof(z_vec, blindings[2], range_exp, self.config.n_partition)
        assert(self.ci.verify_rangeproof(commitments_x, range_proof_x, range_exp))
        assert(self.ci.verify_rangeproof(commitments_y, range_proof_y, range_exp))
        assert(self.ci.verify_rangeproof(commitments_z, range_proof_z, range_exp))
        sum_vec_enc = self.ci.add_commitments([commitments_x, commitments_y, commitments_z])
        sum_vec = self.ci.extract_values(sum_vec_enc)
        
        for t, s in zip(target_vec, sum_vec):
            self.assertEqual(t, s)

    def test_randproof_roundtrip(self):
        x_vec = np.array([0.25, 1.25, -1.5], dtype='float32')
        blindings = self.ci.create_random_blinding_vector(len(x_vec))
        (randproof_x, commitments_x, rand_x) = self.ci.create_randproof(x_vec, blindings)
        assert(self.ci.verify_randproof(commitments_x, rand_x, randproof_x))

    def test_fake_randproof_roundtrip(self):
        x_vec = np.array([0.25, 1.25, -1.5], dtype='float32')
        y_vec = np.array([-0.75, 1.25, -2.0], dtype='float32')
        blindings = self.ci.generate_cancelling_blindings(2, len(x_vec))
        (randproof_x, commitments_x, rand_x) = self.ci.create_randproof(x_vec, blindings[0])
        (randproof_y, commitments_y, rand_y) = self.ci.create_randproof(y_vec, blindings[1])
        assert(not self.ci.verify_randproof(commitments_x, rand_x, randproof_y))
        assert(not self.ci.verify_randproof(commitments_y, rand_y, randproof_x))

    def test_randproof_exception_wrong_number_of_blindings(self):
        x_vec = np.array([0.25, 1.25, -1.5], dtype='float32')
        blindings = self.ci.create_random_blinding_vector(len(x_vec)+1)
        exp_msg = "b'Wrong number of blinding factors supplied.'"
        self.assertRaisesRegex(ProvingException,           \
                               exp_msg,                    \
                               self.ci.create_randproof,   \
                               x_vec,                      \
                               blindings)

    def test_randproof_exception_wrong_number_of_eg_pairs(self):
        x_vec = np.array([0.25, 1.25], dtype='float32')
        y_vec = np.array([-0.75, 1.25, -2.0], dtype='float32')
        blindings_x = self.ci.create_random_blinding_vector(len(x_vec))
        blindings_y = self.ci.create_random_blinding_vector(len(y_vec))
        (randproof_x, commitments_x, rand_x) = self.ci.create_randproof(x_vec, blindings_x)
        (randproof_y, commitments_y, rand_y) = self.ci.create_randproof(y_vec, blindings_y)
        exp_msg = "b'Number of ElGamal pairs does not match number of supplied RandProofs'"
        self.assertRaisesRegex(
            VerificationException,      
            exp_msg,
            self.ci.verify_randproof,
            commitments_x,
            rand_x,
            randproof_y)
        self.assertRaisesRegex(
            VerificationException,
            exp_msg,
            self.ci.verify_randproof,
            commitments_y,
            rand_y,
            randproof_x)

    def test_split_elgamal_pair_vec_with_cancelling_blindings(self):
        x_vec = np.array([-0.75, 1.25, -2.0], dtype='float32')
        y_vec = np.array([1.75, -0.5, -1.0], dtype='float32')

        blindings = self.ci.generate_cancelling_blindings(2, len(x_vec))
        (_, _, x_rand) = self.ci.create_randproof(x_vec, blindings[0])
        (_, _, y_rand) = self.ci.create_randproof(y_vec, blindings[1])

        sum_blindings = self.ci.add_commitments([x_rand, y_rand])
        sum_blindings_ext = self.ci.extract_values(sum_blindings)
        target = np.array([0,0,0], dtype='float32')
        self.assertTrue((target == sum_blindings_ext).all())

    def test_randproof_and_rangeproof_return_same_commitment(self):
        prove_range = self.config.fp_bits
        #x_vec = np.array([-0.75, 1.25, -2.0], dtype='float32')
        fp_max = get_fp_max(self.config.fp_bits, self.config.fp_frac)
        fp_min = get_fp_min(self.config.fp_bits, self.config.fp_frac)
        #x_vec = np.array([fp_min-10, fp_max+10, 0.0], dtype='float32')
        #print(x_vec)
        x_vec = np.random.rand(2000)*fp_max*2.2 - fp_max
        #blindings_x = self.ci.generate_cancelling_blindings(1, len(x_vec))[0]
        blindings_x = self.ci.create_zero_group_element_vector(len(x_vec))
        x_vec_clipped = self.ci.clip_to_range(x_vec, self.config.fp_bits)
        (_, x_enc_rand, _) = self.ci.create_randproof(x_vec_clipped, blindings_x)
        (_, x_enc_range) = self.ci.create_rangeproof(x_vec_clipped, blindings_x, prove_range, 4)

        left_enc, right_enc = self.ci.filter_unequal_commits(x_enc_rand, x_enc_range)
        left = self.ci.extract_values(left_enc)
        right  = self.ci.extract_values(right_enc)

        self.assertTrue(self.ci.commits_equal(x_enc_rand, x_enc_range))
        #self.assertEqual(x_enc_rand, x_enc_range)

    def test_commits_equal(self):
        x_vec = np.array([-0.75, 1.25, -2.0], dtype='float32')
        y_vec = np.array([1.75, -0.5, -1.0], dtype='float32')

        blindings = self.ci.generate_cancelling_blindings(2, len(x_vec))
        x_vec_enc = self.ci.commit(x_vec, blindings[0])
        y_vec_enc = self.ci.commit(y_vec, blindings[1])
        self.assertTrue(self.ci.commits_equal(x_vec_enc, x_vec_enc))
        self.assertFalse(self.ci.commits_equal(x_vec_enc, y_vec_enc))

    def test_zero_group_vec(self):
        x_vec = np.array([0.0, 0.0], dtype='float32')
        x_vec_enc = self.ci.commit_no_blinding(x_vec)
        group_zero_vec = self.ci.create_zero_group_element_vector(len(x_vec))
        self.assertEqual(x_vec_enc, group_zero_vec)
        
    def test_equals_neutral_group_element_vector(self):
        x_vec = np.array([0.0, 0.0], dtype='float32')
        x_vec_enc = self.ci.commit_no_blinding(x_vec)
        self.assertTrue(self.ci.equals_neutral_group_element_vector(x_vec_enc))

    def test_filter_unequal_commits(self):
        x_vec = np.array([-0.75, -0.5, -2.0, 1.0], dtype='float32')
        y_vec = np.array([1.75, -0.5, -1.0, 1.0], dtype='float32')

        x_vec_enc = self.ci.commit_no_blinding(x_vec)
        y_vec_enc = self.ci.commit_no_blinding(y_vec)

        left_enc, right_enc = self.ci.filter_unequal_commits(x_vec_enc, y_vec_enc)
        left = self.ci.extract_values(left_enc)
        right = self.ci.extract_values(right_enc)
        unequal_idx = x_vec != y_vec
        self.assertTrue((left == x_vec[unequal_idx]).all())
        self.assertTrue((right == y_vec[unequal_idx]).all())

    def test_squarerandproof_roundtrip(self):
        vec = np.array([0.25, 1.25, -1.5], dtype='float32')
        blindings_1 = self.ci.create_random_blinding_vector(len(vec))
        blindings_2 = self.ci.create_random_blinding_vector(len(vec))
        (randproof_x, commitments_x, rand_x, square_x) = self.ci.create_squarerandproof(vec, blindings_1, blindings_2)
        assert(self.ci.verify_squarerandproof(commitments_x, rand_x, square_x, randproof_x))

    def test_squarerandproof_verify_l2_roundtrip(self):
        x_vec_1 = np.array([0.25, 1.25, -1.5], dtype='float32')
        x_vec_2 = np.array([-0.25, 1.5, -1.25], dtype='float32')

        blindings_2 = self.ci.create_random_blinding_vector(len(x_vec_1))
        def create_verify_get_sum(vec):
            blindings_1 = self.ci.create_random_blinding_vector(len(vec))
            (randproof_x, commitments_x, rand_x, square_x) = self.ci.create_squarerandproof(vec, blindings_1, blindings_2)
            assert(self.ci.verify_squarerandproof(commitments_x, rand_x, square_x, randproof_x))
            sum = self.ci.add_commitments_transposed([square_x])[0]
            return sum

        s_1, s_2 = create_verify_get_sum(x_vec_1), create_verify_get_sum(x_vec_2)
        assert(s_1 == s_2)

    def test_squarerandproof_verify_l2_roundtrip_fail(self):
        x_vec_1 = np.array([0.25, 1.25, -1.5], dtype='float32')
        x_vec_2 = np.array([-0.25, 1.5, -1.35], dtype='float32')

        blindings_2 = self.ci.create_random_blinding_vector(len(x_vec_1))
        def create_verify_get_sum(vec):
            blindings_1 = self.ci.create_random_blinding_vector(len(vec))
            (randproof_x, commitments_x, rand_x, square_x) = self.ci.create_squarerandproof(vec, blindings_1, blindings_2)
            assert(self.ci.verify_squarerandproof(commitments_x, rand_x, square_x, randproof_x))
            sum = self.ci.add_commitments_transposed([square_x])[0]
            return sum

        s_1, s_2 = create_verify_get_sum(x_vec_1), create_verify_get_sum(x_vec_2)
        assert(s_1 != s_2)

    def test_l2proof_allinone_roundtrip(self):
        range_exp = self.config.fp_bits
        vec = np.array([0.25, 1.25, -1.5], dtype='float32')
        # vec = np.array([-7.9], dtype='float32')
        blindings_1 = self.ci.create_random_blinding_vector(len(vec))
        blindings_2 = self.ci.create_random_blinding_vector(len(vec))
        (randproof_x, commitments_x, rand_x, square_x, rangeproof, square_sum) = self.ci.create_l2proof(vec, blindings_1, blindings_2, range_exp, self.config.n_partition)
        assert(self.ci.verify_l2proof(commitments_x, rand_x, square_x, randproof_x, rangeproof, square_sum, range_exp))

    def test_l2proof_allinone_roundtrip_toolarge(self):
        range_exp = self.config.fp_bits
        # vec = np.array([0.25, 1.25, -1.5], dtype='float32')
        print(f"{range_exp}, {self.config.fp_frac}")
        vec = np.array([-8.0], dtype='float32')
        blindings_1 = self.ci.create_random_blinding_vector(len(vec))
        blindings_2 = self.ci.create_random_blinding_vector(len(vec))
        try:
            (randproof_x, commitments_x, rand_x, square_x, rangeproof, square_sum) = self.ci.create_l2proof(vec, blindings_1, blindings_2, range_exp, self.config.n_partition)
            print(randproof_x)
        except ProvingException:
            pass
        except Exception:
            self.fail("Unexpected exception")
        else:
            self.fail("Expected ProvingException")

    def test_l2proof_allinone_roundtrip_random(self):
        range_exp = self.config.fp_bits
        vec = np.random.normal(0, 2.0, 10).astype(dtype='float32')
        # print(vec, np.linalg.norm(vec))
        # vec = np.array([-7.9], dtype='float32')
        blindings_1 = self.ci.create_random_blinding_vector(len(vec))
        blindings_2 = self.ci.create_random_blinding_vector(len(vec))
        (randproof_x, commitments_x, rand_x, square_x, rangeproof, square_sum) = self.ci.create_l2proof(vec, blindings_1, blindings_2, range_exp, self.config.n_partition)
        assert(self.ci.verify_l2proof(commitments_x, rand_x, square_x, randproof_x, rangeproof, square_sum, range_exp))

