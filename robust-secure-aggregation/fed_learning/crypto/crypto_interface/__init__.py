import os
import sys
from sys import getsizeof
import cffi
import numpy as np
import logging
from typing import List
import threading
# from eventlet import tpool, sleep

from fed_learning.util.async_tools import run_native
from .exception import *


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if sys.platform == "linux" or sys.platform == "linux2":
    # linux
    LIB_PATH = os.path.join(os.path.dirname(__file__), 'rust_crypto/target/release/librust_crypto.so')
else:
    LIB_PATH = os.path.join(os.path.dirname(__file__), 'rust_crypto/target/release/librust_crypto.dylib')

_FFI = cffi.FFI()
_LIB = None

# singleton _LIB
def _get_bindings():
    global _LIB
    if _LIB is None:
        _FFI.cdef("""  
            typedef struct {
                char* data;
                size_t len;
            } PyVec;

            typedef struct {
                long ret;
                char* msg;
                void* res;
            } PyRes;

            PyVec say_hello();
            PyVec commit(float* value_ptr, size_t value_len, char* blinding_ptr, size_t blinding_len);
            PyVec commit_no_blinding(float* value_ptr, size_t len);
            PyVec add_commitments(char** ptr_arr_ptr, size_t* len_arr_ptr, size_t len);
            PyVec add_commitments_transposed(char** ptr_arr_ptr, size_t* len_arr_ptr, size_t len);
            PyVec extract_values(char* bytes, size_t len);
            PyVec generate_cancelling_blindings(size_t n_vec, size_t size_t);
            PyVec select_blindings(char* blinding_ptr, size_t blinding_len, size_t* value_ptr, size_t value_len);
            PyVec select_commitments(char* commit_ptr, size_t commit_len, size_t* value_ptr, size_t value_len);
            PyRes create_rangeproof(float* value_ptr, size_t value_len, char* blinding_ptr, size_t blinding_len, size_t range, size_t n_partition);
            PyRes verify_rangeproof(char* commit_ptr, size_t commit_len, char* proof_ptr, size_t proof_len, size_t range_exp);
            PyRes create_randproof(float* value_ptr, size_t value_len, char* const u8, size_t blinding_len);
            PyRes verify_randproof(char* commit_ptr, size_t commit_len, char* randproof_ptr, size_t proof_len);
            PyRes create_squarerandproof(float* value_ptr, size_t value_len, char* const blinding_1_ptr, size_t blinding_1_len, char* const blinding_2_ptr, size_t blinding_2_len);
            PyRes verify_squarerandproof(char* commit_ptr, size_t commit_len, char* randproof_ptr, size_t proof_len);
            PyRes create_l2proof(float* value_ptr, size_t value_len, char* const blinding_1_ptr, size_t blinding_1_len, char* const blinding_2_ptr, size_t blinding_2_len, size_t range, size_t n_partition);
            PyRes verify_l2proof(char* commit_ptr, size_t commit_len, char* randproof_ptr, size_t proof_len, char* const range_proof_ptr, char* const square_commit_ptr, size_t range);
            PyVec split_elgamal_pair_vector(char* commit_ptr, size_t commit_len);
            PyVec join_to_elgamal_pair_vector(char* ped_commit_ptr, size_t ped_commit_len, char* rand_commit_ptr, size_t rand_commit_len);
            PyVec split_squaretriple_pair_vector(char* commit_ptr, size_t commit_len);
            PyVec join_to_squaretriple_pair_vector(char* ped_commit_ptr, size_t ped_commit_len, char* rand_commit_ptr, size_t rand_commit_len, char* square_commit_ptr, size_t square_commit_len);
            PyVec clip_to_range(float* value_ptr, size_t value_len, size_t range);
            PyRes commits_equal(char* commit_a_ptr, char* commit_b_ptr, size_t commit_len);
            PyRes equals_neutral_group_element_vec(char* commit_ptr, size_t commit_len);
            PyVec add_scalars(char* commit_ptr, size_t commit_len);
            PyVec create_zero_scalar_vector(size_t len);
            PyVec create_zero_group_element_vector(size_t len);
            PyVec create_random_blinding_vector(size_t len);
            PyVec filter_unequal_commits(char* commit_a_ptr, char* commit_b_ptr, size_t commit_len);
            """
        )
        _LIB = _FFI.dlopen(LIB_PATH)
    return _FFI, _LIB

class CryptoInterface(object):

    def __init__(self):
        self.ffi, self.lib = _get_bindings()

    def commit(self, values: np.ndarray, blindings: bytes) -> bytes:
        value_ptr = self._as_float_ptr(values)
        value_len = self._as_size_t(values.size)
        blinding_ptr = self._as_char_ptr(blindings)
        blinding_len = self._as_size_t(len(blindings))
        pyvec = self.lib.commit(value_ptr, value_len, blinding_ptr, blinding_len)
        return self._unpack_to_bytes(pyvec)

    def commit_no_blinding(self, values: np.ndarray) -> bytes:
        value_ptr = self._as_float_ptr(values)
        value_len = self._as_size_t(values.size)
        pyvec = self.lib.commit_no_blinding(value_ptr, value_len)
        return self._unpack_to_bytes(pyvec)

    @run_native
    def extract_values(self, byte_arr: bytes) -> np.ndarray:
        pyvec = self.lib.extract_values(self._as_char_ptr(byte_arr), self._as_size_t(len(byte_arr)))
        return self._unpack_to_ndarray(pyvec)

    def add_commitments(self, commitments: [bytes]) -> bytes:
        ptr_buf, len_buf = self._create_pointer_array(commitments)
        pyvec = self.lib.add_commitments(ptr_buf, len_buf, self._as_size_t(len(commitments)))
        return self._unpack_to_bytes(pyvec)

    def add_commitments_transposed(self, commitments: [bytes]) -> [bytes]:
        ptr_buf, len_buf = self._create_pointer_array(commitments)
        pyvec = self.lib.add_commitments_transposed(ptr_buf, len_buf, self._as_size_t(len(commitments)))
        pyvec_list = self._unpack_to_pyvec_list(pyvec)
        commits = [self._unpack_to_bytes(x) for x in pyvec_list]
        return commits

    def generate_cancelling_blindings(self, n_vec, n_dim) -> [bytes]:
        n_vec_size_t = self._as_size_t(n_vec)
        n_dim_size_t = self._as_size_t(n_dim)
        pyvec = self.lib.generate_cancelling_blindings(n_vec_size_t, n_dim_size_t)
        pyvec_list = self._unpack_to_pyvec_list(pyvec)
        blindings = [self._unpack_to_bytes(x) for x in pyvec_list]
        return blindings

    def select_blinding_values(self, blindings: bytes, indices: np.ndarray):
        logging.info('Calling select_blinding_values from  thread %s' % threading.current_thread())
        indices_ptr = self._as_int_ptr(indices)
        indices_len = self._as_size_t(indices.size)
        blinding_ptr = self._as_char_ptr(blindings)
        blinding_len = self._as_size_t(len(blindings))
        pyvec = self.lib.select_blindings(blinding_ptr, blinding_len, indices_ptr, indices_len)
        blindings = self._unpack_to_bytes(pyvec)
        # pyvec_list = self._unpack_to_pyvec_list(pyvec)
        # blindings = [self._unpack_to_bytes(x) for x in pyvec_list]
        return blindings

    def select_commitments(self, commitments: bytes, indices: np.ndarray):
        logging.info('Calling select_commitments from  thread %s' % threading.current_thread())
        indices_ptr = self._as_int_ptr(indices)
        indices_len = self._as_size_t(indices.size)
        blinding_ptr = self._as_char_ptr(commitments)
        blinding_len = self._as_size_t(len(commitments))
        pyvec = self.lib.select_commitments(blinding_ptr, blinding_len, indices_ptr, indices_len)
        selected = self._unpack_to_bytes(pyvec)
        return selected

    @run_native
    def create_rangeproof(self, values: np.ndarray, blindings: bytes, range_exp, n_partition) -> (bytes, bytes):
        """
            returns (range_proof, commitments)
        """
        logging.info('Calling create_rangeproof from thread %s' % threading.current_thread())
        value_ptr = self._as_float_ptr(values)
        value_len = self._as_size_t(values.size)
        blinding_ptr = self._as_char_ptr(blindings)
        blinding_len = self._as_size_t(len(blindings))
        range_c = self._as_size_t(range_exp)
        n_partition_c = self._as_size_t(n_partition)
        pyres = self.lib.create_rangeproof(value_ptr, value_len, blinding_ptr, blinding_len, range_c, n_partition_c)
        if not pyres.ret == 0:
            err_msg = self._unpack_pyres_err(pyres)
            logging.error('RangeProof ProvingException: %s' % err_msg)
            raise ProvingException(err_msg)

        res_pyvec = self._as_pyvec_ptr(pyres.res)
        (range_proof_pyvec_ptr, commit_pyvec_ptr) = self._unpack_to_pyvec_list(res_pyvec)
        return (self._unpack_to_bytes(range_proof_pyvec_ptr),
                self._unpack_to_bytes(commit_pyvec_ptr))

    @run_native
    def verify_rangeproof(self, commitment: bytes, rangeproof: bytes, range_exp) -> bool:
        logging.info('Calling verify_rangeproof from thread %s' % threading.current_thread())
        commitment_len = self._as_size_t(len(commitment))
        commitment_ptr = self._as_char_ptr(commitment)
        rangeproof_len = self._as_size_t(len(rangeproof))
        rangeproof_ptr = self._as_char_ptr(rangeproof)
        range_c = self._as_size_t(range_exp)
        pyres = self.lib.verify_rangeproof(commitment_ptr, commitment_len, rangeproof_ptr, rangeproof_len, range_c)

        if not pyres.ret == 0:
            err_msg = self._unpack_pyres_err(pyres)
            logging.error('RangeProof VerificationException: %s' % err_msg)
            raise VerificationException(err_msg)

        return self.ffi.cast('bool*', pyres.res)[0]

    @run_native
    def create_randproof(self, values: np.ndarray, blindings: bytes) -> (bytes, bytes, bytes):
        """
            returns (rand_proof, ped_commitments, rand_commitments)
        """
        logging.info('Calling create_randproof from thread %s' % threading.current_thread())
        value_ptr = self._as_float_ptr(values)
        value_len = self._as_size_t(values.size)
        blinding_ptr = self._as_char_ptr(blindings)
        blinding_len = self._as_size_t(len(blindings))

        pyres = self.lib.create_randproof(value_ptr, value_len, blinding_ptr, blinding_len)
        if not pyres.ret == 0:
            err_msg = self._unpack_pyres_err(pyres)
            logging.error('RandProof ProvingException: %s' % err_msg)
            raise ProvingException(err_msg)

        res_pyvec = self._as_pyvec_ptr(pyres.res)
        (rand_proof_pyvec_ptr, commit_pyvec_ptr) = self._unpack_to_pyvec_list(res_pyvec)
        randproofs = self._unpack_to_bytes(rand_proof_pyvec_ptr)
        eg_commits = self._unpack_to_bytes(commit_pyvec_ptr)
        ped_commits, rand_commits = self.split_elgamal_pair_vector(eg_commits)
        return randproofs, ped_commits, rand_commits

    @run_native
    def verify_randproof(self, ped_commits: bytes, rand_commits: bytes, randproof: bytes) -> bool:
        logging.info('Calling verify_randproof from thread %s' % threading.current_thread())
        eg_commits = self.join_to_elgamal_pair_vector(ped_commits, rand_commits)
        eg_commits_len = self._as_size_t(len(eg_commits))
        eg_commits_ptr = self._as_char_ptr(eg_commits)
        randproof_len = self._as_size_t(len(randproof))
        randproof_ptr = self._as_char_ptr(randproof)
        pyres = self.lib.verify_randproof(eg_commits_ptr, eg_commits_len, randproof_ptr, randproof_len)

        if not pyres.ret == 0:
            err_msg = self._unpack_pyres_err(pyres)
            logging.error('RandProof VerificationException: %s' % err_msg)
            raise VerificationException(err_msg)

        return self.ffi.cast('bool*', pyres.res)[0]

    @run_native
    def create_squarerandproof(self, values: np.ndarray, blindings_1: bytes, blindings_2: bytes) -> (bytes, bytes, bytes, bytes):
        """
            returns (rand_proof, ped_commitments, rand_commitments)
        """
        logging.info('Calling create_squarerandproof from thread %s' % threading.current_thread())
        value_ptr = self._as_float_ptr(values)
        value_len = self._as_size_t(values.size)
        blinding_1_ptr = self._as_char_ptr(blindings_1)
        blinding_1_len = self._as_size_t(len(blindings_1))
        blinding_2_ptr = self._as_char_ptr(blindings_2)
        blinding_2_len = self._as_size_t(len(blindings_2))

        pyres = self.lib.create_squarerandproof(value_ptr, value_len, blinding_1_ptr, blinding_1_len, blinding_2_ptr, blinding_2_len)
        if not pyres.ret == 0:
            err_msg = self._unpack_pyres_err(pyres)
            logging.error('SquareRandProof ProvingException: %s' % err_msg)
            raise ProvingException(err_msg)

        res_pyvec = self._as_pyvec_ptr(pyres.res)
        (rand_proof_pyvec_ptr, commit_pyvec_ptr) = self._unpack_to_pyvec_list(res_pyvec)
        randproofs = self._unpack_to_bytes(rand_proof_pyvec_ptr)
        eg_commits = self._unpack_to_bytes(commit_pyvec_ptr)
        ped_commits, rand_commits, square_commits = self.split_square_triple_vector(eg_commits)
        return randproofs, ped_commits, rand_commits, square_commits

    @run_native
    def verify_squarerandproof(self, ped_commits: bytes, rand_commits: bytes, square_commits: bytes, randproof: bytes) -> bool:
        logging.info('Calling verify_squarerandproof from thread %s' % threading.current_thread())
        square_commits = self.join_to_square_triple_vector(ped_commits, rand_commits, square_commits)
        square_commits_len = self._as_size_t(len(square_commits))
        square_commits_ptr = self._as_char_ptr(square_commits)
        randproof_len = self._as_size_t(len(randproof))
        randproof_ptr = self._as_char_ptr(randproof)
        pyres = self.lib.verify_squarerandproof(square_commits_ptr, square_commits_len, randproof_ptr, randproof_len)

        if not pyres.ret == 0:
            err_msg = self._unpack_pyres_err(pyres)
            logging.error('RandProof VerificationException: %s' % err_msg)
            raise VerificationException(err_msg)

        return self.ffi.cast('bool*', pyres.res)[0]

    @run_native
    def create_l2proof(self, values: np.ndarray, blindings_1: bytes, blindings_2: bytes, range_exp, n_partition) -> (bytes, bytes, bytes, bytes, bytes, bytes):
        """
            returns (rand_proof, ped_commitments, rand_commitments, square_commits, range_proof, square_commit)
        """
        logging.info('Calling create_squarerandproof from thread %s' % threading.current_thread())
        value_ptr = self._as_float_ptr(values)
        value_len = self._as_size_t(values.size)
        blinding_1_ptr = self._as_char_ptr(blindings_1)
        blinding_1_len = self._as_size_t(len(blindings_1))
        blinding_2_ptr = self._as_char_ptr(blindings_2)
        blinding_2_len = self._as_size_t(len(blindings_2))
        range_c = self._as_size_t(range_exp)
        n_partition_c = self._as_size_t(n_partition)

        pyres = self.lib.create_l2proof(value_ptr, value_len, blinding_1_ptr, blinding_1_len, blinding_2_ptr, blinding_2_len, range_c, n_partition_c)
        if not pyres.ret == 0:
            err_msg = self._unpack_pyres_err(pyres)
            logging.error('SquareRandProof ProvingException: %s' % err_msg)
            raise ProvingException(err_msg)

        res_pyvec = self._as_pyvec_ptr(pyres.res)
        (rand_proof_pyvec_ptr, commit_pyvec_ptr, range_proof_ptr, square_commit_ptr) = self._unpack_to_pyvec_list(res_pyvec)
        randproofs = self._unpack_to_bytes(rand_proof_pyvec_ptr)
        eg_commits = self._unpack_to_bytes(commit_pyvec_ptr)
        range_proof = self._unpack_to_bytes(range_proof_ptr)
        square_commit = self._unpack_to_bytes(square_commit_ptr)
        ped_commits, rand_commits, square_commits = self.split_square_triple_vector(eg_commits)
        return randproofs, ped_commits, rand_commits, square_commits, range_proof, square_commit

    @run_native
    def verify_l2proof(self, ped_commits: bytes, rand_commits: bytes, square_commits: bytes, randproof: bytes, rangeproof: bytes, square_commit_sum: bytes, range_exp) -> bool:
        logging.info('Calling verify_squarerandproof from thread %s' % threading.current_thread())
        square_commits = self.join_to_square_triple_vector(ped_commits, rand_commits, square_commits)
        square_commits_len = self._as_size_t(len(square_commits))
        square_commits_ptr = self._as_char_ptr(square_commits)
        randproof_len = self._as_size_t(len(randproof))
        randproof_ptr = self._as_char_ptr(randproof)
        rangeproof_ptr = self._as_char_ptr(rangeproof)
        square_commit_sum_ptr = self._as_char_ptr(square_commit_sum)
        pyres = self.lib.verify_l2proof(square_commits_ptr, square_commits_len, randproof_ptr, randproof_len, rangeproof_ptr, square_commit_sum_ptr, range_exp)

        if not pyres.ret == 0:
            err_msg = self._unpack_pyres_err(pyres)
            logging.error('RandProof VerificationException: %s' % err_msg)
            raise VerificationException(err_msg)

        return self.ffi.cast('bool*', pyres.res)[0]

    def split_elgamal_pair_vector(self, commitments: bytes) -> (bytes, bytes):
        commit_ptr = self._as_char_ptr(commitments)
        commit_len = self._as_size_t(len(commitments))
        res_pyvec = self.lib.split_elgamal_pair_vector(commit_ptr, commit_len)
        (left_commit_ptr, right_commit_ptr) = self._unpack_to_pyvec_list(res_pyvec)
        return (self._unpack_to_bytes(left_commit_ptr),
                self._unpack_to_bytes(right_commit_ptr))

    def join_to_elgamal_pair_vector(self, ped_commits: bytes, rand_commits: bytes) -> bytes:
        ped_commit_ptr = self._as_char_ptr(ped_commits)
        ped_commit_len = self._as_size_t(len(ped_commits))
        rand_commit_ptr = self._as_char_ptr(rand_commits)
        rand_commit_len = self._as_size_t(len(rand_commits))
        pyvec = self.lib.join_to_elgamal_pair_vector(
            ped_commit_ptr,
            ped_commit_len,
            rand_commit_ptr,
            rand_commit_len
        )
        return self._unpack_to_bytes(pyvec)

    def split_square_triple_vector(self, commitments: bytes) -> (bytes, bytes):
        commit_ptr = self._as_char_ptr(commitments)
        commit_len = self._as_size_t(len(commitments))
        res_pyvec = self.lib.split_squaretriple_pair_vector(commit_ptr, commit_len)
        (left_commit_ptr, right_commit_ptr, square_commit_ptr) = self._unpack_to_pyvec_list(res_pyvec)
        return (self._unpack_to_bytes(left_commit_ptr),
                self._unpack_to_bytes(right_commit_ptr),
                self._unpack_to_bytes(square_commit_ptr))

    def join_to_square_triple_vector(self, ped_commits: bytes, rand_commits: bytes, square_commits: bytes) -> bytes:
        ped_commit_ptr = self._as_char_ptr(ped_commits)
        ped_commit_len = self._as_size_t(len(ped_commits))
        rand_commit_ptr = self._as_char_ptr(rand_commits)
        rand_commit_len = self._as_size_t(len(rand_commits))
        square_commit_ptr = self._as_char_ptr(square_commits)
        square_commit_len = self._as_size_t(len(square_commits))
        pyvec = self.lib.join_to_squaretriple_pair_vector(
            ped_commit_ptr,
            ped_commit_len,
            rand_commit_ptr,
            rand_commit_len,
            square_commit_ptr,
            square_commit_len
        )
        return self._unpack_to_bytes(pyvec)

    def clip_to_range(self, values: np.ndarray, range) -> np.ndarray:
        value_ptr = self._as_float_ptr(values)
        value_len = self._as_size_t(values.size)
        range_c = self._as_size_t(range)
        pyvec = self.lib.clip_to_range(value_ptr, value_len, range_c)
        return self._unpack_to_ndarray(pyvec)

    def create_zero_scalar_vector(self, len) -> bytes:
        len_c = self._as_size_t(len)
        pyvec = self.lib.create_zero_scalar_vector(len_c)
        return self._unpack_to_bytes(pyvec)

    def commits_equal(self, commits_a: bytes, commits_b: bytes) -> bool:
        if not len(commits_a) == len(commits_b):
            return False
        len_c = self._as_size_t(len(commits_a))
        commits_a_ptr = self._as_char_ptr(commits_a)
        commits_b_ptr = self._as_char_ptr(commits_b)
        pyres = self.lib.commits_equal(commits_a_ptr, commits_b_ptr, len_c)
        return self.ffi.cast('bool*', pyres.res)[0]

    def equals_neutral_group_element_vector(self, commits: bytes) -> bool:
        len_c = self._as_size_t(len(commits))
        commits_ptr = self._as_char_ptr(commits)
        pyres = self.lib.equals_neutral_group_element_vec(commits_ptr, len_c)
        return self.ffi.cast('bool*', pyres.res)[0]

    def add_scalars(self, scalars: bytes) -> bytes:
        len_c = self._as_size_t(len(scalars))
        commits_ptr = self._as_char_ptr(scalars)
        pyvec = self.lib.add_scalars(commits_ptr, len_c)
        return self._unpack_to_bytes(pyvec)

    def create_zero_group_element_vector(self, len) -> bytes:
        len_c = self._as_size_t(len)
        pyvec = self.lib.create_zero_group_element_vector(len_c)
        return self._unpack_to_bytes(pyvec)

    def create_random_blinding_vector(self, len) -> bytes:
        len_c = self._as_size_t(len)
        pyvec = self.lib.create_random_blinding_vector(len_c)
        return self._unpack_to_bytes(pyvec)

    def filter_unequal_commits(self, commits_a, commits_b) -> (bytes, bytes):
        assert(len(commits_a) == len(commits_b))
        len_c = self._as_size_t(len(commits_a))
        commits_a_ptr = self._as_char_ptr(commits_a)
        commits_b_ptr = self._as_char_ptr(commits_b)
        #print(self.commits_equal(commits_a, commits_b))
        res_pyvec = self.lib.filter_unequal_commits(commits_a_ptr, commits_b_ptr, len_c)
        (left_pyvec, right_pyvec) = self._unpack_to_pyvec_list(res_pyvec)
        left_commit_vec = self._unpack_to_bytes(left_pyvec)
        right_commit_vec = self._unpack_to_bytes(right_pyvec)
        return left_commit_vec, right_commit_vec

    def sum_of_squares(self, values: np.ndarray, range) -> np.ndarray:
        value_ptr = self._as_float_ptr(values)
        value_len = self._as_size_t(values.size)
        range_c = self._as_size_t(range)
        pyres = self.lib.clip_to_range(value_ptr, value_len, range_c)
        return self.ffi.cast('bool*', pyres.res)[0]

    def _create_pointer_array(self, data :[bytes]):
        assert(isinstance(data, list))
        ptr_list = [self._as_char_ptr(d) for d in data]
        len_list = [self._as_size_t(len(d)) for d in data]

        ptr_buf = self.ffi.new("char*[]", ptr_list)
        len_buf = self.ffi.new("size_t[]", len_list)

        return ptr_buf, len_buf
    
    # TODO mlei: void* instead char*
    def _as_char_ptr(self, b: bytes):
        assert(isinstance(b, bytes))
        return self.ffi.cast('char*', self.ffi.from_buffer(b))

    def _as_double_ptr(self, array: np.ndarray):
        """
        Cast a np.float64 array to a double*.
        """
        assert(isinstance(array, np.ndarray))
        return self.ffi.cast('double*', array.ctypes.data)

    def _as_float_ptr(self, array: np.ndarray):
        """
        Cast a np.float32 array to a float*.
        """
        assert(isinstance(array, np.ndarray))
        return self.ffi.cast('float*', array.ctypes.data)

    def _as_int_ptr(self, array: np.ndarray):
        """
        Cast a np.float32 array to a float*.
        """
        assert(isinstance(array, np.ndarray))
        return self.ffi.cast('size_t*', array.ctypes.data)

    def _as_size_t(self, num):
        """
        Cast num to something like a rust usize.
        """
        return self.ffi.cast('size_t', num)

    def _as_pyvec_ptr(self, ptr):
        return self.ffi.cast('PyVec*', ptr)

    def _unpack_to_bytes(self, pyvec) -> bytes:
        return self.ffi.unpack(pyvec.data, pyvec.len)

    def _unpack_to_ndarray(self, pyvec) -> List[np.ndarray]:
        # TODO mlei: what if client and server have different bit arch?
        byte_length = pyvec.len*np.dtype(np.float32).itemsize
        return np.frombuffer(self.ffi.buffer(pyvec.data, byte_length), np.dtype(np.float32))

    def _unpack_to_pyvec_list(self, pyvec) -> []:
        pyvec_arr_ptr = self.ffi.cast('PyVec*', pyvec.data)
        return self.ffi.unpack(pyvec_arr_ptr, pyvec.len)

    def _unpack_pyres_err(self, pyres) -> str:
        err_msg_ptr = self.ffi.cast('char*', pyres.msg)
        return self.ffi.string(err_msg_ptr)
