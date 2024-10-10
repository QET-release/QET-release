# MIT License
# Copyright (c) 2024 QET-release

import numpy as np
from .qet_vanilla import QET_Vanilla

def rtn_quant(vecs, mem_in_byte=None, cell_bit=None):
    """
    Quantizes the input vectors based on the given memory size and cell bit width.

    Parameters
    ----------
    vecs : ndarray
        The input vectors to be quantized, as a NumPy array of shape (n_rows, n_cols).
    mem_in_byte : int, optional
        Total memory size in bytes allocated for storing the quantized vectors.
    cell_bit : int, optional
        The number of bits per quantization cell. If None, it is computed based on mem_in_byte.

    Returns
    -------
    vecs_quant : ndarray
        The quantized vectors.
    quant_size : float
        The quantization step size.
    min_value : float
        The minimum value in the original vectors (used for reconstructing the original vectors).
    """
    # Get the dimensions of the input vectors
    n_row, n_col = vecs.shape

    # If cell_bit is not provided, calculate it based on the available memory size
    if cell_bit is None:
        # Compute the number of bits per cell using the total memory size
        cell_bit = mem_in_byte * 8 // (n_row * n_col)

    # Calculate the range and step size for quantization
    max_value = np.max(vecs)
    min_value = np.min(vecs)
    quant_levels = (2 ** cell_bit) - 1
    quant_size = (max_value - min_value) / quant_levels

    # Perform quantization
    vecs_quant = np.round((vecs - min_value) / quant_size)
    # Ensure the quantized values remain within the valid range [0, quant_levels]
    vecs_quant = np.clip(vecs_quant, 0, quant_levels)

    # Convert quant_size and min_value to the same dtype as vecs
    return vecs_quant, quant_size.astype(vecs.dtype), min_value.astype(vecs.dtype)

def rtn_dequant(vecs_quant, quant_size, min_value):
    """
    Reconstructs the original vectors from the quantized vectors.

    Parameters
    ----------
    vecs_quant : ndarray
        The quantized vectors.
    quant_size : float
        The quantization step size used during quantization.
    min_value : float
        The minimum value used during quantization.

    Returns
    -------
    vecs_reconstructed : ndarray
        The reconstructed vectors after inverse quantization.
    """
    # Reverse the quantization process to reconstruct the original vectors
    vecs_reconstructed = vecs_quant * quant_size + min_value

    return vecs_reconstructed

class QET_Config:
    """
    Configuration class for QET. Holds parameters related to quantization and memory allocation.
    """
    def __init__(self, quant_num, mem_rate, cbo_bit, qet_layer):
        self.quant_num = quant_num      # Number of quantization steps
        self.mem_rate = mem_rate        # Memory allocation rate for each step
        self.cbo_bit = cbo_bit          # Bits per codebook entry
        self.qet_layer = qet_layer      # Number of layers in QET

# Default configuration for QET
Default_Config = QET_Config(
    quant_num=2, mem_rate=[0.7, 0.3], cbo_bit=[10, 10], qet_layer=[3, 0])

def get_Ks(n_row, n_col, M, mem_in_byte, dtype, cbo_bit=10):
    """
    Calculate the maximum number of codewords (Ks) that can be used within a given memory size.

    Args:
        n_row (int): Number of rows (vectors) in the dataset.
        n_col (int): Dimension of each vector.
        M (int): Number of subspaces for Product Quantization.
        mem_in_byte (int): Available memory size in bytes.
        dtype (data-type): Data type of the input vectors.
        cbo_bit (int): Bits per codebook entry.

    Returns:
        int: The maximum number of codewords that can be used.
    """
    if mem_in_byte <= 0:
        return 0

    cell_bytes = np.dtype(dtype).itemsize  # Size of each data point in bytes

    # Binary search for the maximum Ks that fits within the memory size
    low, high = 1, n_row
    max_Ks = 0
    
    while low < high - 1:
        mid = (low + high) // 2
        bits_per_code = (mid - 1).bit_length()  # Calculate bits needed to represent codewords
        
        # Calculate memory required for codewords and encoded data
        codewords_memory = mid * n_col * cbo_bit + 2 * cell_bytes * 8
        codes_memory = n_row * M * bits_per_code
        
        total_memory = (codewords_memory + codes_memory + 7) // 8

        if total_memory <= mem_in_byte:
            low = mid
        else:
            high = mid
    
    max_Ks = low
    return max_Ks

class QET:
    """
    A class that implements Quantum Entanglement Tree (QET) for high-dimensional vector quantization,
    based on Product Quantization (PQ).
    """

    def __init__(self, M, MemorySize=None, Ks=None, metric="l2", verbose=False, QetConfig=None):
        """
        Initialize the QET encoder with an entanglement tree.

        Args:
            M (int): Number of subspaces (must divide the vector dimension evenly).
            MemorySize (int): Memory size available for the encoder, in bytes.
            Ks (int, optional): The number of codewords for each subspace. Default is None.
            metric (str): The distance metric to use ("l2" or "dot").
            verbose (bool): Flag for detailed logging.
            QetConfig (QET_Config, optional): Configuration object for QET.
        """
        assert metric in ["l2", "dot"], "Metric must be 'l2' or 'dot'."
        self.M = M
        self.MemorySize = MemorySize
        self.metric = metric
        self.verbose = verbose

        # Use the default configuration if none is provided
        self.QetConfig = Default_Config if QetConfig is None else QetConfig

        # Initialize internal variables
        self.qetv = [None] * self.QetConfig.quant_num
        self.Ks = [None] * self.QetConfig.quant_num
        self.cbo_vecs = [None] * self.QetConfig.quant_num
        self.cbo_step = [None] * self.QetConfig.quant_num
        self.cbo_min = [None] * self.QetConfig.quant_num

    def __eq__(self, other):
        """
        Check equality with another QET instance.

        Args:
            other (QET): Another instance to compare with.

        Returns:
            bool: True if both instances are equal, False otherwise.
        """
        if not isinstance(other, QET):
            return False
        return (self.qetv == other.qetv and self.QetConfig == other.QetConfig)

    @property
    def Ds(self):
        """
        Get the dimension of each sub-vector.

        Returns:
            int: Dimension of each sub-vector, Ds = D / M.
        """
        return self.qetv.pq.Ds

    def fit(self, vecs, pq_iter=20, seed=123, minit="points"):
        """
        Fit the Product Quantization (PQ) model with the entangled vectors.

        Args:
            vecs (np.ndarray): Input vectors of shape (N, D).
            pq_iter (int): Number of iterations for training the PQ model.
            seed (int): Random seed for initialization.
            minit (str): Initialization method for PQ.

        Returns:
            QET: The fitted model instance.
        """
        assert vecs.ndim == 2, "Input vectors should be a 2D array."
        N, D = vecs.shape
        InputType = vecs.dtype
        assert D % self.M == 0, "Input dimension must be divisible by M."

        # Determine the maximum Ks if not provided
        if self.Ks[0] is None:
            memory = (self.MemorySize * 8 - (N * D * 0.5) * (sum(self.QetConfig.qet_layer))) // 8
            for i in range(self.QetConfig.quant_num):
                self.Ks[i] = get_Ks(N, D, self.M, int(memory * self.QetConfig.mem_rate[i]), InputType, self.QetConfig.cbo_bit[i])
        
        # Initialize the PQ models for each quantization step
        for i in range(self.QetConfig.quant_num):
            self.qetv[i] = QET_Vanilla(M=self.M, Ks=self.Ks[i], metric=self.metric, qetv_layer=self.QetConfig.qet_layer[i], verbose=self.verbose)
        
        input_vecs = vecs
        for i in range(self.QetConfig.quant_num):
            # Fit the current PQ model
            self.qetv[i].fit(vecs=input_vecs, pq_iter=pq_iter, seed=seed, minit=minit)
            
            # Quantize the codewords using the RTN method
            n1, n2, n3 = self.qetv[i].pq.codewords.shape
            codewords = self.qetv[i].pq.codewords.reshape(n1, n2 * n3)
            self.cbo_vecs[i], self.cbo_step[i], self.cbo_min[i] = rtn_quant(vecs=codewords, cell_bit=self.QetConfig.cbo_bit[i])
            
            # Update input_vecs for the next quantization step
            if i < self.QetConfig.quant_num - 1:
                recon_codewords = rtn_dequant(self.cbo_vecs[i], self.cbo_step[i], self.cbo_min[i])
                self.qetv[i].pq.codewords = recon_codewords.reshape(n1, n2, n3)
                codes, imap = self.qetv[i].encode(vecs=input_vecs)
                recon_vecs = self.qetv[i].decode(codes=codes, imap=imap)
                input_vecs = input_vecs - recon_vecs
            
            # Clear memory to free space
            self.qetv[i].pq.codewords = None

        return self

    def encode(self, vecs):
        """
        Encode the input vectors using the PQ model after performing entanglement.

        Args:
            vecs (np.ndarray): Input vectors of shape (N, D).

        Returns:
            tuple:
                codes (list of np.ndarray): The PQ codes of the vectors for each quantization step.
                imap (list of np.ndarray): The indicator maps from the entanglement process for each step.
        """
        assert vecs.dtype == self.cbo_step[0].dtype, "Input vectors must have the same dtype as the codewords."
        assert vecs.ndim == 2, "Input vectors should be a 2D array."
        input_vecs = vecs
        _, D = vecs.shape
        codes = [None] * self.QetConfig.quant_num
        imap = [None] * self.QetConfig.quant_num

        for i in range(self.QetConfig.quant_num):
            # Reconstruct and set PQ codewords for encoding
            recon_codewords = rtn_dequant(self.cbo_vecs[i], self.cbo_step[i], self.cbo_min[i])
            self.qetv[i].pq.codewords = recon_codewords.reshape(self.M, self.Ks[i], D // self.M)
            
            # Perform encoding
            codes[i], imap[i] = self.qetv[i].encode(vecs=input_vecs)
            
            # Prepare input_vecs for next step encoding
            if i < self.QetConfig.quant_num - 1:
                recon_vecs = self.qetv[i].decode(codes=codes[i], imap=imap[i])
                input_vecs = input_vecs - recon_vecs
            
            # Clear PQ codewords to release memory
            self.qetv[i].pq.codewords = None
    
        return codes, imap

    def decode(self, codes, imap):
        """
        Decode the PQ codes and reverse the entanglement process to reconstruct the original vectors.

        Args:
            codes (list of np.ndarray): The PQ codes representing the vectors for each quantization step.
            imap (list of np.ndarray): The indicator maps from the entanglement process for each step.

        Returns:
            np.ndarray: The reconstructed vectors.
        """
        recon_vecs = [None] * self.QetConfig.quant_num
        _, _, d = imap[0].shape

        for i in range(self.QetConfig.quant_num):
            # Reconstruct PQ codewords for decoding
            recon_codewords = rtn_dequant(self.cbo_vecs[i], self.cbo_step[i], self.cbo_min[i])
            self.qetv[i].pq.codewords = recon_codewords.reshape(self.M, self.Ks[i], 2 * d // self.M)
            
            # Decode the vectors for this quantization step
            recon_vecs[i] = self.qetv[i].decode(codes=codes[i], imap=imap[i])
            
            # Clear PQ codewords to free memory
            self.qetv[i].pq.codewords = None
    
        return sum(recon_vecs)
