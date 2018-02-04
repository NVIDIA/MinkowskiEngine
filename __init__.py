import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

from SparseConvolution import SparseConvolutionFunction, SparseConvolution, Metadata, RegionType
from SparseConvolutionNetwork import SparseConvolutionNetwork
