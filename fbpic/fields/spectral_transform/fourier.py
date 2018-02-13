# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Partile-In-Cell code (FB-PIC)
It defines the FFT object, which performs Fourier transforms along the axis 0,
and is used in spectral_transformer.py
"""
import numpy as np
import numba
# Check if CUDA is available, then import CUDA functions
from fbpic.utils.cuda import cuda_installed
if cuda_installed:
    from pyculib import fft as cufft, blas as cublas
    from fbpic.utils.cuda import cuda, cuda_tpb_bpg_2d
    from .cuda_methods import cuda_copy_2d_to_1d, cuda_copy_1d_to_2d
# Check if the MKL FFT is available
try:
    from .mkl_fft import MKLFFT
    mkl_installed = True
except OSError:
    import pyfftw
    mkl_installed = False

class FFT(object):
    """
    Object that performs Fourier transform of 2D arrays along the z axis,
    (axis 0) either on the CPU (using pyfftw) or on the GPU (using cufft)

    See the methods `transform` and `inverse transform` for more information
    """

    def __init__(self, Nr, Nz, N_kz, interp_data_type, m, 
                 use_cuda=False, nthreads=None ):
        """
        Initialize an FFT object

        Parameters
        ----------
        Nr, Nz: int
            Number of grid points along the z and r axis (axis 0 and -1)

        N_kz: int
            Number of points in k space, along the z axis
            (For modes m>0, N_kz is equal to Nz ; but for mode m=0,
            N_kz is about half of Nz, because the interpolation grid
            can be represented by real numbers.)

        interp_data_type: a numpy type
            Either np.float64 or np.complex128
            The type of the data on the interpolation grid 
            (typically float for m=0 and complex for m>0)

        m: int
            The index of the azimuthal mode

        use_cuda: bool, optional
           Whether to perform the Fourier transform on the z axis

        nthreads : int, optional
            Number of threads for the FFTW transform.
            If None, the default number of threads of numba is used
            (environment variable NUMBA_NUM_THREADS)
        """
        # Check whether to use cuda
        self.use_cuda = use_cuda
        if (self.use_cuda is True) and (cuda_installed is False) :
            self.use_cuda = False
            print('** Cuda not available for Fourier transform.')
            print('** Performing the Fourier transform on the CPU.')

        # Check whether to use MKL
        self.use_mkl = mkl_installed

        # Initialize the object for calculation on the GPU
        if self.use_cuda:
            # Initialize the dimension of the grid and blocks
            self.dim_grid, self.dim_block = cuda_tpb_bpg_2d( Nz, Nr)

            # Initialize 1d buffer for cufft
            self.buffer1d_in = cuda.device_array(
                (Nz*Nr,), dtype=np.complex128)
            self.buffer1d_out = cuda.device_array(
                (Nz*Nr,), dtype=np.complex128)
            # Initialize the cuda libraries object
            self.fft = cufft.FFTPlan( shape=(Nz,), itype=np.complex128,
                                      otype=np.complex128, batch=Nr )
            self.blas = cublas.Blas()   # For normalization of the iFFT
            self.inv_Nz = 1./Nz         # For normalization of the iFFT

        # Initialize the object for calculation on the CPU
        else:

            # Initialize dummy arrays for the FFT plan
            interp_buffer = np.zeros( (Nz, Nr), dtype=interp_data_type )
            spect_buffer = np.zeros( (N_kz, Nr), dtype=np.complex128 )

            # For MKL FFT
            if self.use_mkl:
                self.mklfft = MKLFFT( spect_buffer )
            # For FFTW
            else:
                # Determine number of threads
                if nthreads is None:
                    # Get the default number of threads for numba
                    nthreads = numba.config.NUMBA_NUM_THREADS
                self.fft = pyfftw.FFTW( interp_buffer, spect_buffer,
                        axes=(0,), direction='FFTW_FORWARD', threads=nthreads)
                self.ifft = pyfftw.FFTW( spect_buffer, interp_buffer,
                        axes=(0,), direction='FFTW_BACKWARD', threads=nthreads)


    def transform( self, array_in, array_out ):
        """
        Perform the Fourier transform of array_in,
        and store the result in array_out

        Parameters
        ----------
        array_in, array_out: cuda device arrays or numpy arrays
            When using the GPU, these should be cuda device array.
            When using the CPU, array_out should be one of the
            two buffers that are returned by `get_buffers`
        """
        if self.use_cuda :
            # Perform the FFT on the GPU
            # (The cuFFT API requires 1D arrays)
            cuda_copy_2d_to_1d[self.dim_grid, self.dim_block](
                array_in, self.buffer1d_in )
            self.fft.forward( self.buffer1d_in, out=self.buffer1d_out )
            cuda_copy_1d_to_2d[self.dim_grid, self.dim_block](
                self.buffer1d_out, array_out )
        elif self.use_mkl:
            # Perform the FFT on the CPU using MKL
            self.mklfft.transform( array_in, array_out )
        else :
            # Perform the FFT on the CPU using FFTW
            self.fft.update_arrays( new_input_array=array_in,
                                    new_output_array=array_out )
            self.fft()

    def inverse_transform( self, array_in, array_out ):
        """
        Perform the inverse Fourier transform of array_in,
        and store the result in array_out

        Parameters
        ----------
        array_in, array_out: cuda device arrays or numpy arrays
            When using the GPU, these should be cuda device array.
            When using the CPU, array_in should be one of the
            two buffers that are returned by `get_buffers`
        """
        if self.use_cuda :
            # Perform the inverse FFT on the GPU
            # (The cuFFT API requires 1D arrays)
            cuda_copy_2d_to_1d[self.dim_grid, self.dim_block](
                array_in, self.buffer1d_in )
            self.fft.inverse( self.buffer1d_in, out=self.buffer1d_out )
            self.blas.scal( self.inv_Nz, self.buffer1d_out ) # Normalization
            cuda_copy_1d_to_2d[self.dim_grid, self.dim_block](
                self.buffer1d_out, array_out )
        elif self.use_mkl:
            # Perform the inverse FFT on the CPU using MKL
            self.mklfft.inverse_transform( array_in, array_out )
        else :
            # Perform the inverse FFT on the CPU using FFTW
            self.ifft.update_arrays( new_input_array=array_in,
                                    new_output_array=array_out )
            self.ifft()
