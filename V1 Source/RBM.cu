/******************************************************************************/
/*                                                                            */
/*  RBM.CU - Core CUDA routines for RBM                                       */
/*                                                                            */
/******************************************************************************/

#define STRICT
#include <windows.h>
#include <commctrl.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include <malloc.h>
#include <new.h>
#include <float.h>
#include <process.h>

#include <driver_types.h>
#include <cuda_runtime_api.h>

#include "deep.rh"
#include "const.h"
#include "classes.h"
#include "extern.h"
#include "funcdefs.h"

// These are for the reductions used in device_len_dot and in device_max_inc/w.
// The number of threads MUST be a power of two!
// The number of blocks given here is a maximum.  The actual number may be less.

#define REDUC_THREADS 256
#define REDUC_BLOCKS 64

static float *reduc_fdata = NULL ;


// This is used as intermediary between device's float and hosts double

static float *fdata = NULL ;


// These are set in ?_cuda_init and used by the host routine that launches the kernel
// They are basic app parameters, constant for all launches
// Names that begin with d_ are in the device namespace.
// Names that begin with h_ are in the host namespace and equal the device value.
// This lets us save a little time by avoiding the need to pass a bunch of parameters in the launch.
// We could, of course, just pass data pointers as parameters.  But that's overhead.
// So instead we use cudaMemcpyToSymbol() to copy the values in the host namespace
// to values on the device.  This lets __global routines address the values that are
// already set on the device rather than having to use passed parameters.
// The savings is probably small, but worthwhile.

__constant__ int d_ncases ;        // Number of cases (needed for using shuffle_index as random sampler)
__constant__ int d_n_inputs ;      // Number of inputs (size of visible, bottom layer)
__constant__ int d_n_inputs_cols ; // Ditto, extended to multiple of 128 bytes
__constant__ int d_nhid ;          // Number of hidden neurons
__constant__ int d_nhid_cols ;     // Ditto, extended to multiple of 128 bytes
__constant__ int d_mean_field ;    // Use mean field instead of random sampling?
__constant__ int d_greedy_mean_field ;    // Use mean field for greedy training?

static       float *h_data = NULL ;
__constant__ float *d_data ;
static       float *h_data_mean = NULL ;
__constant__ float *d_data_mean ;
static       float *h_in_bias = NULL ;
__constant__ float *d_in_bias ;
static       float *h_hid_bias = NULL ;
__constant__ float *d_hid_bias ;
static       float *h_w = NULL ;
__constant__ float *d_w ;
static       float *h_wtr = NULL ;
__constant__ float *d_wtr ;

static       int *h_shuffle_index = NULL ;
__constant__ int *d_shuffle_index ;
static       float *h_visible1 = NULL ;
__constant__ float *d_visible1 ;
static       float *h_visible2 = NULL ;
__constant__ float *d_visible2 ;
static       float *h_hidden1 = NULL ;
__constant__ float *d_hidden1 ;
static       float *h_hidden2 = NULL ;
__constant__ float *d_hidden2 ;
static       float *h_hidden_act = NULL ;
__constant__ float *d_hidden_act ;
static       float *h_in_bias_inc = NULL ;
__constant__ float *d_in_bias_inc ;
static       float *h_hid_bias_inc = NULL ;
__constant__ float *d_hid_bias_inc ;
static       float *h_hid_on_frac = NULL ;
__constant__ float *d_hid_on_frac ;
static       float *h_hid_on_smoothed = NULL ;
__constant__ float *d_hid_on_smoothed ;
static       float *h_w_inc = NULL ;
__constant__ float *d_w_inc ;
static       float *h_w_grad = NULL ;
__constant__ float *d_w_grad ;
static       float *h_prev_grad = NULL ;
__constant__ float *d_prev_grad ;
static       float *h_err_vec = NULL ;
__constant__ float *d_err_vec ;
static       float *h_len_out = NULL ;
__constant__ float *d_len_out ;
static       float *h_dot_out = NULL ;
__constant__ float *d_dot_out ;


static cudaDeviceProp deviceProp ;

// Function declarations

__global__ void device_recon_error ( int nc ) ;
__global__ void device_fetch_vis1 ( int istart , int random_offset ) ;
__global__ void device_vis_to_hid ( int nc ) ;
__global__ void device_hid_to_vis ( int nc , int random_offset ) ;
__global__ void device_hid_to_vis_direct ( int nc ) ;
__global__ void device_vis2_to_hid2 ( int nc ) ;
__global__ void device_sample_hidden2 ( int nc , int random_offset ) ;
__global__ void device_len_dot () ;
__global__ void device_max_inc ( int inc_vs_w ) ;
__global__ void device_update_in_bias ( int nc , float rate , float momentum ) ;
__global__ void device_update_hid_bias ( int nc , float rate , float momentum , int random_offset , float sparse_pen , float sparse_targ ) ;
__global__ void device_update_weights ( int nc , float rate , float momentum , float weight_pen , float sparse_pen , float sparse_targ ) ;
__global__ void device_transpose () ;


/*
--------------------------------------------------------------------------------

   RBM_CUDA_INIT - Initialize for CUDA RBM processing

   Fdata is used here to translate data from double (on the host) to float (on the device).
   It is freed here, immediately after use, in most routines, but then
   permanently allocated as a last step.

--------------------------------------------------------------------------------
*/


int rbm_cuda_init (
   int ncases ,            // Number of cases, needed for using shuffle_index for random sampling
   int ncols ,             // Number of columns in data (may exceed n_inputs)
   int n_inputs ,          // Number of inputs
   int nhid ,              // Number of hidden neurons
   int mean_field ,        // Use mean field instead of random sampling?
   int greedy_mean_field , // Use mean field for greedy training?
   int max_batch ,         // Max size of any batch
   double *data ,          // Input data, ncases rows by ncols columns
   double *data_mean ,     // Mean of each input, needed for weight sparsity penalty
   double *in_bias ,       // Input bias vector
   double *hid_bias ,      // Hidden bias vector
   double *w ,             // Weight matrix
   char *error_msg         // Returns text of error if problem
   )
{
   int i, j, n_inputs_cols, nhid_cols ;
   char msg[256] ;
   cudaError_t error_id ;

   MEMTEXT ( "RBM.cu: rbm_cuda_init starting" ) ;

   error_id = cudaSetDevice ( 0 ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init SetDevice failed %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      MEMTEXT ( error_msg ) ;
      audit ( error_msg ) ;
      cuda_enable = 0 ;
      return ERROR_CUDA_ERROR ;
      }

   cudaGetDeviceProperties ( &deviceProp , 0 ) ;

/*
   Extend the size of matrices to make sure every row starts on a 128-byte cache-line boundary
   This is not critical for the latest CUDA devices (although it does help a bit)
   but it makes a huge difference on older devices.
*/

   n_inputs_cols = (n_inputs + 31) / 32 * 32 ;
   nhid_cols = (nhid + 31) / 32 * 32 ;


/*
   Constants
*/

   cudaMemcpyToSymbol ( d_ncases , &ncases , sizeof(int) , 0 , cudaMemcpyHostToDevice ) ;
   cudaMemcpyToSymbol ( d_n_inputs , &n_inputs , sizeof(int) , 0 , cudaMemcpyHostToDevice ) ;
   cudaMemcpyToSymbol ( d_n_inputs_cols , &n_inputs_cols , sizeof(int) , 0 , cudaMemcpyHostToDevice ) ;
   cudaMemcpyToSymbol ( d_nhid , &nhid , sizeof(int) , 0 , cudaMemcpyHostToDevice ) ;
   cudaMemcpyToSymbol ( d_nhid_cols , &nhid_cols , sizeof(int) , 0 , cudaMemcpyHostToDevice ) ;
   cudaMemcpyToSymbol ( d_mean_field , &mean_field , sizeof(int) , 0 , cudaMemcpyHostToDevice ) ;
   cudaMemcpyToSymbol ( d_greedy_mean_field , &greedy_mean_field , sizeof(int) , 0 , cudaMemcpyHostToDevice ) ;


/*
   Data - We must extract only the (first) n_inputs columns from the ncols columns in data
*/

   fdata = (float *) MALLOC ( ncases * n_inputs * sizeof(float) ) ;
   if (fdata == NULL)
      return ERROR_INSUFFICIENT_MEMORY ;

   error_id = cudaMalloc ( (void **) &h_data , (size_t) (ncases * n_inputs * sizeof(float)) ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC data = %llu", (unsigned long long) h_data ) ;
   MEMTEXT ( msg ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc data (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }

   for (i=0 ; i<ncases ; i++) {
      for (j=0 ; j<n_inputs ; j++)
         fdata[i*n_inputs+j] = (float) data[i*ncols+j] ;
      }

   error_id = cudaMemcpy ( h_data , fdata , ncases * n_inputs * sizeof(float) , cudaMemcpyHostToDevice ) ;
   FREE ( fdata ) ;
   fdata = NULL ;

   if (error_id == cudaSuccess)
      error_id = cudaMemcpyToSymbol ( d_data , &h_data , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) ;

   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad data copy %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_ERROR ;
      }


/*
   Data mean
*/

   fdata = (float *) MALLOC ( n_inputs * sizeof(float) ) ;
   if (fdata == NULL)
      return ERROR_INSUFFICIENT_MEMORY ;

   error_id = cudaMalloc ( (void **) &h_data_mean , (size_t) (n_inputs * sizeof(float)) ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC data_mean = %llu", (unsigned long long) h_data_mean ) ;
   MEMTEXT ( msg ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc data_mean (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }

   for (i=0 ; i<n_inputs ; i++)
      fdata[i] = (float) data_mean[i] ;

   error_id = cudaMemcpy ( h_data_mean , fdata , n_inputs * sizeof(float) , cudaMemcpyHostToDevice ) ;
   FREE ( fdata ) ;
   fdata = NULL ;

   if (error_id == cudaSuccess)
      error_id = cudaMemcpyToSymbol ( d_data_mean , &h_data_mean , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) ;

   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad data_mean copy %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_ERROR ;
      }


/*
   Input bias
*/

   fdata = (float *) MALLOC ( n_inputs * sizeof(float) ) ;
   if (fdata == NULL)
      return ERROR_INSUFFICIENT_MEMORY ;

   error_id = cudaMalloc ( (void **) &h_in_bias , (size_t) (n_inputs * sizeof(float)) ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC in_bias = %llu", (unsigned long long) h_in_bias ) ;
   MEMTEXT ( msg ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc in_bias (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }

   for (i=0 ; i<n_inputs ; i++)
      fdata[i] = (float) in_bias[i] ;

   error_id = cudaMemcpy ( h_in_bias , fdata , n_inputs * sizeof(float) , cudaMemcpyHostToDevice ) ;
   FREE ( fdata ) ;
   fdata = NULL ;

   if (error_id == cudaSuccess)
      error_id = cudaMemcpyToSymbol ( d_in_bias , &h_in_bias , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) ;

   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad in_bias copy %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_ERROR ;
      }


/*
   Hidden bias
*/

   fdata = (float *) MALLOC ( nhid * sizeof(float) ) ;
   if (fdata == NULL)
      return ERROR_INSUFFICIENT_MEMORY ;

   error_id = cudaMalloc ( (void **) &h_hid_bias , (size_t) (nhid * sizeof(float)) ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC hid_bias = %llu", (unsigned long long) h_hid_bias ) ;
   MEMTEXT ( msg ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc hid_bias (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }

   for (i=0 ; i<nhid ; i++)
      fdata[i] = (float) hid_bias[i] ;

   error_id = cudaMemcpy ( h_hid_bias , fdata , nhid * sizeof(float) , cudaMemcpyHostToDevice ) ;
   FREE ( fdata ) ;
   fdata = NULL ;

   if (error_id == cudaSuccess)
      error_id = cudaMemcpyToSymbol ( d_hid_bias , &h_hid_bias , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) ;

   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad hid_bias copy %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_ERROR ;
      }


/*
   Weight array
*/

   fdata = (float *) MALLOC ( n_inputs_cols * nhid_cols * sizeof(float) ) ;
   if (fdata == NULL)
      return ERROR_INSUFFICIENT_MEMORY ;

   error_id = cudaMalloc ( (void **) &h_w , (size_t) (n_inputs_cols * nhid * sizeof(float)) ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC w = %llu", (unsigned long long) h_w ) ;
   MEMTEXT ( msg ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc w (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }

   error_id = cudaMalloc ( (void **) &h_wtr , (size_t) (n_inputs * nhid_cols * sizeof(float)) ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC wtr = %llu", (unsigned long long) h_wtr ) ;
   MEMTEXT ( msg ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc wtr (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }

   for (j=0 ; j<nhid ; j++) {
      for (i=0 ; i<n_inputs ; i++)
         fdata[j*n_inputs_cols+i] = (float) w[j*n_inputs+i] ;
      for ( ; i<n_inputs_cols ; i++)
         fdata[j*n_inputs_cols+i] = 0.0f ;
      }

   error_id = cudaMemcpy ( h_w , fdata , n_inputs_cols * nhid * sizeof(float) , cudaMemcpyHostToDevice ) ;

   if (error_id == cudaSuccess) {
      for (i=0 ; i<n_inputs ; i++) {
         for (j=0 ; j<nhid ; j++)
            fdata[i*nhid_cols+j] = (float) w[j*n_inputs+i] ;  // Transpose
         for ( ; j<nhid_cols ; j++)
            fdata[i*nhid_cols+j] = 0.0f ;
         }
      error_id = cudaMemcpy ( h_wtr , fdata , n_inputs * nhid_cols * sizeof(float) , cudaMemcpyHostToDevice ) ;
      }
   
   FREE ( fdata ) ;
   fdata = NULL ;

   if (error_id == cudaSuccess) {
      error_id = cudaMemcpyToSymbol ( d_w , &h_w , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) ;
      error_id = cudaMemcpyToSymbol ( d_wtr , &h_wtr , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) ;
      }

   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad w copy %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_ERROR ;
      }


/*
   Vector work areas that are not initialized here
*/

   error_id = cudaMalloc ( (void **) &h_shuffle_index , (size_t) (ncases * sizeof(int)) ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC shuffle_index = %llu", (unsigned long long) h_shuffle_index ) ;
   MEMTEXT ( msg ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc shuffle_index (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }
   cudaMemcpyToSymbol ( d_shuffle_index , &h_shuffle_index , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) ;

   error_id = cudaMalloc ( (void **) &h_visible1 , (size_t) (max_batch * n_inputs_cols * sizeof(float)) ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC visible1 = %llu", (unsigned long long) h_visible1 ) ;
   MEMTEXT ( msg ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc visible1 (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }
   cudaMemcpyToSymbol ( d_visible1 , &h_visible1 , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) ;

   error_id = cudaMalloc ( (void **) &h_visible2 , (size_t) (max_batch * n_inputs_cols * sizeof(float)) ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC visible2 = %llu", (unsigned long long) h_visible2 ) ;
   MEMTEXT ( msg ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc visible2 (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }
   cudaMemcpyToSymbol ( d_visible2 , &h_visible2 , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) ;

   error_id = cudaMalloc ( (void **) &h_hidden1 , (size_t) (max_batch * nhid_cols * sizeof(float)) ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC hidden1 = %llu", (unsigned long long) h_hidden1 ) ;
   MEMTEXT ( msg ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc hidden1 (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }
   cudaMemcpyToSymbol ( d_hidden1 , &h_hidden1 , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) ;

   error_id = cudaMalloc ( (void **) &h_hidden2 , (size_t) (max_batch * nhid_cols * sizeof(float)) ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC hidden2 = %llu", (unsigned long long) h_hidden2 ) ;
   MEMTEXT ( msg ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc hidden2 (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }
   cudaMemcpyToSymbol ( d_hidden2 , &h_hidden2 , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) ;

   error_id = cudaMalloc ( (void **) &h_hidden_act , (size_t) (max_batch * nhid_cols * sizeof(float)) ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC hidden_act = %llu", (unsigned long long) h_hidden_act ) ;
   MEMTEXT ( msg ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc hidden_act (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }
   cudaMemcpyToSymbol ( d_hidden_act , &h_hidden_act , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) ;

   error_id = cudaMalloc ( (void **) &h_hid_on_frac , (size_t) (max_batch * nhid_cols * sizeof(float)) ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC hid_on_frac = %llu", (unsigned long long) h_hid_on_frac ) ;
   MEMTEXT ( msg ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc hid_on_frac (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }
   cudaMemcpyToSymbol ( d_hid_on_frac , &h_hid_on_frac , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) ;

   error_id = cudaMalloc ( (void **) &h_in_bias_inc , (size_t) (n_inputs * sizeof(float)) ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC in_bias_inc = %llu", (unsigned long long) h_in_bias_inc ) ;
   MEMTEXT ( msg ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc in_bias_inc (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }
   cudaMemcpyToSymbol ( d_in_bias_inc , &h_in_bias_inc , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) ;

   error_id = cudaMalloc ( (void **) &h_hid_bias_inc , (size_t) (nhid * sizeof(float)) ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC hid_bias_inc = %llu", (unsigned long long) h_hid_bias_inc ) ;
   MEMTEXT ( msg ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc hid_bias_inc (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }
   cudaMemcpyToSymbol ( d_hid_bias_inc , &h_hid_bias_inc , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) ;

   error_id = cudaMalloc ( (void **) &h_hid_on_smoothed , (size_t) (nhid * sizeof(float)) ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC hid_on_smoothed = %llu", (unsigned long long) h_hid_on_smoothed ) ;
   MEMTEXT ( msg ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc hid_on_smoothed (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }
   cudaMemcpyToSymbol ( d_hid_on_smoothed , &h_hid_on_smoothed , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) ;

   error_id = cudaMalloc ( (void **) &h_w_inc , (size_t) (n_inputs_cols * nhid * sizeof(float)) ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC w_inc = %llu", (unsigned long long) h_w_inc ) ;
   MEMTEXT ( msg ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc w_inc (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }
   cudaMemcpyToSymbol ( d_w_inc , &h_w_inc , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) ;

   error_id = cudaMalloc ( (void **) &h_w_grad , (size_t) (n_inputs_cols * nhid * sizeof(float)) ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC w_grad = %llu", (unsigned long long) h_w_grad ) ;
   MEMTEXT ( msg ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc w_grad (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }
   cudaMemcpyToSymbol ( d_w_grad , &h_w_grad , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) ;

   error_id = cudaMalloc ( (void **) &h_prev_grad , (size_t) (n_inputs_cols * nhid * sizeof(float)) ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC prev_grad = %llu", (unsigned long long) h_prev_grad ) ;
   MEMTEXT ( msg ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc prev_grad (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }
   cudaMemcpyToSymbol ( d_prev_grad , &h_prev_grad , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) ;

   error_id = cudaMalloc ( (void **) &h_err_vec , (size_t) (n_inputs * sizeof(float)) ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC err_vec = %llu", (unsigned long long) h_err_vec ) ;
   MEMTEXT ( msg ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc err_vec (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }
   cudaMemcpyToSymbol ( d_err_vec , &h_err_vec , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) ;

   error_id = cudaMalloc ( (void **) &h_len_out , (size_t) (REDUC_BLOCKS * sizeof(float)) ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC len_out = %llu", (unsigned long long) h_len_out ) ;
   MEMTEXT ( msg ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc len_out (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }
   cudaMemcpyToSymbol ( d_len_out , &h_len_out , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) ;

   error_id = cudaMalloc ( (void **) &h_dot_out , (size_t) (REDUC_BLOCKS * sizeof(float)) ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC dot_out = %llu", (unsigned long long) h_dot_out ) ;
   MEMTEXT ( msg ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc dot_out (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }
   cudaMemcpyToSymbol ( d_dot_out , &h_dot_out , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) ;

   MEMTEXT ( "CUDA init reduc_fdata" ) ;
   reduc_fdata = (float *) MALLOC ( REDUC_BLOCKS * sizeof(float) ) ;
   if (reduc_fdata == NULL) {
      sprintf_s ( error_msg , 255 , "CUDA init bad MALLOC reduc_fdata" ) ;
      return ERROR_CUDA_MEMORY ;  // New error return
      }

/*
   Initialize things to starting values
*/

   fdata = (float *) MALLOC ( n_inputs_cols * nhid_cols * sizeof(float) ) ;
   if (fdata == NULL)
      return ERROR_INSUFFICIENT_MEMORY ;

   for (i=0 ; i<n_inputs_cols * nhid_cols ; i++)
      fdata[i] = 0.0f ;

   error_id = cudaMemcpy ( h_in_bias_inc , fdata , n_inputs * sizeof(float) , cudaMemcpyHostToDevice ) ;
   if (error_id  ==  cudaSuccess)
      error_id = cudaMemcpy ( h_hid_bias_inc , fdata , nhid * sizeof(float) , cudaMemcpyHostToDevice ) ;
   if (error_id  ==  cudaSuccess)
      error_id = cudaMemcpy ( h_w_inc , fdata , n_inputs_cols * nhid * sizeof(float) , cudaMemcpyHostToDevice ) ;
   if (error_id  ==  cudaSuccess)
      error_id = cudaMemcpy ( h_w_grad , fdata , n_inputs_cols * nhid * sizeof(float) , cudaMemcpyHostToDevice ) ;
   if (error_id  ==  cudaSuccess)
      error_id = cudaMemcpy ( h_prev_grad , fdata , n_inputs_cols * nhid * sizeof(float) , cudaMemcpyHostToDevice ) ;

   if (error_id  ==  cudaSuccess) {
      for (i=0 ; i<nhid ; i++)
         fdata[i] = (float) 0.5 ;
      error_id = cudaMemcpy ( h_hid_on_smoothed , fdata , nhid * sizeof(float) , cudaMemcpyHostToDevice ) ;
      }

   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad final inits (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_ERROR ;
      }

   i = max_batch * n_inputs_cols ;
   if (max_batch * nhid_cols > i)
      i = max_batch * nhid_cols ;
   if (n_inputs_cols * nhid_cols > i)
      i = n_inputs_cols * nhid_cols ;

   fdata = (float *) REALLOC ( fdata , i * sizeof(float) ) ; // Used for passing parameters back to host
   if (fdata == NULL)
      return ERROR_INSUFFICIENT_MEMORY ;


/*
   Set cache/shared memory preferences
*/

   error_id = cudaFuncSetCacheConfig ( device_recon_error , cudaFuncCachePreferL1 ) ;
   if (error_id == cudaSuccess)
      error_id = cudaFuncSetCacheConfig ( device_fetch_vis1 , cudaFuncCachePreferL1 ) ;
   if (error_id == cudaSuccess)
      error_id = cudaFuncSetCacheConfig ( device_vis_to_hid , cudaFuncCachePreferL1 ) ;
   if (error_id == cudaSuccess)
      error_id = cudaFuncSetCacheConfig ( device_hid_to_vis , cudaFuncCachePreferL1 ) ;
   if (error_id == cudaSuccess)
      error_id = cudaFuncSetCacheConfig ( device_hid_to_vis_direct , cudaFuncCachePreferL1 ) ;
   if (error_id == cudaSuccess)
      error_id = cudaFuncSetCacheConfig ( device_vis2_to_hid2 , cudaFuncCachePreferL1 ) ;
   if (error_id == cudaSuccess)
      error_id = cudaFuncSetCacheConfig ( device_sample_hidden2 , cudaFuncCachePreferL1 ) ;
   if (error_id == cudaSuccess)
      error_id = cudaFuncSetCacheConfig ( device_len_dot , cudaFuncCachePreferNone ) ;
   if (error_id == cudaSuccess)
      error_id = cudaFuncSetCacheConfig ( device_max_inc , cudaFuncCachePreferNone ) ;
   if (error_id == cudaSuccess)
      error_id = cudaFuncSetCacheConfig ( device_update_in_bias , cudaFuncCachePreferL1 ) ;
   if (error_id == cudaSuccess)
      error_id = cudaFuncSetCacheConfig ( device_update_hid_bias , cudaFuncCachePreferL1 ) ;
   if (error_id == cudaSuccess)
      error_id = cudaFuncSetCacheConfig ( device_update_weights , cudaFuncCachePreferL1 ) ;
   if (error_id == cudaSuccess)
      error_id = cudaFuncSetCacheConfig ( device_transpose , cudaFuncCachePreferL1 ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaFuncSetCacheConfig" ) ;
      return ERROR_CUDA_ERROR ;
      }

   MEMTEXT ( "RBM.cu: rbm_cuda_init finished" ) ;
   return 0 ;
}


/*
--------------------------------------------------------------------------------

   shuffle_to_device - Copy the shuffle vector to the device

--------------------------------------------------------------------------------
*/


int cuda_shuffle_to_device (
   int ncases ,
   int *shuffle_index
   )
{
   char msg[256] ;
   cudaError_t error_id ;

   error_id = cudaMemcpy ( h_shuffle_index , shuffle_index , ncases * sizeof(int) , cudaMemcpyHostToDevice ) ;

   if (error_id  !=  cudaSuccess) {
      sprintf_s ( msg , 255 , "CUDA bad shuffle_to_device %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      return ERROR_CUDA_ERROR ;
      }
   return 0 ;
}


/*
--------------------------------------------------------------------------------

   params_to_device - Copy the weights and biases to the device
                      This is called only by rbm_cuda_wt_init(),
                      not by rbm_thr2().

--------------------------------------------------------------------------------
*/


int cuda_params_to_device (
   int n_inputs ,
   int nhid ,
   double *in_bias ,
   double *hid_bias ,
   double *w
   )
{
   int i, j, n_inputs_cols, nhid_cols ;
   char msg[256] ;
   cudaError_t error_id ;

   n_inputs_cols = (n_inputs + 31) / 32 * 32 ;
   nhid_cols = (nhid + 31) / 32 * 32 ;

   for (i=0 ; i<n_inputs ; i++)
      fdata[i] = (float) in_bias[i] ;
   error_id = cudaMemcpy ( h_in_bias , fdata , n_inputs * sizeof(float) , cudaMemcpyHostToDevice ) ;

   if (error_id  ==  cudaSuccess) {
      for (i=0 ; i<nhid ; i++)
         fdata[i] = (float) hid_bias[i] ;
      error_id = cudaMemcpy ( h_hid_bias , fdata , nhid * sizeof(float) , cudaMemcpyHostToDevice ) ;
      }

   if (error_id  ==  cudaSuccess) {
      for (j=0 ; j<nhid ; j++) {
         for (i=0 ; i<n_inputs ; i++)
            fdata[j*n_inputs_cols+i] = (float) w[j*n_inputs+i] ;
         }
      error_id = cudaMemcpy ( h_w , fdata , n_inputs_cols * nhid * sizeof(float) , cudaMemcpyHostToDevice ) ;
      }

   if (error_id == cudaSuccess) {
      for (i=0 ; i<n_inputs ; i++) {
         for (j=0 ; j<nhid ; j++)
            fdata[i*nhid_cols+j] = (float) w[j*n_inputs+i] ;  // Transpose
         }
      error_id = cudaMemcpy ( h_wtr , fdata , n_inputs * nhid_cols * sizeof(float) , cudaMemcpyHostToDevice ) ;
      }

   if (error_id  !=  cudaSuccess) {
      sprintf_s ( msg , 255 , "CUDA bad params_to_device %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      return ERROR_CUDA_ERROR ;
      }

   return 0 ;
}


/*
------------------------------------------------------------------------------------------------

  cuda_params_from_device

------------------------------------------------------------------------------------------------
*/

int cuda_params_from_device (
   int n_inputs ,
   int nhid ,
   double *in_bias ,
   double *hid_bias ,
   double *w
   )
{
   int ivis, ihid, n_inputs_cols ;
   char msg[256] ;
   cudaError_t error_id ;

   n_inputs_cols = (n_inputs + 31) / 32 * 32 ;

   error_id = cudaMemcpy ( fdata , h_w , nhid * n_inputs_cols * sizeof(float) , cudaMemcpyDeviceToHost ) ;
   for (ihid=0 ; ihid<nhid ; ihid++) {
      for (ivis=0 ; ivis<n_inputs ; ivis++)
         w[ihid*n_inputs+ivis] = fdata[ihid*n_inputs_cols+ivis] ;
      }

   if (error_id == cudaSuccess) {
      error_id = cudaMemcpy ( fdata , h_in_bias , n_inputs * sizeof(float) , cudaMemcpyDeviceToHost ) ;
      for (ivis=0 ; ivis<n_inputs ; ivis++)
         in_bias[ivis] = fdata[ivis] ;
      }

   if (error_id == cudaSuccess) {
      error_id = cudaMemcpy ( fdata , h_hid_bias , nhid * sizeof(float) , cudaMemcpyDeviceToHost ) ;
      for (ihid=0 ; ihid<nhid ; ihid++)
         hid_bias[ihid] = fdata[ihid] ;
      }

   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_params_from_device Memcpy error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      return 1 ;
      }

   return 0 ;
}


/*
------------------------------------------------------------------------------------------------

   cuda_recon_error - Compute reconstruction error

------------------------------------------------------------------------------------------------
*/

__global__ void device_recon_error (
   int nc       // Number of cases in this batch
   )
{
   int icase, ivis ;
   float errsum ;

   ivis = blockIdx.x * blockDim.x + threadIdx.x ;

   if (ivis >= d_n_inputs)
      return ;

   errsum = 0.0f ;

#if RECON_ERR_XENT
   for (icase=0 ; icase<nc ; icase++) {
      errsum -= d_visible1[icase*d_n_inputs_cols+ivis] * __logf(d_visible2[icase*d_n_inputs_cols+ivis]+0.0000000001f) +
                (1.0f - d_visible1[icase*d_n_inputs_cols+ivis]) * __logf(1.0f-d_visible2[icase*d_n_inputs_cols+ivis]+0.0000000001f) ;
      }
#else
   float diff ;
   for (icase=0 ; icase<nc ; icase++) {
      diff = d_visible1[icase*d_n_inputs_cols+ivis] - d_visible2[icase*d_n_inputs_cols+ivis] ;
      errsum += diff * diff ;
      }
#endif

   d_err_vec[ivis] = errsum ;
}


int cuda_recon_error (
   int n_inputs ,          // Number of inputs
   int nc ,                // Number of cases in this batch
   double *err_vec         // Cumulates MSE for each input; n_inputs long
   )
{
   int i, warpsize, blocks_per_grid, threads_per_block ;
   char msg[256] ;
   cudaError_t error_id ;

   warpsize = deviceProp.warpSize ;      // Threads per warp, likely 32 well into the future

   threads_per_block = (n_inputs + warpsize - 1) / warpsize * warpsize ;
   if (threads_per_block > 4 * warpsize)
      threads_per_block = 4 * warpsize ;
   blocks_per_grid = (n_inputs + threads_per_block - 1) / threads_per_block ;

   device_recon_error <<< blocks_per_grid , threads_per_block >>> ( nc ) ;   
   cudaThreadSynchronize() ;
   error_id = cudaGetLastError () ;
   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_recon_error launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      return 1 ;
      }

   error_id = cudaMemcpy ( fdata , h_err_vec , n_inputs * sizeof(float) , cudaMemcpyDeviceToHost ) ;
   for (i=0 ; i<n_inputs ; i++)
      err_vec[i] = fdata[i] ;

   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_recon_error Memcpy error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      return 1 ;
      }

   return 0 ;
}


/*
------------------------------------------------------------------------------------------------

   cuda_fetch_vis1 saves in visible1 the actual input, shuffled and batch selected.

   If greedy_mean_field is false it then samples.

------------------------------------------------------------------------------------------------
*/

__global__ void device_fetch_vis1 (
   int istart ,        // First case in this batch
   int random_offset   // Starting index in shuffle_index for random sampling
   )
{
   int k, icase, ivis ;
   float frand ;

   ivis = blockIdx.x * blockDim.x + threadIdx.x ;
   if (ivis >= d_n_inputs)
      return ;

   icase = blockIdx.y ;

   d_visible1[icase*d_n_inputs_cols+ivis] = d_data[d_shuffle_index[istart+icase]*d_n_inputs+ivis] ;

   if (! d_greedy_mean_field) {
      k = ((unsigned int) (icase * d_n_inputs + ivis + random_offset)) % d_ncases ;
      frand = (float) d_shuffle_index[k] / (float) d_ncases ;
      d_visible1[icase*d_n_inputs_cols+ivis] = (frand < d_visible1[icase*d_n_inputs_cols+ivis])  ?  1.0f : 0.0f ;
      }
}

int cuda_fetch_vis1 (
   int istart ,           // First case in this batch
   int istop ,            // One past last case
   int n_inputs ,         // Number of inputs
   int random_offset ,    // Starting index in shuffle_index for random sampling
   double *visible1       // If non-NULL, return n_inputs * (istop-istart) long
   )
{
   int icase, ivis, warpsize, threads_per_block, n_inputs_cols ;
   char msg[256] ;
   dim3 block_launch ;
   cudaError_t error_id ;

   warpsize = deviceProp.warpSize ;      // Threads per warp, likely 32 well into the future

   threads_per_block = (n_inputs + warpsize - 1) / warpsize * warpsize ;
   if (threads_per_block > 4 * warpsize)
      threads_per_block = 4 * warpsize ;
   block_launch.x = (n_inputs + threads_per_block - 1) / threads_per_block ;
   block_launch.y = istop - istart ;
   block_launch.z = 1 ;

   device_fetch_vis1 <<< block_launch , threads_per_block >>> ( istart , random_offset ) ;   
   cudaThreadSynchronize() ;
   error_id = cudaGetLastError () ;
   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_fetch_vis1 launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      return 1 ;
      }

   if (visible1 != NULL) {
      n_inputs_cols = (n_inputs + 31) / 32 * 32 ;
      error_id = cudaMemcpy ( fdata , h_visible1 , (istop - istart) * n_inputs_cols * sizeof(float) , cudaMemcpyDeviceToHost ) ;
      for (icase=0 ; icase<istop-istart ; icase++) {
         for (ivis=0 ; ivis<n_inputs ; ivis++)
            visible1[icase*n_inputs+ivis] = fdata[icase*n_inputs_cols+ivis] ;
         }
      if (error_id != cudaSuccess) {
         sprintf_s ( msg , 255 , "cuda_fetch_vis1 Memcpy error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
         audit ( msg ) ;
         return 1 ;
         }
      }

   return 0 ;
}


/*
------------------------------------------------------------------------------------------------

   cuda_vis_to_hid uses visible1 to compute hidden1 probabilities
                   Also copies to hidden2 for later use in MC chain loop

------------------------------------------------------------------------------------------------
*/

__global__ void device_vis_to_hid (
   int nc                 // Number of cases in this batch
   )
{
   int icase, ivis, ihid ;
   float sum, Q ;

   ihid = blockIdx.x * blockDim.x + threadIdx.x ;
   if (ihid >= d_nhid)
      return ;

   icase = blockIdx.y ;

   sum = d_hid_bias[ihid] ;
   for (ivis=0 ; ivis<d_n_inputs ; ivis++)
      sum += d_wtr[ivis*d_nhid_cols+ihid] * d_visible1[icase*d_n_inputs_cols+ivis] ;
   Q = 1.0f / (1.0f + __expf(-sum)) ;
   d_hidden1[icase*d_nhid_cols+ihid] = Q ;
   d_hidden2[icase*d_nhid_cols+ihid] = Q ;     // We'll need this for MC chain loop
   d_hid_on_frac[icase*d_nhid_cols+ihid] = Q ;
}

int cuda_vis_to_hid (
   int nc ,                // Number of cases in this batch
   int nhid ,              // Number of hidden neurons
   double *hidden1 ,       // Work vector nhid * (istop-istart) long
   double *hidden_act ,    // Work vector nhid * (istop-istart) long
   double *hid_on_frac     // Work vector nhid * (istop-istart) long
   )
{
   int icase, ihid, warpsize, threads_per_block, nhid_cols ;
   char msg[256] ;
   dim3 block_launch ;
   cudaError_t error_id ;

   warpsize = deviceProp.warpSize ;      // Threads per warp, likely 32 well into the future

   threads_per_block = (nhid + warpsize - 1) / warpsize * warpsize ;
   if (threads_per_block > 4 * warpsize)
      threads_per_block = 4 * warpsize ;

   block_launch.x = (nhid + threads_per_block - 1) / threads_per_block ;
   block_launch.y = nc ;
   block_launch.z = 1 ;

   device_vis_to_hid <<< block_launch , threads_per_block >>> ( nc ) ;   
   cudaThreadSynchronize() ;
   error_id = cudaGetLastError () ;
   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_vis_to_hid launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      return 1 ;
      }

   if (hidden1 != NULL) {
      nhid_cols = (nhid + 31) / 32 * 32 ;
      error_id = cudaMemcpy ( fdata , h_hidden1 , nc * nhid_cols * sizeof(float) , cudaMemcpyDeviceToHost ) ;
      for (icase=0 ; icase<nc ; icase++) {
         for (ihid=0 ; ihid<nhid ; ihid++)
            hidden1[icase*nhid+ihid] = fdata[icase*nhid_cols+ihid] ;
         }
      if (error_id == cudaSuccess) {
         error_id = cudaMemcpy ( fdata , h_hidden_act , nc * nhid_cols * sizeof(float) , cudaMemcpyDeviceToHost ) ;
         for (icase=0 ; icase<nc ; icase++) {
            for (ihid=0 ; ihid<nhid ; ihid++)
               hidden_act[icase*nhid+ihid] = fdata[icase*nhid_cols+ihid] ;
            }
         }
      if (error_id == cudaSuccess) {
         error_id = cudaMemcpy ( fdata , h_hid_on_frac , nc * nhid_cols * sizeof(float) , cudaMemcpyDeviceToHost ) ;
         for (icase=0 ; icase<nc ; icase++) {
            for (ihid=0 ; ihid<nhid ; ihid++)
               hid_on_frac[icase*nhid+ihid] = fdata[icase*nhid_cols+ihid] ;
            }
         }
      if (error_id != cudaSuccess) {
         sprintf_s ( msg , 255 , "cuda_vis_to_hid Memcpy error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
         audit ( msg ) ;
         return 1 ;
         }
      }

   return 0 ;
}


/*
------------------------------------------------------------------------------------------------

   cuda_hid_to_vis uses hidden1 to compute and optionally sample visible2

   The 'direct' version does not sample for hidden.  It's for reproduction error
   for finding initial weights.

------------------------------------------------------------------------------------------------
*/

__global__ void device_hid_to_vis (
   int nc ,                // Number of cases in this batch
   int random_offset       // Starting index in shuffle_index for random sampling
   )
{
   int k, icase, ivis, ihid ;
   float sum, P, frand ;

   ivis = blockIdx.x * blockDim.x + threadIdx.x ;
   if (ivis >= d_n_inputs)
      return ;

   icase = blockIdx.y ;

   sum = d_in_bias[ivis] ;
   for (ihid=0 ; ihid<d_nhid ; ihid++)
      sum += d_w[ihid*d_n_inputs_cols+ivis] * d_hidden_act[icase*d_nhid_cols+ihid] ;
   P = 1.0f / (1.0f + __expf(-sum)) ;

   if (d_mean_field)
      d_visible2[icase*d_n_inputs_cols+ivis] = P ;
   else {
      k = ((unsigned int) (icase * d_n_inputs + ivis + random_offset)) % d_ncases ;
      frand = (float) d_shuffle_index[k] / (float) d_ncases ;
      d_visible2[icase*d_n_inputs_cols+ivis] = (frand < P)  ?  1.0f : 0.0f ;
      }

}

int cuda_hid_to_vis (
   int nc ,                // Number of cases in this batch
   int n_inputs ,          // Number of inputs
   int random_offset ,     // Starting index in shuffle_index for random sampling
   double *visible2        // Work vector n_inputs * nc long
   )
{
   int icase, ivis, warpsize, threads_per_block, n_inputs_cols ;
   char msg[256] ;
   dim3 block_launch ;
   cudaError_t error_id ;

   warpsize = deviceProp.warpSize ;      // Threads per warp, likely 32 well into the future

   threads_per_block = (n_inputs + warpsize - 1) / warpsize * warpsize ;
   if (threads_per_block > 4 * warpsize)
      threads_per_block = 4 * warpsize ;

   block_launch.x = (n_inputs + threads_per_block - 1) / threads_per_block ;
   block_launch.y = nc ;
   block_launch.z = 1 ;

   device_hid_to_vis <<< block_launch , threads_per_block >>> ( nc , random_offset ) ;   
   cudaThreadSynchronize() ;
   error_id = cudaGetLastError () ;
   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_hid_to_vis launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      return 1 ;
      }

   if (visible2 != NULL) {
      n_inputs_cols = (n_inputs + 31) / 32 * 32 ;
      error_id = cudaMemcpy ( fdata , h_visible2 , nc * n_inputs_cols * sizeof(float) , cudaMemcpyDeviceToHost ) ;
      for (icase=0 ; icase<nc ; icase++) {
         for (ivis=0 ; ivis<n_inputs ; ivis++)
            visible2[icase*n_inputs+ivis] = fdata[icase*n_inputs_cols+ivis] ;
         }
      if (error_id != cudaSuccess) {
         sprintf_s ( msg , 255 , "cuda_hid_to_vis Memcpy error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
         audit ( msg ) ;
         return 1 ;
         }
      }

   return 0 ;
}

__global__ void device_hid_to_vis_direct (
   int nc                 // Number of cases in this batch
   )
{
   int icase, ivis, ihid ;
   float sum ;

   ivis = blockIdx.x * blockDim.x + threadIdx.x ;
   if (ivis >= d_n_inputs)
      return ;

   icase = blockIdx.y ;

   sum = d_in_bias[ivis] ;
   for (ihid=0 ; ihid<d_nhid ; ihid++)
      sum += d_w[ihid*d_n_inputs_cols+ivis] * d_hidden1[icase*d_nhid_cols+ihid] ;
   d_visible2[icase*d_n_inputs_cols+ivis] = 1.0f / (1.0f + __expf(-sum)) ;

}

int cuda_hid_to_vis_direct (
   int nc ,                // Number of cases in this batch
   int n_inputs            // Number of inputs
   )
{
   int warpsize, threads_per_block ;
   char msg[256] ;
   dim3 block_launch ;
   cudaError_t error_id ;

   warpsize = deviceProp.warpSize ;      // Threads per warp, likely 32 well into the future

   threads_per_block = (n_inputs + warpsize - 1) / warpsize * warpsize ;
   if (threads_per_block > 4 * warpsize)
      threads_per_block = 4 * warpsize ;

   block_launch.x = (n_inputs + threads_per_block - 1) / threads_per_block ;
   block_launch.y = nc ;
   block_launch.z = 1 ;

   device_hid_to_vis_direct <<< block_launch , threads_per_block >>> ( nc ) ;   
   cudaThreadSynchronize() ;
   error_id = cudaGetLastError () ;
   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_hid_to_vis launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      return 1 ;
      }

   return 0 ;
}


/*
------------------------------------------------------------------------------------------------

   cuda_vis2_to_hid2 uses visible2 to compute hidden2

------------------------------------------------------------------------------------------------
*/

__global__ void device_vis2_to_hid2 (
   int nc         // Number of cases in this batch
   )
{
   int icase, ivis, ihid ;
   float sum ;

   ihid = blockIdx.x * blockDim.x + threadIdx.x ;
   if (ihid >= d_nhid)
      return ;

   icase = blockIdx.y ;

   sum = d_hid_bias[ihid] ;
   for (ivis=0 ; ivis<d_n_inputs ; ivis++)
      sum += d_wtr[ivis*d_nhid_cols+ihid] * d_visible2[icase*d_n_inputs_cols+ivis] ;
   d_hidden2[icase*d_nhid_cols+ihid] = 1.0f / (1.0f + __expf(-sum)) ;
}

int cuda_vis2_to_hid2 (
   int nc ,                // Number of cases in this batch
   int nhid ,              // Number of hidden neurons
   double *hidden2         // Work vector nhid * (istop-istart) long
   )
{
   int icase, ihid, warpsize, threads_per_block, nhid_cols ;
   char msg[256] ;
   dim3 block_launch ;
   cudaError_t error_id ;

   warpsize = deviceProp.warpSize ;      // Threads per warp, likely 32 well into the future

   threads_per_block = (nhid + warpsize - 1) / warpsize * warpsize ;
   block_launch.x = (nhid + threads_per_block - 1) / threads_per_block ;
   block_launch.y = nc ;
   block_launch.z = 1 ;

   device_vis2_to_hid2 <<< block_launch , threads_per_block >>> ( nc ) ;   
   cudaThreadSynchronize() ;
   error_id = cudaGetLastError () ;
   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_vis_to_hid launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      return 1 ;
      }

   if (hidden2 != NULL) {
      nhid_cols = (nhid + 31) / 32 * 32 ;
      error_id = cudaMemcpy ( fdata , h_hidden2 , nc * nhid_cols * sizeof(float) , cudaMemcpyDeviceToHost ) ;
      for (icase=0 ; icase<nc ; icase++) {
         for (ihid=0 ; ihid<nhid ; ihid++)
            hidden2[icase*nhid+ihid] = fdata[icase*nhid_cols+ihid] ;
         }
      if (error_id != cudaSuccess) {
         sprintf_s ( msg , 255 , "cuda_vis2_to_hid2 Memcpy error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
         audit ( msg ) ;
         return 1 ;
         }
      }

   return 0 ;
}


/*
------------------------------------------------------------------------------------------------

   cuda_sample_hidden2 samples hidden2 into hidden_act

------------------------------------------------------------------------------------------------
*/

__global__ void device_sample_hidden2 (
   int nc ,                // Number of cases in this batch
   int random_offset       // Starting index in shuffle_index for random sampling
   )
{
   int k, icase, ihid ;
   float frand ;

   ihid = blockIdx.x * blockDim.x + threadIdx.x ;
   if (ihid >= d_nhid)
      return ;

   icase = blockIdx.y ;

   k = ((unsigned int) (icase * d_nhid + ihid + random_offset)) % d_ncases ;
   frand = (float) d_shuffle_index[k] / (float) d_ncases ;

   d_hidden_act[icase*d_nhid_cols+ihid] = (frand < d_hidden2[icase*d_nhid_cols+ihid])  ?  1.0f : 0.0f ;
}

int cuda_sample_hidden2 (
   int nc ,                // Number of cases in this batch
   int nhid ,              // Number of hidden neurons
   int random_offset ,     // Starting index in shuffle_index for random sampling
   double *hidden_act      // Work vector nhid * (istop-istart) long
   )
{
   int icase, ihid, warpsize, threads_per_block, nhid_cols ;
   char msg[256] ;
   dim3 block_launch ;
   cudaError_t error_id ;

   warpsize = deviceProp.warpSize ;      // Threads per warp, likely 32 well into the future

   threads_per_block = (nhid + warpsize - 1) / warpsize * warpsize ;
   block_launch.x = (nhid + threads_per_block - 1) / threads_per_block ;
   block_launch.y = nc ;
   block_launch.z = 1 ;

   device_sample_hidden2 <<< block_launch , threads_per_block >>> ( nc , random_offset ) ;   
   cudaThreadSynchronize() ;
   error_id = cudaGetLastError () ;
   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_sample_hidden2 launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      return 1 ;
      }

   if (hidden_act != NULL) {
      nhid_cols = (nhid + 31) / 32 * 32 ;
      error_id = cudaMemcpy ( fdata , h_hidden_act , nc * nhid_cols * sizeof(float) , cudaMemcpyDeviceToHost ) ;
      for (icase=0 ; icase<nc ; icase++) {
         for (ihid=0 ; ihid<nhid ; ihid++)
            hidden_act[icase*nhid+ihid] = fdata[icase*nhid_cols+ihid] ;
         }
      if (error_id != cudaSuccess) {
         sprintf_s ( msg , 255 , "cuda_sample_hidden2 Memcpy error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
         audit ( msg ) ;
         return 1 ;
         }
      }

   return 0 ;
}


/*
------------------------------------------------------------------------------------------------

   cuda_len_dot

   WARNING - This requires that the unused elements at the end of each row be zero!

------------------------------------------------------------------------------------------------
*/

__global__ void device_len_dot ()
{
   __shared__ float partial_len[REDUC_THREADS], partial_dot[REDUC_THREADS] ;
   int i, n, index ;
   float sum_len, sum_dot ;

   index = threadIdx.x ;
   n = d_n_inputs_cols * d_nhid ;

   sum_len = sum_dot = 0.0f ;   
   for (i=blockIdx.x*blockDim.x+index ; i<n ; i+=blockDim.x*gridDim.x) {
      sum_len += d_w_grad[i] * d_w_grad[i] ;
      sum_dot += d_w_grad[i] * d_prev_grad[i] ;
      d_prev_grad[i] = d_w_grad[i] ;
      }

   partial_len[index] = sum_len ;
   partial_dot[index] = sum_dot ;
   __syncthreads() ;

   for (i=blockDim.x>>1 ; i ; i>>=1) {
      if (index < i) {
         partial_len[index] += partial_len[index+i] ;
         partial_dot[index] += partial_dot[index+i] ;
         }
      __syncthreads() ;
      }

   if (index == 0) {
      d_len_out[blockIdx.x] = partial_len[0] ;
      d_dot_out[blockIdx.x] = partial_dot[0] ;
      }
}


int cuda_len_dot (
   int n ,           // Number of weights; Not important; just heuristically sets # blocks
   double *len,      // Computed squared length
   double *dot       // Computed dot product
   )
{
   int i, blocks_per_grid ;
   double sum ;
   char msg[256] ;
   cudaError_t error_id ;

   blocks_per_grid = (n + REDUC_THREADS - 1) / REDUC_THREADS ;
   if (blocks_per_grid > REDUC_BLOCKS)
      blocks_per_grid = REDUC_BLOCKS ;

   device_len_dot <<< blocks_per_grid , REDUC_THREADS >>> () ;   
   cudaThreadSynchronize() ;

   error_id = cudaGetLastError () ;
   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_len_dot launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      return 1 ;
      }

   error_id = cudaMemcpy ( reduc_fdata , h_len_out , blocks_per_grid * sizeof(float) , cudaMemcpyDeviceToHost ) ;
   sum = 0.0 ;
   for (i=0 ; i<blocks_per_grid ; i++)
      sum += reduc_fdata[i] ;
   *len = sum ;

   if (error_id == cudaSuccess) {
      error_id = cudaMemcpy ( reduc_fdata , h_dot_out , blocks_per_grid * sizeof(float) , cudaMemcpyDeviceToHost ) ;
      sum = 0.0 ;
      for (i=0 ; i<blocks_per_grid ; i++)
         sum += reduc_fdata[i] ;
      *dot = sum ;
      }

   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_len_dot Memcpy error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      return 1 ;
      }

   return 0 ;
}


/*
------------------------------------------------------------------------------------------------

   cuda_max_inc_w - Compute max inc or max w

   This borrows ?_len_out for its block output

   WARNING - This requires that the unused elements at the end of each row be zero!

------------------------------------------------------------------------------------------------
*/

__global__ void device_max_inc ( int inc_vs_w )
{
   __shared__ float partial_max[REDUC_THREADS] ;
   int i, n, index ;
   float max_inc_w ;

   index = threadIdx.x ;
   n = d_n_inputs_cols * d_nhid ;

   max_inc_w = 0.0f ;   
   if (inc_vs_w) {
      for (i=blockIdx.x*blockDim.x+index ; i<n ; i+=blockDim.x*gridDim.x) {
         if (fabs(d_w_inc[i]) > max_inc_w)
            max_inc_w = fabs(d_w_inc[i]) ;
         }
      }
   else {
      for (i=blockIdx.x*blockDim.x+index ; i<n ; i+=blockDim.x*gridDim.x) {
         if (fabs(d_w[i]) > max_inc_w)
            max_inc_w = fabs(d_w[i]) ;
         }
      }

   partial_max[index] = max_inc_w ;
   __syncthreads() ;

   for (i=blockDim.x>>1 ; i ; i>>=1) {
      if (index < i) {
         if (partial_max[index+i] > partial_max[index])
            partial_max[index] = partial_max[index+i] ;
         }
      __syncthreads() ;
      }

   if (index == 0)
      d_len_out[blockIdx.x] = partial_max[0] ;
}


int cuda_max_inc_w (
   int n ,              // Number of weights; Not important; just heuristically sets # blocks
   double *max_inc_w ,  // Computed max absolute weight
   int inc_vs_w         // Which to compute
   )
{
   int i, blocks_per_grid ;
   char msg[256] ;
   cudaError_t error_id ;

   blocks_per_grid = (n + REDUC_THREADS - 1) / REDUC_THREADS ;
   if (blocks_per_grid > REDUC_BLOCKS)
      blocks_per_grid = REDUC_BLOCKS ;

   device_max_inc <<< blocks_per_grid , REDUC_THREADS >>> ( inc_vs_w ) ;   
   cudaThreadSynchronize() ;

   error_id = cudaGetLastError () ;
   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_max_inc_w launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      return 1 ;
      }

   error_id = cudaMemcpy ( reduc_fdata , h_len_out , blocks_per_grid * sizeof(float) , cudaMemcpyDeviceToHost ) ;
   *max_inc_w = 0.0 ;
   for (i=0 ; i<blocks_per_grid ; i++) {
      if (reduc_fdata[i] > *max_inc_w)
         *max_inc_w = reduc_fdata[i] ;
      }

   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_max_inc_w Memcpy error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      return 1 ;
      }

   return 0 ;
}


/*
------------------------------------------------------------------------------------------------

   cuda_update_in_bias

------------------------------------------------------------------------------------------------
*/

__global__ void device_update_in_bias (
   int nc ,               // Number of cases in this batch
   float rate ,           // Learning rate
   float momentum         // Learning momentum
   )
{
   int icase, ivis ;
   float sum ;

   ivis = blockIdx.x * blockDim.x + threadIdx.x ;

   if (ivis >= d_n_inputs)
      return ;

   sum = 0.0f ;

   for (icase=0 ; icase<nc ; icase++)
      sum += d_visible1[icase*d_n_inputs_cols+ivis] - d_visible2[icase*d_n_inputs_cols+ivis] ;

   d_in_bias_inc[ivis] = momentum * d_in_bias_inc[ivis] + rate * sum / nc ;
   d_in_bias[ivis] += d_in_bias_inc[ivis] ;
}


int cuda_update_in_bias (
   int nc ,                // Number of cases in this batch
   int n_inputs ,          // Number of inputs
   double rate ,           // Learning rate
   double momentum ,       // Learning momentum
   double *in_bias ,       // Input bias vector, n_inputs long
   double *in_bias_inc     // Input bias increment vector, carries over from batch to batch, n_inputs long
   )
{
   int i, warpsize, blocks_per_grid, threads_per_block ;
   char msg[256] ;
   cudaError_t error_id ;

   warpsize = deviceProp.warpSize ;      // Threads per warp, likely 32 well into the future

   threads_per_block = (n_inputs + warpsize - 1) / warpsize * warpsize ;
   if (threads_per_block > 4 * warpsize)
      threads_per_block = 4 * warpsize ;
   blocks_per_grid = (n_inputs + threads_per_block - 1) / threads_per_block ;

   device_update_in_bias <<< blocks_per_grid , threads_per_block >>> ( nc , (float) rate , (float) momentum ) ;   
   cudaThreadSynchronize() ;
   error_id = cudaGetLastError () ;
   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_update_in_bias launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      return 1 ;
      }

   if (in_bias != NULL  &&  error_id == cudaSuccess) {
      error_id = cudaMemcpy ( fdata , h_in_bias , n_inputs * sizeof(float) , cudaMemcpyDeviceToHost ) ;
      for (i=0 ; i<n_inputs ; i++)
         in_bias[i] = fdata[i] ;
      if (error_id == cudaSuccess) {
         error_id = cudaMemcpy ( fdata , h_in_bias_inc , n_inputs * sizeof(float) , cudaMemcpyDeviceToHost ) ;
         for (i=0 ; i<n_inputs ; i++)
            in_bias_inc[i] = fdata[i] ;
         }
      }

   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_update_in_bias Memcpy error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      return 1 ;
      }

   return 0 ;
}


/*
------------------------------------------------------------------------------------------------

   cuda_update_hid_bias

------------------------------------------------------------------------------------------------
*/

__global__ void device_update_hid_bias (
   int nc ,               // Number of cases in this batch
   float rate ,           // Learning rate
   float momentum ,       // Learning momentum
   int random_offset ,    // Starting index in shuffle_index for random sampling hidden1 if not mean_field
   float sparse_pen ,     // Sparsity penalty
   float sparse_targ      // Sparsity target
   )
{
   int icase, ihid, k ;
   float sum, frac_on, frand ;

   ihid = blockIdx.x * blockDim.x + threadIdx.x ;

   if (ihid >= d_nhid)
      return ;

   sum = frac_on = 0.0f ;
   if (d_mean_field) {
      for (icase=0 ; icase<nc ; icase++) {
         sum += d_hidden1[icase*d_nhid_cols+ihid] - d_hidden2[icase*d_nhid_cols+ihid] ;
         frac_on += d_hid_on_frac[icase*d_nhid_cols+ihid] ;
         }
      }
   else {
      for (icase=0 ; icase<nc ; icase++) {
         k = ((unsigned int) (icase * d_nhid + ihid + random_offset)) % d_ncases ;
         frand = (float) d_shuffle_index[k] / (float) d_ncases ;
         d_hidden_act[icase*d_nhid_cols+ihid] = (frand < d_hidden1[icase*d_nhid_cols+ihid])  ?  1.0f : 0.0f ;
         sum += d_hidden_act[icase*d_nhid_cols+ihid] - d_hidden2[icase*d_nhid_cols+ihid] ;
         frac_on += d_hid_on_frac[icase*d_nhid_cols+ihid] ;
         }
      }

   sum /= nc ;
   frac_on /= nc ;
   d_hid_on_smoothed[ihid] = 0.95f * d_hid_on_smoothed[ihid] + 0.05f * frac_on ;
   sum -= sparse_pen * (d_hid_on_smoothed[ihid] - sparse_targ) ;
   if (d_hid_on_smoothed[ihid] < 0.01)
      sum -= 0.5 * (d_hid_on_smoothed[ihid] - 0.01) ;       // 0.5 is heuristic
   if (d_hid_on_smoothed[ihid] > 0.99)
      sum -= 0.5 * (d_hid_on_smoothed[ihid] - 0.99) ;

   d_hid_bias_inc[ihid] = momentum * d_hid_bias_inc[ihid] + rate * sum ;
   d_hid_bias[ihid] += d_hid_bias_inc[ihid] ;
}


int cuda_update_hid_bias (
   int nc ,                // Number of cases in this batch
   int nhid ,              // Number of hidden neurons
   double rate ,           // Learning rate
   double momentum ,       // Learning momentum
   int random_offset ,     // Starting index in shuffle_index for random sampling hidden1 if not mean_field
   double sparse_pen ,     // Sparsity penalty
   double sparse_targ ,    // Sparsity target
   double *hid_bias ,      // Hidden bias vector, nhid long
   double *hid_bias_inc    // Hidden bias increment vector, carries over from batch to batch, nhid long
   )
{
   int i, warpsize, blocks_per_grid, threads_per_block ;
   char msg[256] ;
   cudaError_t error_id ;

   warpsize = deviceProp.warpSize ;      // Threads per warp, likely 32 well into the future

   threads_per_block = (nhid + warpsize - 1) / warpsize * warpsize ;
   if (threads_per_block > 4 * warpsize)
      threads_per_block = 4 * warpsize ;
   blocks_per_grid = (nhid + threads_per_block - 1) / threads_per_block ;

   device_update_hid_bias <<< blocks_per_grid , threads_per_block >>>
              ( nc , (float) rate , (float) momentum , random_offset ,
              (float) sparse_pen , (float) sparse_targ ) ;   
   cudaThreadSynchronize() ;
   error_id = cudaGetLastError () ;
   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_update_in_bias launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      return 1 ;
      }

   if (hid_bias != NULL) {
      error_id = cudaMemcpy ( fdata , h_hid_bias , nhid * sizeof(float) , cudaMemcpyDeviceToHost ) ;
      for (i=0 ; i<nhid ; i++)
         hid_bias[i] = fdata[i] ;
      if (error_id == cudaSuccess) {
         error_id = cudaMemcpy ( fdata , h_hid_bias_inc , nhid * sizeof(float) , cudaMemcpyDeviceToHost ) ;
         for (i=0 ; i<nhid ; i++)
            hid_bias_inc[i] = fdata[i] ;
         }
      if (error_id != cudaSuccess) {
         sprintf_s ( msg , 255 , "cuda_update_hid_bias Memcpy error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
         audit ( msg ) ;
         return 1 ;
         }
      }

   return 0 ;
}


/*
------------------------------------------------------------------------------------------------

   cuda_update_weights

------------------------------------------------------------------------------------------------
*/

__global__ void device_update_weights (
   int nc ,               // Number of cases in this batch
   float rate ,           // Learning rate
   float momentum ,       // Learning momentum
   float weight_pen ,     // Weight penalty
   float sparse_pen ,     // Sparsity penalty
   float sparse_targ      // Sparsity target
   )
{
   int icase, ivis, ihid ;
   float sum ;

   ivis = blockIdx.x * blockDim.x + threadIdx.x ;
   if (ivis >= d_n_inputs)
      return ;

   ihid = blockIdx.y ;

   sum = 0.0f ;
   if (d_mean_field) {
      for (icase=0 ; icase<nc ; icase++)
         sum +=  d_hidden1[icase*d_nhid_cols+ihid] * d_visible1[icase*d_n_inputs_cols+ivis] -
                 d_hidden2[icase*d_nhid_cols+ihid] * d_visible2[icase*d_n_inputs_cols+ivis] ;
      }
   else {
      for (icase=0 ; icase<nc ; icase++)
         sum +=  d_hidden_act[icase*d_nhid_cols+ihid] * d_visible1[icase*d_n_inputs_cols+ivis] -
                 d_hidden2[icase*d_nhid_cols+ihid] * d_visible2[icase*d_n_inputs_cols+ivis] ;
      }
   sum /= nc ;
   sum -= weight_pen * d_w[ihid*d_n_inputs_cols+ivis] ;
   sum -= d_data_mean[ivis] * sparse_pen * (d_hid_on_smoothed[ihid] - sparse_targ) ;
   if (d_hid_on_smoothed[ihid] < 0.01)
      sum -= d_data_mean[ivis] * 0.5 * (d_hid_on_smoothed[ihid] - 0.01) ;       // 0.5 is heuristic
   if (d_hid_on_smoothed[ihid] > 0.99)
      sum -= d_data_mean[ivis] * 0.5 * (d_hid_on_smoothed[ihid] - 0.99) ;

   d_w_grad[ihid*d_n_inputs_cols+ivis] = sum ;
   d_w_inc[ihid*d_n_inputs_cols+ivis] = momentum * d_w_inc[ihid*d_n_inputs_cols+ivis] + rate * sum ;
   d_w[ihid*d_n_inputs_cols+ivis] += d_w_inc[ihid*d_n_inputs_cols+ivis] ;
}


int cuda_update_weights (
   int nc ,                // Number of cases in this batch
   int n_inputs ,          // Number of inputs
   int nhid ,              // Number of hidden neurons
   double rate ,           // Learning rate
   double momentum ,       // Learning momentum
   double weight_pen ,     // Weight penalty
   double sparse_pen ,     // Sparsity penalty
   double sparse_targ ,    // Sparsity target
   double *w ,             // Weight matrix, nhid sets of n_inputs weights
   double *w_inc ,         // Weight increment array, carries over from batch to batch, nhid * n_inputs
   double *w_grad          // We'll need grad for auto update of rate; nhid * n_inputs
   )
{
   int ivis, ihid, warpsize, threads_per_block ;
   int n_inputs_cols ;
   char msg[256] ;
   dim3 block_launch ;
   cudaError_t error_id ;

   warpsize = deviceProp.warpSize ; // Threads per warp, likely 32 well into the future

   threads_per_block = (n_inputs + warpsize - 1) / warpsize * warpsize ;
   if (threads_per_block > 4 * warpsize)
      threads_per_block = 4 * warpsize ;

   block_launch.x = (n_inputs + threads_per_block - 1) / threads_per_block ;
   block_launch.y = nhid ;
   block_launch.z = 1 ;

   device_update_weights <<< block_launch , threads_per_block >>>
              ( nc , (float) rate , (float) momentum , (float) weight_pen ,
              (float) sparse_pen , (float) sparse_targ ) ;   
   cudaThreadSynchronize() ;
   error_id = cudaGetLastError () ;
   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_update_weights launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      return 1 ;
      }

   if (w != NULL) {
      n_inputs_cols = (n_inputs + 31) / 32 * 32 ;
      error_id = cudaMemcpy ( fdata , h_w , nhid * n_inputs_cols * sizeof(float) , cudaMemcpyDeviceToHost ) ;
      for (ihid=0 ; ihid<nhid ; ihid++) {
         for (ivis=0 ; ivis<n_inputs ; ivis++)
            w[ihid*n_inputs+ivis] = fdata[ihid*n_inputs_cols+ivis] ;
         }
      if (error_id == cudaSuccess) {
         error_id = cudaMemcpy ( fdata , h_w_inc , nhid * n_inputs_cols * sizeof(float) , cudaMemcpyDeviceToHost ) ;
         for (ihid=0 ; ihid<nhid ; ihid++) {
            for (ivis=0 ; ivis<n_inputs ; ivis++)
               w_inc[ihid*n_inputs+ivis] = fdata[ihid*n_inputs_cols+ivis] ;
            }
         }
      if (error_id == cudaSuccess) {
         error_id = cudaMemcpy ( fdata , h_w_grad , nhid * n_inputs_cols * sizeof(float) , cudaMemcpyDeviceToHost ) ;
         for (ihid=0 ; ihid<nhid ; ihid++) {
            for (ivis=0 ; ivis<n_inputs ; ivis++)
               w_grad[ihid*n_inputs+ivis] = fdata[ihid*n_inputs_cols+ivis] ;
            }
         }
      if (error_id != cudaSuccess) {
         sprintf_s ( msg , 255 , "cuda_update_weights Memcpy error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
         audit ( msg ) ;
         return 1 ;
         }
      }

   return 0 ;
}


/*
------------------------------------------------------------------------------------------------

   cuda_transpose

------------------------------------------------------------------------------------------------
*/

__global__ void device_transpose ()
{
   int ivis, ihid ;

   ivis = blockIdx.x * blockDim.x + threadIdx.x ;
   if (ivis >= d_n_inputs)
      return ;

   ihid = blockIdx.y ;

   d_wtr[ivis*d_nhid_cols+ihid] = d_w[ihid*d_n_inputs_cols+ivis] ;
}


int cuda_transpose (
   int n_inputs ,    // Number of inputs
   int nhid          // Number of hidden neurons
   )
{
   int warpsize, threads_per_block ;
   char msg[256] ;
   dim3 block_launch ;
   cudaError_t error_id ;

   warpsize = deviceProp.warpSize ;  // Threads per warp, likely 32 well into the future

   threads_per_block = (n_inputs + warpsize - 1) / warpsize * warpsize ;
   block_launch.x = (n_inputs + threads_per_block - 1) / threads_per_block ;
   block_launch.y = nhid ;
   block_launch.z = 1 ;

   device_transpose <<< block_launch , threads_per_block >>> () ;
   cudaThreadSynchronize() ;
   error_id = cudaGetLastError () ;
   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_transpose launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      return 1 ;
      }

   return 0 ;
}


/*
--------------------------------------------------------------------------------

   RBM_CUDA_CLEANUP - Cleanup after CUDA RBM processing

--------------------------------------------------------------------------------
*/

void rbm_cuda_cleanup ()
{
   char msg[256] ;

   sprintf_s ( msg, 255, "CUDA rbm_cuda_cleanup" ) ;
   MEMTEXT ( msg ) ;
   if (h_data != NULL) {
      cudaFree ( h_data ) ;
      h_data = NULL ;
      }
   if (h_data_mean != NULL) {
      cudaFree ( h_data_mean ) ;
      h_data_mean = NULL ;
      }
   if (h_in_bias != NULL) {
      cudaFree ( h_in_bias ) ;
      h_in_bias = NULL ;
      }
   if (h_hid_bias != NULL) {
      cudaFree ( h_hid_bias ) ;
      h_hid_bias = NULL ;
      }
   if (h_w != NULL) {
      cudaFree ( h_w ) ;
      h_w = NULL ;
      }
   if (h_wtr != NULL) {
      cudaFree ( h_wtr ) ;
      h_wtr = NULL ;
      }
   if (h_shuffle_index != NULL) {
      cudaFree ( h_shuffle_index ) ;
      h_shuffle_index = NULL ;
      }
   if (h_visible1 != NULL) {
      cudaFree ( h_visible1 ) ;
      h_visible1 = NULL ;
      }
   if (h_visible2 != NULL) {
      cudaFree ( h_visible2 ) ;
      h_visible2 = NULL ;
      }
   if (h_hidden1 != NULL) {
      cudaFree ( h_hidden1 ) ;
      h_hidden1 = NULL ;
      }
   if (h_hidden2 != NULL) {
      cudaFree ( h_hidden2 ) ;
      h_hidden2 = NULL ;
      }
   if (h_hidden_act != NULL) {
      cudaFree ( h_hidden_act ) ;
      h_hidden_act = NULL ;
      }
   if (h_in_bias_inc != NULL) {
      cudaFree ( h_in_bias_inc ) ;
      h_in_bias_inc = NULL ;
      }
   if (h_hid_bias_inc != NULL) {
      cudaFree ( h_hid_bias_inc ) ;
      h_hid_bias_inc = NULL ;
      }
   if (h_hid_on_frac != NULL) {
      cudaFree ( h_hid_on_frac ) ;
      h_hid_on_frac = NULL ;
      }
   if (h_hid_on_smoothed != NULL) {
      cudaFree ( h_hid_on_smoothed ) ;
      h_hid_on_smoothed = NULL ;
      }
   if (h_w_inc != NULL) {
      cudaFree ( h_w_inc ) ;
      h_w_inc = NULL ;
      }
   if (h_w_grad != NULL) {
      cudaFree ( h_w_grad ) ;
      h_w_grad = NULL ;
      }
   if (h_prev_grad != NULL) {
      cudaFree ( h_prev_grad ) ;
      h_prev_grad = NULL ;
      }
   if (h_err_vec != NULL) {
      cudaFree ( h_err_vec ) ;
      h_err_vec = NULL ;
      }
   if (h_len_out != NULL) {
      cudaFree ( h_len_out ) ;
      h_len_out = NULL ;
      }
   if (h_dot_out != NULL) {
      cudaFree ( h_dot_out ) ;
      h_dot_out = NULL ;
      }

   if (reduc_fdata != NULL) {
      FREE ( reduc_fdata ) ;
      reduc_fdata = NULL ;
      }

   if (fdata != NULL) {
      FREE ( fdata ) ;
      fdata = NULL ;
      }

   cudaDeviceReset () ;
}
