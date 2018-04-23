/******************************************************************************/
/*                                                                            */
/*  MLFN.CU - Core CUDA routines for MLFN                                     */
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

// This is used as intermediary between device's float and hosts double

static float *fdata = NULL ;
static int n_hid_weights ;  // Total number of hidden weights across all layers
static int n_out_weights ;  // Total number of output weights


// This is strictly for printing memory allocation info for the user

static double total_memory = 0.0 ;


// These are for the reductions used in device_mse
// The number of threads MUST be a power of two!
// The number of blocks given here is a maximum.  The actual number may be less.

#define REDUC_THREADS 256
#define REDUC_BLOCKS 64

static float *reduc_fdata = NULL ;


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

__constant__ int d_ncases ;                // Number of cases in complete training set
__constant__ int d_n_trn_inputs ;          // Number of first-layer inputs (training data)
__constant__ int d_ntarg ;                 // Number of targets (output neurons)

static       int *h_nhid = NULL ;          // Number of neurons in each of the hidden layers
__constant__ int *d_nhid ;

static       float *h_trn_data = NULL ;    // Raw training data; ncases by n_trn_inputs
__constant__ float *d_trn_data ;

static       float *h_targets = NULL ;     // Target data; ncases by ntarg
__constant__ float *d_targets ;

static       int *h_class = NULL ;         // If classification (SoftMax), class id is here
__constant__ int *d_class ;

static       float *hidden_weights = NULL ;// Weight matricies for hidden layer
static       float **h_whid = NULL ;
__constant__ float **d_whid ;

static       float *h_wout = NULL ;
__constant__ float *d_wout ;

static       double *activations = NULL ;  // Activations of this layer, which we compute
static       double **h_act = NULL ;       // Array of pointers to each layer
__constant__ double **d_act ;

static       double *h_output = NULL ;     // Output activations
__constant__ double *d_output ;

static       float *h_mse_out = NULL ;
__constant__ float *d_mse_out ;

static       double *h_this_delta = NULL ; // Delta for current layer
__constant__ double *d_this_delta ;

static       double *h_prior_delta = NULL ;// Delta for next layer back
__constant__ double *d_prior_delta ;

// WARNING... If gradient is ever double instead of float, see MLFN_CUDA.CPP for integer overflow check!
static       int h_gradlen ;               // Length of complete gradient for a case
__constant__ int d_gradlen ;
static       float *h_gradient = NULL ;    // Gradient for all layers, including output
__constant__ float *d_gradient ;
static       float **h_grad_ptr = NULL ;   // Pointers to locations in gradient for each layer
__constant__ float **d_grad_ptr ;

static cudaDeviceProp deviceProp ;

// Function declarations

__global__ void device_hidden_activation ( int istart , int istop , int ilayer ) ;
__global__ void device_output_activation ( int istart , int n_inputs , int ilayer ) ;
__global__ void device_output_delta ( int istart , int istop , int ntarg ) ;
__global__ void device_softmax_delta ( int istart , int istop , int ntarg ) ;
__global__ void device_output_gradient ( int nc , int ilayer ) ;
__global__ void device_first_hidden_gradient ( int istart , int istop , int only_hidden ) ;
__global__ void device_subsequent_hidden_gradient ( int nc , int ilayer , int last_hidden ) ;
__global__ void device_move_delta ( int nhid ) ;
__global__ void device_softmax ( int istart , int istop ) ;
__global__ void device_fetch_gradient ( int nc ) ;
__global__ void device_mse () ;
__global__ void device_ll () ;


/*
--------------------------------------------------------------------------------

   MLFN_CUDA_INIT - Initialize for CUDA MLFN processing

   This is called once before training begins, and mlfn_cuda_cleanup must
   be called after training is complete.

   Fdata is used here to translate data from double (on the host) to float (on the device).
   It is freed here, immediately after use, in most routines, but then
   permanently allocated as a last step.

--------------------------------------------------------------------------------
*/


int mlfn_cuda_init (
   int classifier ,       // Is this for classification? (SoftMax outputs)
   int *class_ids ,       // Class ids if classifier
   int ncases ,           // Number of training cases
   int n_inputs ,         // Number of inputs
   int ncols ,            // Number of columns in data (may exceed n_inputs)
   double *data ,         // Input data, ncases rows by ncols columns, of which first n_inputs are used
   int ntarg ,            // Number of targets (outputs; classes in classification)
   double *targets ,      // Targets, ncases by ntarg
   int max_batch ,        // Max size of any batch
   int n_layers ,         // Number of layers of neurons, including output
   int *nhid ,            // Number of neurons in each hidden layer
   char *error_msg        // Returns text of error if problem
   )
{
   int i, j, n, n_total, n_max, n_prior, memsize ;
   float *gptr, *fptr[MAX_LAYERS] ;
   double *dptr[MAX_LAYERS] ;
   char msg[256] ;
   cudaError_t error_id ;

   MEMTEXT ( "MLFN.cu: mlfn_cuda_init starting" ) ;
   cudalog ( "" ) ;

   CudaTimers.mlfn_ncalls_weights = 0 ;
   CudaTimers.mlfn_weights = 0 ;
   for (i=0 ; i<MAX_LAYERS ; i++) {
      CudaTimers.mlfn_ncalls_hidden[i] = 0 ;
      CudaTimers.mlfn_hidden[i] = 0 ;
      }
   CudaTimers.mlfn_ncalls_outact = 0 ;
   CudaTimers.mlfn_outact = 0 ;
   CudaTimers.mlfn_ncalls_softmax = 0 ;
   CudaTimers.mlfn_softmax = 0 ;
   CudaTimers.mlfn_ncalls_ll = 0 ;
   CudaTimers.mlfn_ll = 0 ;
   CudaTimers.mlfn_ncalls_mse = 0 ;
   CudaTimers.mlfn_mse = 0 ;
   CudaTimers.mlfn_ncalls_wpen = 0 ;
   CudaTimers.mlfn_wpen = 0 ;
   CudaTimers.mlfn_ncalls_outdelta = 0 ;
   CudaTimers.mlfn_outdelta = 0 ;
   CudaTimers.mlfn_ncalls_outgrad = 0 ;
   CudaTimers.mlfn_outgrad = 0 ;
   for (i=0 ; i<MAX_LAYERS ; i++) {
      CudaTimers.mlfn_ncalls_subgrad[i] = 0 ;
      CudaTimers.mlfn_subgrad[i] = 0 ;
      }
   CudaTimers.mlfn_ncalls_firstgrad = 0 ;
   CudaTimers.mlfn_firstgrad = 0 ;
   CudaTimers.mlfn_ncalls_fetchgrad = 0 ;
   CudaTimers.mlfn_fetchgrad = 0 ;


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
   Constants
*/

   cudaMemcpyToSymbol ( d_ncases , &ncases , sizeof(int) , 0 , cudaMemcpyHostToDevice ) ;
   cudaMemcpyToSymbol ( d_n_trn_inputs , &n_inputs , sizeof(int) , 0 , cudaMemcpyHostToDevice ) ;
   cudaMemcpyToSymbol ( d_ntarg , &ntarg , sizeof(int) , 0 , cudaMemcpyHostToDevice ) ;


/*
   nhid - Array of number of neurons in each hidden layer
*/

   memsize = (n_layers-1) * sizeof(int) ;
   total_memory += memsize ;
   error_id = cudaMalloc ( (void **) &h_nhid , (size_t) memsize ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC nhid = %llx  (%d bytes, total=%.2lf MB)",
               (unsigned long long) h_nhid, memsize, total_memory / (1024 * 1024) ) ;
   cudalog ( msg ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc nhid (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }
   error_id = cudaMemcpy ( h_nhid , nhid , (n_layers-1) * sizeof(int) , cudaMemcpyHostToDevice ) ;
   if (error_id  ==  cudaSuccess)
      error_id = cudaMemcpyToSymbol ( d_nhid , &h_nhid , sizeof(int *) , 0 , cudaMemcpyHostToDevice ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMemcpy nhid (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_ERROR ;
      }


/*
   Data - We must extract only the first n_inputs columns from the ncols columns in data
*/

   fdata = (float *) MALLOC ( ncases * n_inputs * sizeof(float) ) ;
   if (fdata == NULL)
      return ERROR_INSUFFICIENT_MEMORY ;

   memsize = ncases * n_inputs * sizeof(float) ;
   total_memory += memsize ;
   error_id = cudaMalloc ( (void **) &h_trn_data , (size_t) memsize ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC data = %llx  (%d bytes, total=%.2lf MB)",
               (unsigned long long) h_trn_data, memsize, total_memory / (1024 * 1024) ) ;
   cudalog ( msg ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc data (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }

   for (i=0 ; i<ncases ; i++) {
      for (j=0 ; j<n_inputs ; j++)
         fdata[i*n_inputs+j] = (float) data[i*ncols+j] ;
      }

   error_id = cudaMemcpy ( h_trn_data , fdata , ncases * n_inputs * sizeof(float) , cudaMemcpyHostToDevice ) ;
   FREE ( fdata ) ;
   fdata = NULL ;

   if (error_id == cudaSuccess)
      error_id = cudaMemcpyToSymbol ( d_trn_data , &h_trn_data , sizeof(float *) , 0 , cudaMemcpyHostToDevice ) ;

   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad data copy %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_ERROR ;
      }


/*
   Targets
*/

   fdata = (float *) MALLOC ( ncases * ntarg * sizeof(float) ) ;
   if (fdata == NULL)
      return ERROR_INSUFFICIENT_MEMORY ;

   memsize = ncases * ntarg * sizeof(float) ;
   total_memory += memsize ;
   error_id = cudaMalloc ( (void **) &h_targets , (size_t) memsize ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC targets = %llx  (%d bytes, total=%.2lf MB)",
               (unsigned long long) h_targets, memsize, total_memory / (1024 * 1024) ) ;
   cudalog ( msg ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc targets (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }

   for (i=0 ; i<ncases ; i++) {
      for (j=0 ; j<ntarg ; j++)
         fdata[i*ntarg+j] = (float) targets[i*ntarg+j] ;
      }

   error_id = cudaMemcpy ( h_targets , fdata , ncases * ntarg * sizeof(float) , cudaMemcpyHostToDevice ) ;
   FREE ( fdata ) ;
   fdata = NULL ;

   if (error_id == cudaSuccess)
      error_id = cudaMemcpyToSymbol ( d_targets , &h_targets , sizeof(float *) , 0 , cudaMemcpyHostToDevice ) ;

   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad targets copy %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_ERROR ;
      }


/*
   Classes if this is a classifier
*/

   if (classifier) {
      memsize = ncases * sizeof(int) ;
      total_memory += memsize ;
      error_id = cudaMalloc ( (void **) &h_class , (size_t) memsize ) ;
      sprintf_s ( msg, 255 , "CUDA MALLOC class = %llx  (%d bytes, total=%.2lf MB)",
                  (unsigned long long) h_class, memsize, total_memory / (1024 * 1024) ) ;
      cudalog ( msg ) ;
      if (error_id  !=  cudaSuccess) {
         sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc class (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
         return ERROR_CUDA_MEMORY ;
         }

      error_id = cudaMemcpy ( h_class , class_ids , ncases * sizeof(int) , cudaMemcpyHostToDevice ) ;

      if (error_id == cudaSuccess)
         error_id = cudaMemcpyToSymbol ( d_class , &h_class , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) ;

      if (error_id  !=  cudaSuccess) {
         sprintf_s ( error_msg , 255 , "CUDA init bad class copy %d: %s", error_id, cudaGetErrorString(error_id) ) ;
         return ERROR_CUDA_ERROR ;
         }
      }

/*
   Activations
*/

   n_total = 0 ;
   for (i=0 ; i<n_layers-1 ; i++)
      n_total += nhid[i] ;

   memsize = n_total * max_batch * sizeof(double) ;
   total_memory += memsize ;
   error_id = cudaMalloc ( (void **) &activations , (size_t) memsize ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC activations = %llx  (%d bytes, total=%.2lf MB)",
               (unsigned long long) activations, memsize, total_memory / (1024 * 1024) ) ;
   cudalog ( msg ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc activations (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }

   memsize = (n_layers-1) * sizeof(void *) ;
   total_memory += memsize ;
   error_id = cudaMalloc ( (void **) &h_act , (size_t) memsize ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC act = %llx  (%d bytes, total=%.2lf MB)",
               (unsigned long long) h_act, memsize, total_memory / (1024 * 1024) ) ;
   cudalog ( msg ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc act (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }

   cudaMemcpyToSymbol ( d_act , &h_act , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) ;

   n_total = 0 ;
   for (i=0 ; i<n_layers-1 ; i++) {
      dptr[i] = activations + n_total * max_batch ;
      n_total += nhid[i] ;
      }

   error_id = cudaMemcpy ( h_act , &dptr[0] , (n_layers-1) * sizeof(void *) , cudaMemcpyHostToDevice ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad act ptr copy %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_ERROR ;
      }


/*
   Output activations
*/

   memsize = ncases * ntarg * sizeof(double) ;
   total_memory += memsize ;
   error_id = cudaMalloc ( (void **) &h_output , (size_t) memsize ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC output = %llx  (%d bytes, total=%.2lf MB)",
               (unsigned long long) h_output, memsize, total_memory / (1024 * 1024) ) ;
   cudalog ( msg ) ;
   if (error_id  ==  cudaSuccess)
      error_id = cudaMemcpyToSymbol ( d_output , &h_output , sizeof(float *) , 0 , cudaMemcpyHostToDevice ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc output (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }

/*
   Hidden layer weights
*/

   n_total = 0 ;
   n_prior = n_inputs ;
   for (i=0 ; i<n_layers-1 ; i++) {
      n_total += nhid[i] * (n_prior + 1) ;
      n_prior = nhid[i] ;
      }

   n_hid_weights = n_total ;

   memsize = n_total * sizeof(float) ;
   total_memory += memsize ;
   error_id = cudaMalloc ( (void **) &hidden_weights , (size_t) memsize ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC hidden_weights = %llx  (%d bytes, total=%.2lf MB)",
               (unsigned long long) hidden_weights, memsize, total_memory / (1024 * 1024) ) ;
   cudalog ( msg ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc hidden_weights (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }

   memsize = (n_layers-1) * sizeof(float *) ;
   total_memory += memsize ;
   error_id = cudaMalloc ( (void **) &h_whid , (size_t) memsize ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC whid = %llx  (%d bytes, total=%.2lf MB)",
               (unsigned long long) h_whid, memsize, total_memory / (1024 * 1024) ) ;
   cudalog ( msg ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc whid (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }

   cudaMemcpyToSymbol ( d_whid , &h_whid , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) ;

   n_total = 0 ;
   n_prior = n_inputs ;
   for (i=0 ; i<n_layers-1 ; i++) {
      fptr[i] = hidden_weights + n_total ;
      n_total += nhid[i] * (n_prior + 1) ;
      n_prior = nhid[i] ;
      }

   error_id = cudaMemcpy ( h_whid , &fptr[0] , (n_layers-1) * sizeof(float *) , cudaMemcpyHostToDevice ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad whid ptr copy %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_ERROR ;
      }


/*
   Output weights
*/

   n_out_weights = ntarg * (nhid[n_layers-2]+1) ;
   memsize = n_out_weights * sizeof(float) ;
   total_memory += memsize ;
   error_id = cudaMalloc ( (void **) &h_wout , (size_t) memsize ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC wout = %llx  (%d bytes, total=%.2lf MB)",
               (unsigned long long) h_wout, memsize, total_memory / (1024 * 1024) ) ;
   cudalog ( msg ) ;
   if (error_id  ==  cudaSuccess)
      error_id = cudaMemcpyToSymbol ( d_wout , &h_wout , sizeof(float *) , 0 , cudaMemcpyHostToDevice ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc wout (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }

/*
   This delta, next delta
*/

   n_max = ntarg ;
   for (i=1 ; i<n_layers-1 ; i++) {  // We do not store delta for first hidden layer, so skip 0
      if (nhid[i] > n_max)
         n_max = nhid[i] ;
      }

   memsize = n_max * max_batch * sizeof(double) ;
   total_memory += memsize ;
   error_id = cudaMalloc ( (void **) &h_this_delta , (size_t) memsize ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC this_delta = %llx  (%d bytes, total=%.2lf MB)",
               (unsigned long long) h_this_delta, memsize, total_memory / (1024 * 1024) ) ;
   cudalog ( msg ) ;
   if (error_id  ==  cudaSuccess)
      error_id = cudaMemcpyToSymbol ( d_this_delta , &h_this_delta , sizeof(float *) , 0 , cudaMemcpyHostToDevice ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc this_delta (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }

   memsize = n_max * max_batch * sizeof(double) ;
   total_memory += memsize ;
   error_id = cudaMalloc ( (void **) &h_prior_delta , (size_t) memsize ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC prior_delta = %llx  (%d bytes, total=%.2lf MB)",
               (unsigned long long) h_prior_delta, memsize, total_memory / (1024 * 1024) ) ;
   cudalog ( msg ) ;
   if (error_id  ==  cudaSuccess)
      error_id = cudaMemcpyToSymbol ( d_prior_delta , &h_prior_delta , sizeof(float *) , 0 , cudaMemcpyHostToDevice ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc prior_delta (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }

/*
   Gradient (all layers, including output); grad_ptr
*/

   h_gradlen = 0 ;
   n_prior = n_inputs ;
   for (i=0 ; i<n_layers-1 ; i++) {
      h_gradlen += nhid[i] * (n_prior + 1) ;
      n_prior = nhid[i] ;
      }
   h_gradlen += ntarg * (n_prior + 1) ;
   cudaMemcpyToSymbol ( d_gradlen , &h_gradlen , sizeof(int) , 0 , cudaMemcpyHostToDevice ) ;

   memsize = h_gradlen * max_batch * sizeof(float) ;
   total_memory += memsize ;
   error_id = cudaMalloc ( (void **) &h_gradient , (size_t) memsize ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC h_gradient = %llx  (%d bytes, total=%.2lf MB)",
               (unsigned long long) h_gradient, memsize, total_memory / (1024 * 1024) ) ;
   cudalog ( msg ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc h_gradient (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }

   cudaMemcpyToSymbol ( d_gradient , &h_gradient , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) ;

   memsize = n_layers * sizeof(float *) ;
   total_memory += memsize ;
   error_id = cudaMalloc ( (void **) &h_grad_ptr , (size_t) memsize ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC grad_ptr = %llx  (%d bytes, total=%.2lf MB)",
               (unsigned long long) h_whid, memsize, total_memory / (1024 * 1024) ) ;
   cudalog ( msg ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc grad_ptr (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }

   cudaMemcpyToSymbol ( d_grad_ptr , &h_grad_ptr , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) ;

   gptr = h_gradient ;
   for (i=0 ; i<n_layers ; i++) {
      fptr[i] = gptr ;

      if (i == 0) {                        // First hidden layer?
         n = nhid[i] * (n_inputs+1) ;
         gptr += n ;
         }

      else if (i < n_layers-1) {           // Subsequent hidden layer?
         n = nhid[i] * (nhid[i-1]+1) ;
         gptr += n ;
         }
      }

   error_id = cudaMemcpy ( h_grad_ptr , &fptr[0] , n_layers * sizeof(void *) , cudaMemcpyHostToDevice ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad grad_ptr copy %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_ERROR ;
      }


/*
   MSE reduction stuff
*/

   memsize = REDUC_BLOCKS * sizeof(float) ;
   total_memory += memsize ;
   error_id = cudaMalloc ( (void **) &h_mse_out , (size_t) memsize ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC mse_out = %llx  (%d bytes, total=%.2lf MB)",
               (unsigned long long) h_mse_out, memsize, total_memory / (1024 * 1024) ) ;
   cudalog ( msg ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc mse_out (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }
   cudaMemcpyToSymbol ( d_mse_out , &h_mse_out , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) ;

   MEMTEXT ( "CUDA init reduc_fdata" ) ;
   reduc_fdata = (float *) MALLOC ( REDUC_BLOCKS * sizeof(float) ) ;
   if (reduc_fdata == NULL) {
      sprintf_s ( error_msg , 255 , "CUDA init bad MALLOC reduc_fdata" ) ;
      return ERROR_INSUFFICIENT_MEMORY ;  // New error return
      }


/*
   Allocate fdata large enough to handle all subsequent double <-> float transactions
*/

   fdata = (float *) MALLOC ( h_gradlen * sizeof(float) ) ;
   if (fdata == NULL)
      return ERROR_INSUFFICIENT_MEMORY ;


/*
   Set cache/shared memory preferences
*/

   error_id = cudaFuncSetCacheConfig ( device_hidden_activation , cudaFuncCachePreferL1 ) ;
   if (error_id == cudaSuccess)
      error_id = cudaFuncSetCacheConfig ( device_output_activation , cudaFuncCachePreferL1 ) ;
   if (error_id == cudaSuccess)
      error_id = cudaFuncSetCacheConfig ( device_output_delta , cudaFuncCachePreferL1 ) ;
   if (error_id == cudaSuccess)
      error_id = cudaFuncSetCacheConfig ( device_softmax_delta , cudaFuncCachePreferL1 ) ;
   if (error_id == cudaSuccess)
      error_id = cudaFuncSetCacheConfig ( device_output_gradient , cudaFuncCachePreferL1 ) ;
   if (error_id == cudaSuccess)
      error_id = cudaFuncSetCacheConfig ( device_first_hidden_gradient , cudaFuncCachePreferL1 ) ;
   if (error_id == cudaSuccess)
      error_id = cudaFuncSetCacheConfig ( device_subsequent_hidden_gradient , cudaFuncCachePreferL1 ) ;
   if (error_id == cudaSuccess)
      error_id = cudaFuncSetCacheConfig ( device_move_delta , cudaFuncCachePreferL1 ) ;
   if (error_id == cudaSuccess)
      error_id = cudaFuncSetCacheConfig ( device_softmax , cudaFuncCachePreferL1 ) ;
   if (error_id == cudaSuccess)
      error_id = cudaFuncSetCacheConfig ( device_fetch_gradient , cudaFuncCachePreferL1 ) ;
   if (error_id == cudaSuccess)
      error_id = cudaFuncSetCacheConfig ( device_mse  , cudaFuncCachePreferNone ) ;
   if (error_id == cudaSuccess)
      error_id = cudaFuncSetCacheConfig ( device_ll  , cudaFuncCachePreferNone ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaFuncSetCacheConfig" ) ;
      return ERROR_CUDA_ERROR ;
      }

   MEMTEXT ( "MLFN.cu: mlfn_cuda_init finished" ) ;
   return 0 ;
}


/*
--------------------------------------------------------------------------------

   cuda_weights_to_device - Called from MLFN_CUDA.CPP to copy weights

--------------------------------------------------------------------------------
*/

int cuda_weights_to_device (
   int n_inputs ,
   int ntarg ,
   int n_layers ,
   int *nhid ,
   double **hid_weights ,
   double *final_layer_weights )
{
   int n_prior, ilayer, ineuron, ivar ;
   double *wptr ;
   float *fptr ;
   char msg[256] ;
   cudaError_t error_id ;
   
   fptr = fdata ;
   n_prior = n_inputs ;

   for (ilayer=0 ; ilayer<n_layers-1 ; ilayer++) {
      wptr = hid_weights[ilayer] ;
      for (ivar=0 ; ivar<=n_prior ; ivar++) {
         for (ineuron=0 ; ineuron<nhid[ilayer] ; ineuron++)
            *fptr++ = (float) wptr[ineuron*(n_prior+1)+ivar] ;
         }
      n_prior = nhid[ilayer] ;
      }

   error_id = cudaMemcpy ( hidden_weights , fdata , n_hid_weights * sizeof(float) , cudaMemcpyHostToDevice ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( msg , 255 , "CUDA ERROR: bad weights_to_device hid %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( "" ) ;
      audit ( msg ) ;
      MEMTEXT ( msg ) ;
      return ERROR_CUDA_ERROR ;
      }

   fptr = fdata ;
   wptr = final_layer_weights ;

   for (ivar=0 ; ivar<=n_prior ; ivar++) {
      for (ineuron=0 ; ineuron<ntarg ; ineuron++)
         *fptr++ = (float) wptr[ineuron*(n_prior+1)+ivar] ;
      }

   error_id = cudaMemcpy ( h_wout , fdata , n_out_weights * sizeof(float) , cudaMemcpyHostToDevice ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( msg , 255 , "CUDA ERROR: bad weights_to_device out %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( "" ) ;
      audit ( msg ) ;
      MEMTEXT ( msg ) ;
      return ERROR_CUDA_ERROR ;
      }

   return 0 ;
}


/*
--------------------------------------------------------------------------------

   hidden_activation - Compute activations for a single hidden layer

--------------------------------------------------------------------------------
*/

__global__ void device_hidden_activation (
   int istart ,       // First case in this batch
   int istop ,        // One past last case
   int ilayer         // Layer to process
   )
{
   int icase, ihid, i_input, n_inputs, nhid ;
   float *f_inptr, *wptr ;
   double sum, *actptr, *d_inptr ;

   ihid = blockIdx.x * blockDim.x + threadIdx.x ;
   nhid = d_nhid[ilayer] ;

   if (ihid >= nhid)
      return ;

   icase = blockIdx.y ;

   wptr = d_whid[ilayer] ;
   actptr = d_act[ilayer] ;
   sum = 0.0 ;

   if (ilayer == 0) {
      n_inputs = d_n_trn_inputs ;
      f_inptr = d_trn_data + (icase+istart)*n_inputs ;
      for (i_input=0 ; i_input<n_inputs ; i_input++)
         sum += wptr[i_input*nhid+ihid] * f_inptr[i_input] ;
      sum += wptr[n_inputs*nhid+ihid] ;  // Bias
      }
   else {
      n_inputs = d_nhid[ilayer-1] ;
      d_inptr = d_act[ilayer-1] + icase*n_inputs ;
      for (i_input=0 ; i_input<n_inputs ; i_input++)
         sum += wptr[i_input*nhid+ihid] * d_inptr[i_input] ;
      sum += wptr[n_inputs*nhid+ihid] ;  // Bias
      }

   actptr[icase*nhid+ihid] = 1.0 / (1.0 + __expf(-sum)) ;
}

int cuda_hidden_activation (
   int istart ,    // First case in this batch
   int istop ,     // One past last case
   int nhid ,      // Number of hidden neurons in this layer
   int ilayer      // Layer to process
   )
{
   int warpsize, threads_per_block ;
   char msg[256] ;
   dim3 block_launch ;
   cudaError_t error_id ;

   warpsize = deviceProp.warpSize ;      // Threads per warp, likely 32 well into the future

   threads_per_block = (nhid + warpsize - 1) / warpsize * warpsize ;
   if (threads_per_block > 4 * warpsize)
      threads_per_block = 4 * warpsize ;

   block_launch.x = (nhid + threads_per_block - 1) / threads_per_block ;
   block_launch.y = istop - istart ;
   block_launch.z = 1 ;

   device_hidden_activation <<< block_launch , threads_per_block >>> ( istart , istop , ilayer ) ;   
   cudaDeviceSynchronize() ;
   error_id = cudaGetLastError () ;
   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_hidden_activation launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      MEMTEXT ( msg ) ;
      return 1 ;
      }

   return 0 ;
}


/*
--------------------------------------------------------------------------------

   output_activation - Compute activations for the output layer

--------------------------------------------------------------------------------
*/

__global__ void device_output_activation (
   int istart ,       // First case in this batch
   int n_inputs ,     // Number of inputs to the output layer, not counting bias
   int ilayer         // Hidden layer which feeds the output layer
   )
{
   int icase, iout, i_input ;
   double sum, *inptr ;

   iout = blockIdx.x * blockDim.x + threadIdx.x ;

   if (iout >= d_ntarg)
      return ;

   icase = blockIdx.y ;

   inptr = d_act[ilayer] + icase * n_inputs ;
   sum = 0.0 ;

   for (i_input=0 ; i_input<n_inputs ; i_input++)
      sum += d_wout[i_input*d_ntarg+iout] * inptr[i_input] ;
   sum += d_wout[n_inputs*d_ntarg+iout] ;  // Bias

   d_output[(icase+istart)*d_ntarg+iout] = sum ;
}

int cuda_output_activation (
   int istart ,    // First case in this batch
   int istop ,     // One past last case
   int n_inputs ,  // Number of inputs to the output layer, not counting bias
   int ntarg ,     // Number of targets (outputs)
   int ilayer      // Hidden layer which feeds the output layer
   )
{
   int warpsize, threads_per_block ;
   char msg[256] ;
   dim3 block_launch ;
   cudaError_t error_id ;

   warpsize = deviceProp.warpSize ;      // Threads per warp, likely 32 well into the future

   threads_per_block = (ntarg + warpsize - 1) / warpsize * warpsize ;
   if (threads_per_block > 4 * warpsize)
      threads_per_block = 4 * warpsize ;

   block_launch.x = (ntarg + threads_per_block - 1) / threads_per_block ;
   block_launch.y = istop - istart ;
   block_launch.z = 1 ;

   device_output_activation <<< block_launch , threads_per_block >>> ( istart , n_inputs , ilayer ) ;   
   cudaDeviceSynchronize() ;
   error_id = cudaGetLastError () ;
   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_output_activation launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      MEMTEXT ( msg ) ;
      return 1 ;
      }

   return 0 ;
}


/*
--------------------------------------------------------------------------------

   output_delta - Put output delta into this_delta

--------------------------------------------------------------------------------
*/

__global__ void device_output_delta (
   int istart ,       // First case in this batch
   int istop ,        // One past last case
   int ntarg          // Number of targets (outputs)
   )
{
   int icase, iout ;

   iout = blockIdx.x * blockDim.x + threadIdx.x ;

   if (iout >= d_ntarg)
      return ;

   icase = blockIdx.y ;

   d_this_delta[icase*ntarg+iout] = 2.0 * (d_targets[(icase+istart)*ntarg+iout] - d_output[(icase+istart)*ntarg+iout]) ;
}

__global__ void device_softmax_delta (
   int istart ,       // First case in this batch
   int istop ,        // One past last case
   int ntarg          // Number of targets (outputs)
   )
{
   int icase, iout ;

   iout = blockIdx.x * blockDim.x + threadIdx.x ;

   if (iout >= d_ntarg)
      return ;

   icase = blockIdx.y ;

   d_this_delta[icase*ntarg+iout] = d_targets[(icase+istart)*ntarg+iout] - d_output[(icase+istart)*ntarg+iout] ;
}

int cuda_output_delta (
   int istart ,      // First case in this batch
   int istop ,       // One past last case
   int classifier ,  // Is this a classifier (SoftMax outputs)?
   int ntarg         // Number of targets (outputs)
   )
{
   int warpsize, threads_per_block ;
   char msg[256] ;
   dim3 block_launch ;
   cudaError_t error_id ;

   warpsize = deviceProp.warpSize ;      // Threads per warp, likely 32 well into the future

   threads_per_block = (ntarg + warpsize - 1) / warpsize * warpsize ;
   if (threads_per_block > 4 * warpsize)
      threads_per_block = 4 * warpsize ;

   block_launch.x = (ntarg + threads_per_block - 1) / threads_per_block ;
   block_launch.y = istop - istart ;
   block_launch.z = 1 ;

   if (classifier)
      device_softmax_delta <<< block_launch , threads_per_block >>> ( istart , istop , ntarg ) ;   
   else
      device_output_delta <<< block_launch , threads_per_block >>> ( istart , istop , ntarg ) ;   

   cudaDeviceSynchronize() ;
   error_id = cudaGetLastError () ;
   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_output_delta launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      MEMTEXT ( msg ) ;
      return 1 ;
      }

   return 0 ;
}


/*
--------------------------------------------------------------------------------

   output_gradient - Compute output layer gradient

--------------------------------------------------------------------------------
*/

__global__ void device_output_gradient (
   int nc ,        // Number of cases in batch
   int ilayer      // Hidden layer which feeds the output layer
   )
{
   int icase, iout, ihid, nhid ;
   float *gptr ;
   double input ;

   ihid = blockIdx.x * blockDim.x + threadIdx.x ;
   nhid = d_nhid[ilayer] ;       // Neurons in last hidden layer
   icase = blockIdx.y ;

   if (ihid > nhid)
      return ;
   else if (ihid < nhid)
      input = d_act[ilayer][icase*nhid+ihid] ;
   else
      input = 1.0 ;              // Bias

   iout = blockIdx.z ;

   gptr = d_grad_ptr[ilayer+1] + icase * d_gradlen ; // Gradient of output layer

   gptr[iout*(nhid+1)+ihid] = d_this_delta[icase*d_ntarg+iout] * input ;
}

int cuda_output_gradient (
   int nc ,        // Number of cases in batch
   int nhid ,      // Number of neurons in last hidden layer
   int ilayer ,    // And its layer index
   int ntarg       // Number of targets (outputs)
   )
{
   int warpsize, threads_per_block ;
   char msg[256] ;
   dim3 block_launch ;
   cudaError_t error_id ;

   warpsize = deviceProp.warpSize ;      // Threads per warp, likely 32 well into the future

   threads_per_block = (nhid + 1 + warpsize - 1) / warpsize * warpsize ;
   if (threads_per_block > 4 * warpsize)
      threads_per_block = 4 * warpsize ;

   block_launch.x = (nhid + 1 + threads_per_block - 1) / threads_per_block ; // Include bias
   block_launch.y = nc ;
   block_launch.z = ntarg ;

   device_output_gradient <<< block_launch , threads_per_block >>> ( nc , ilayer ) ;   
   cudaDeviceSynchronize() ;
   error_id = cudaGetLastError () ;
   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_output_gradient launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      MEMTEXT ( msg ) ;
      return 1 ;
      }

   return 0 ;
}


/*
--------------------------------------------------------------------------------

   first_hidden_gradient - Compute gradient for first hidden layer

--------------------------------------------------------------------------------
*/

__global__ void device_first_hidden_gradient (
   int istart ,       // First case in this batch
   int istop ,        // One past last case
   int only_hidden    // Is this the only hidden layer?
   )
{
   int j, icase, iin, ihid, nhid, ninp1, n_next ;
   float *gptr, *next_weights, input ;
   double *delta_ptr, this_act, delta ;

   iin = blockIdx.x * blockDim.x + threadIdx.x ;
   icase = blockIdx.y ;

   if (iin > d_n_trn_inputs)
      return ;
   else if (iin < d_n_trn_inputs)
      input = d_trn_data[(icase+istart)*d_n_trn_inputs+iin] ;  // Feed coming into this layer
   else
      input = 1.0f ;             // Bias

   ihid = blockIdx.z ;
   nhid = d_nhid[0] ;            // Neurons in this hidden layer
   ninp1 = d_n_trn_inputs + 1 ;  // We mustn't forget the bias

   if (only_hidden) {
      n_next = d_ntarg ;
      next_weights = d_wout + ihid * n_next ;
      }
   else {
      n_next = d_nhid[1] ;
      next_weights = d_whid[1] + ihid * n_next;
      }

   delta_ptr = d_this_delta + icase * n_next ; // Delta for this case

   delta = 0.0 ;
   for (j=0 ; j<n_next ; j++)
      delta += delta_ptr[j] * next_weights[j] ;
   this_act = d_act[0][icase*nhid+ihid] ;
   delta *= this_act * (1.0 - this_act) ;

   gptr = d_grad_ptr[0] + icase * d_gradlen ;  // Gradient of first hidden layer
   gptr[ihid*ninp1+iin] = delta * input ;
}

int cuda_first_hidden_gradient (
   int istart ,       // First case in this batch
   int istop ,        // One past last case
   int nin ,          // Number of model inputs
   int nhid ,         // Number of neurons in this layer
   int only_hidden    // Is this the only hidden layer?
   )
{
   int warpsize, threads_per_block ;
   char msg[256] ;
   dim3 block_launch ;
   cudaError_t error_id ;

   warpsize = deviceProp.warpSize ;      // Threads per warp, likely 32 well into the future

   threads_per_block = (nin + 1 + warpsize - 1) / warpsize * warpsize ;
   if (threads_per_block > 4 * warpsize)
      threads_per_block = 4 * warpsize ;

   block_launch.x = (nin + 1 + threads_per_block - 1) / threads_per_block ; // Include bias
   block_launch.y = istop - istart ;
   block_launch.z = nhid ;

   device_first_hidden_gradient <<< block_launch , threads_per_block >>> ( istart , istop , only_hidden ) ;   
   cudaDeviceSynchronize() ;
   error_id = cudaGetLastError () ;
   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_first_hidden_gradient launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      MEMTEXT ( msg ) ;
      return 1 ;
      }

   return 0 ;
}


/*
-----------------------------------------------------------------------------------

   subsequent_hidden_gradient - Compute gradient for hidden layers other than first

-----------------------------------------------------------------------------------
*/

__global__ void device_subsequent_hidden_gradient (
   int nc ,           // Number of cases in batch
   int ilayer ,       // Hidden layer being processed
   int last_hidden    // Is this the last hidden layer?
   )
{
   int j, icase, iin, ihid, nhid, nin, ninp1, n_next ;
   float *gptr, *next_weights ;
   double *delta_ptr, *prior_delta_ptr, this_act, delta, input ;

   iin = blockIdx.x * blockDim.x + threadIdx.x ;
   icase = blockIdx.y ;
   nin = d_nhid[ilayer-1] ;      // Number of inputs to each neuron in this layer

   if (iin > nin)
      return ;
   else if (iin < nin)
      input = d_act[ilayer-1][icase*nin+iin] ;
   else
      input = 1.0 ;              // Bias

   ihid = blockIdx.z ;
   nhid = d_nhid[ilayer] ;       // Neurons in this hidden layer
   ninp1 = nin + 1 ;             // We mustn't forget the bias, so nin+1

   if (last_hidden) {
      n_next = d_ntarg ;
      next_weights = d_wout + ihid * n_next ;
      }
   else {
      n_next = d_nhid[ilayer+1] ;
      next_weights = d_whid[ilayer+1] + ihid * n_next ;
      }

   delta_ptr = d_this_delta + icase * n_next ;      // Coming from the next layer, which was just done
   prior_delta_ptr = d_prior_delta + icase * nhid ; // Save for the next layer done, one layer back

   delta = 0.0 ;
   for (j=0 ; j<n_next ; j++)
      delta += delta_ptr[j] * next_weights[j] ;
   this_act = d_act[ilayer][icase*nhid+ihid] ;
   delta *= this_act * (1.0 - this_act) ;
   prior_delta_ptr[ihid] = delta ;            // Save it for the next layer back

   gptr = d_grad_ptr[ilayer] + icase * d_gradlen ;  // Gradient of this hidden layer
   gptr[ihid*ninp1+iin] = delta * input ;
}


__global__ void device_move_delta (
   int nhid      // Number of neurons in the layer just processed
   )
{
   int icase, ihid ;

   ihid = blockIdx.x * blockDim.x + threadIdx.x ;

   if (ihid >= nhid)
      return ;

   icase = blockIdx.y ;

   d_this_delta[icase*nhid+ihid] = d_prior_delta[icase*nhid+ihid] ;
}

int cuda_subsequent_hidden_gradient (
   int nc ,           // Number of cases in batch
   int ilayer ,       // Hidden layer being processed
   int nhid_this ,    // Number of hidden neurons in this layer
   int nhid_prior ,   // And in prior layer
   int last_hidden    // Is this the last hidden layer?
   )
{
   int warpsize, threads_per_block ;
   char msg[256] ;
   dim3 block_launch ;
   cudaError_t error_id ;

   warpsize = deviceProp.warpSize ;      // Threads per warp, likely 32 well into the future

   threads_per_block = (nhid_prior + 1 + warpsize - 1) / warpsize * warpsize ;
   if (threads_per_block > 4 * warpsize)
      threads_per_block = 4 * warpsize ;

   block_launch.x = (nhid_prior + 1 + threads_per_block - 1) / threads_per_block ; // Include bias
   block_launch.y = nc ;
   block_launch.z = nhid_this ;

   device_subsequent_hidden_gradient <<< block_launch , threads_per_block >>> ( nc , ilayer , last_hidden ) ;   
   cudaDeviceSynchronize() ;
   error_id = cudaGetLastError () ;
   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_subsequent_hidden_gradient launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      MEMTEXT ( msg ) ;
      return 1 ;
      }

/*
   Move deltas from prior to current to prepare for next layer back
*/

   warpsize = deviceProp.warpSize ;      // Threads per warp, likely 32 well into the future

   threads_per_block = (nhid_this + warpsize - 1) / warpsize * warpsize ;
   if (threads_per_block > 4 * warpsize)
      threads_per_block = 4 * warpsize ;

   block_launch.x = (nhid_this + threads_per_block - 1) / threads_per_block ;
   block_launch.y = nc ;
   block_launch.z = 1 ;

   device_move_delta <<< block_launch , threads_per_block >>> ( nhid_this ) ;   
   cudaDeviceSynchronize() ;
   error_id = cudaGetLastError () ;
   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_move_delta launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      MEMTEXT ( msg ) ;
      return 1 ;
      }

   return 0 ;
}


/*
--------------------------------------------------------------------------------

   softmax - Do SoftMax modification of outputs for a batch

--------------------------------------------------------------------------------
*/

__global__ void device_softmax (
   int istart ,       // First case in this batch
   int istop          // One past last case
   )
{
   int icase, iout ;
   double *outptr, sum ;

   icase = blockIdx.x * blockDim.x + threadIdx.x ;

   if (icase >= istop - istart)
      return ;

   outptr = d_output + (icase + istart) * d_ntarg ;  // Output vector for this case
   sum = 0.0 ;

   for (iout=0 ; iout<d_ntarg ; iout++) {
      if (outptr[iout] < 300.0)
         outptr[iout] = __expf ( outptr[iout] ) ;
      else
         outptr[iout] = __expf ( 300.0 ) ;
      sum += outptr[iout] ;
      }

   for (iout=0 ; iout<d_ntarg ; iout++)
      outptr[iout] /= sum ;
}

int cuda_softmax (
   int istart ,       // First case in this batch
   int istop          // One past last case
   )
{
   int n, warpsize, blocks_per_grid, threads_per_block ;
   char msg[256] ;
   cudaError_t error_id ;

   warpsize = deviceProp.warpSize ;      // Threads per warp, likely 32 well into the future

   n = istop - istart ;   // Number of elements

   threads_per_block = (n + warpsize - 1) / warpsize * warpsize ;
   if (threads_per_block > 4 * warpsize)
      threads_per_block = 4 * warpsize ;

   blocks_per_grid = (n + threads_per_block - 1) / threads_per_block ;

   device_softmax <<< blocks_per_grid , threads_per_block >>> ( istart , istop ) ;   
   cudaDeviceSynchronize() ;
   error_id = cudaGetLastError () ;
   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_softmax launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      MEMTEXT ( msg ) ;
      return 1 ;
      }

   return 0 ;
}


/*
--------------------------------------------------------------------------------

   fetch_gradient - Retrieve sum across batch of complete gradient

--------------------------------------------------------------------------------
*/

__global__ void device_fetch_gradient (
   int nc          // Number of cases in batch
   )
{
   int index, icase ;
   float *gptr ;
   double sum ;

   index = blockIdx.x * blockDim.x + threadIdx.x ;

   if (index >= d_gradlen)
      return ;

   sum = 0.0 ;
   gptr = d_gradient + index ;
   for (icase=0 ; icase<nc ; icase++)   // For all cases in this batch
      sum += gptr[icase*d_gradlen] ;
   *gptr = sum ;
}

int cuda_fetch_gradient (
   int nc ,        // Number of cases in batch
   double *grad    // Gradient sum output here
   )
{
   int i, warpsize, blocks_per_grid, threads_per_block ;
   char msg[256] ;
   cudaError_t error_id ;

   warpsize = deviceProp.warpSize ;      // Threads per warp, likely 32 well into the future

   threads_per_block = (h_gradlen + warpsize - 1) / warpsize * warpsize ;
   if (threads_per_block > 4 * warpsize)
      threads_per_block = 4 * warpsize ;

   blocks_per_grid = (h_gradlen + threads_per_block - 1) / threads_per_block ;

   device_fetch_gradient <<< blocks_per_grid , threads_per_block >>> ( nc ) ;   
   cudaDeviceSynchronize() ;
   error_id = cudaGetLastError () ;
   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_fetch_gradient launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      MEMTEXT ( msg ) ;
      return 1 ;
      }

   error_id = cudaMemcpy ( fdata , h_gradient , h_gradlen * sizeof(float) , cudaMemcpyDeviceToHost ) ;
   for (i=0 ; i<h_gradlen ; i++)
      grad[i] += fdata[i] ;
   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_fetch_gradient copy error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      MEMTEXT ( msg ) ;
      return 1 ;
      }

   return 0 ;
}


/*
------------------------------------------------------------------------------------------------

   cuda_mse - Given output activations and targets, compute mse
              This would be called after the entire training set is processed,
              not in batches.
             
------------------------------------------------------------------------------------------------
*/

__global__ void device_mse ()
{
   __shared__ double partial_mse[REDUC_THREADS] ;
   int i, index ;
   unsigned int n ;
   double diff, sum_mse ;

   index = threadIdx.x ;
   n = d_ncases * d_ntarg ;

   sum_mse = 0.0 ;   
   for (i=blockIdx.x*blockDim.x+index ; i<n ; i+=blockDim.x*gridDim.x) {
      diff = d_output[i] - d_targets[i] ;
      sum_mse += diff * diff ;
      }

   partial_mse[index] = sum_mse ;
   __syncthreads() ;

   for (i=blockDim.x>>1 ; i ; i>>=1) {
      if (index < i)
         partial_mse[index] += partial_mse[index+i] ;
      __syncthreads() ;
      }

   if (index == 0)
      d_mse_out[blockIdx.x] = partial_mse[0] ;
}


int cuda_mse (
   int n ,           // Number of values; ncases * ntarg
   double *mse       // Computed mse criterion
   )
{
   int i, blocks_per_grid ;
   double sum ;
   char msg[256] ;
   cudaError_t error_id ;

   blocks_per_grid = (n + REDUC_THREADS - 1) / REDUC_THREADS ;
   if (blocks_per_grid > REDUC_BLOCKS)
      blocks_per_grid = REDUC_BLOCKS ;

   device_mse <<< blocks_per_grid , REDUC_THREADS >>> () ;   
   cudaDeviceSynchronize() ;

   error_id = cudaGetLastError () ;
   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_mse launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      MEMTEXT ( msg ) ;
      return 1 ;
      }

   error_id = cudaMemcpy ( reduc_fdata , h_mse_out , blocks_per_grid * sizeof(float) , cudaMemcpyDeviceToHost ) ;
   sum = 0.0 ;
   for (i=0 ; i<blocks_per_grid ; i++)
      sum += reduc_fdata[i] ;
   *mse = sum / n ;

   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_mse Memcpy error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      MEMTEXT ( msg ) ;
      return 1 ;
      }

   return 0 ;
}


/*
------------------------------------------------------------------------------------------------

   cuda_ll - Given output activations and targets, compute log likelihood
             This would be called after the entire training set is processed,
             not in batches.
             
------------------------------------------------------------------------------------------------
*/

__global__ void device_ll ()
{
   __shared__ double partial_ll[REDUC_THREADS] ;
   int i, n, ntarg, index ;
   double sum_ll ;

   index = threadIdx.x ;
   n = d_ncases ;
   ntarg = d_ntarg ;

   sum_ll = 0.0 ;   
   for (i=blockIdx.x*blockDim.x+index ; i<n ; i+=blockDim.x*gridDim.x)
      sum_ll -= log ( d_output[i*ntarg+d_class[i]] + 1.e-30 ) ;

   partial_ll[index] = sum_ll ;
   __syncthreads() ;

   for (i=blockDim.x>>1 ; i ; i>>=1) {
      if (index < i)
         partial_ll[index] += partial_ll[index+i] ;
      __syncthreads() ;
      }

   if (index == 0)
      d_mse_out[blockIdx.x] = partial_ll[0] ;
}


int cuda_ll (
   int n ,          // Number of values; ncases
   double *ll       // Computed dot product
   )
{
   int i, blocks_per_grid ;
   double sum ;
   char msg[256] ;
   cudaError_t error_id ;

   blocks_per_grid = (n + REDUC_THREADS - 1) / REDUC_THREADS ;
   if (blocks_per_grid > REDUC_BLOCKS)
      blocks_per_grid = REDUC_BLOCKS ;

   device_ll <<< blocks_per_grid , REDUC_THREADS >>> () ;   
   cudaDeviceSynchronize() ;

   error_id = cudaGetLastError () ;
   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_ll launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      MEMTEXT ( msg ) ;
      return 1 ;
      }

   error_id = cudaMemcpy ( reduc_fdata , h_mse_out , blocks_per_grid * sizeof(float) , cudaMemcpyDeviceToHost ) ;
   sum = 0.0 ;
   for (i=0 ; i<blocks_per_grid ; i++)
      sum += reduc_fdata[i] ;
   *ll = sum / n ;

   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_ll Memcpy error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      MEMTEXT ( msg ) ;
      return 1 ;
      }

   return 0 ;
}


/*
--------------------------------------------------------------------------------

   MLFN_CUDA_CLEANUP - Cleanup after CUDA MLFN processing

--------------------------------------------------------------------------------
*/

void mlfn_cuda_cleanup ( int classifier , int n_layers )
{
   int ilayer ;
   double sum ;
   char msg[256] ;

   MEMTEXT ( "CUDA mlfn_cuda_cleanup starting" ) ;

   if (h_trn_data != NULL) {
      cudaFree ( h_trn_data ) ;
      h_trn_data = NULL ;
      }

   if (h_targets != NULL) {
      cudaFree ( h_targets ) ;
      h_targets = NULL ;
      }

   if (h_class != NULL) {
      cudaFree ( h_class ) ;
      h_class = NULL ;
      }

   if (h_nhid != NULL) {
      cudaFree ( h_nhid ) ;
      h_nhid = NULL ;
      }

   if (hidden_weights != NULL) {
      cudaFree ( hidden_weights ) ;
      hidden_weights = NULL ;
      }

   if (h_whid != NULL) {
      cudaFree ( h_whid ) ;
      h_whid = NULL ;
      }

   if (h_wout != NULL) {
      cudaFree ( h_wout ) ;
      h_wout = NULL ;
      }

   if (activations != NULL) {
      cudaFree ( activations ) ;
      activations = NULL ;
      }

   if (h_act != NULL) {
      cudaFree ( h_act ) ;
      h_act = NULL ;
      }

   if (h_output != NULL) {
      cudaFree ( h_output ) ;
      h_output = NULL ;
      }

   if (h_this_delta != NULL) {
      cudaFree ( h_this_delta ) ;
      h_this_delta = NULL ;
      }

   if (h_prior_delta != NULL) {
      cudaFree ( h_prior_delta ) ;
      h_prior_delta = NULL ;
      }

   if (h_gradient != NULL) {
      cudaFree ( h_gradient ) ;
      h_gradient = NULL ;
      }

   if (h_grad_ptr != NULL) {
      cudaFree ( h_grad_ptr ) ;
      h_grad_ptr = NULL ;
      }

   if (h_mse_out != NULL) {
      cudaFree ( h_mse_out ) ;
      h_mse_out = NULL ;
      }

   if (fdata != NULL) {
      FREE ( fdata ) ;
      fdata = NULL ;
      }

   if (reduc_fdata != NULL) {
      FREE ( reduc_fdata ) ;
      reduc_fdata = NULL ;
      }

   total_memory = 0.0 ;

   cudaDeviceReset () ;


/*
   Print CUDA timers
*/

   sum = 0.0 ;
   for (ilayer=0 ; ilayer<MAX_LAYERS ; ilayer++) {
      sum += CudaTimers.mlfn_hidden[ilayer] ;
      sum += CudaTimers.mlfn_subgrad[ilayer] ;
      }

   sum += CudaTimers.mlfn_weights + CudaTimers.mlfn_outact + CudaTimers.mlfn_softmax +
          CudaTimers.mlfn_ll + CudaTimers.mlfn_mse + CudaTimers.mlfn_wpen + CudaTimers.mlfn_outdelta +
          CudaTimers.mlfn_outgrad + CudaTimers.mlfn_firstgrad + CudaTimers.mlfn_fetchgrad ;

   cudalog ( "" ) ;
   cudalog ( "" ) ;
   cudalog ( "MLFN CUDA times in seconds: total, (percent), per launch" ) ;

   sprintf ( msg, "  Send weights =   %8.3lf   (%5.1lf percent) %10.6lf per launch",
             0.001 * CudaTimers.mlfn_weights,
             100.0 * CudaTimers.mlfn_weights / sum,
             0.001 * CudaTimers.mlfn_weights / CudaTimers.mlfn_ncalls_weights ) ;
   cudalog ( msg ) ;

   for (ilayer=0 ; ilayer<n_layers-1 ; ilayer++) {
      sprintf ( msg, "  Hid %2d act =     %8.3lf   (%5.1lf percent) %10.6lf per launch", ilayer+1,
                0.001 * CudaTimers.mlfn_hidden[ilayer],
                100.0 * CudaTimers.mlfn_hidden[ilayer] / sum,
                0.001 * CudaTimers.mlfn_hidden[ilayer] / CudaTimers.mlfn_ncalls_hidden[ilayer] ) ;
      cudalog ( msg ) ;
      }

   sprintf ( msg, "  Output act =     %8.3lf   (%5.1lf percent) %10.6lf per launch",
             0.001 * CudaTimers.mlfn_outact,
             100.0 * CudaTimers.mlfn_outact / sum,
             0.001 * CudaTimers.mlfn_outact / CudaTimers.mlfn_ncalls_outact ) ;
   cudalog ( msg ) ;

   if (classifier) {
      sprintf ( msg, "  SoftMax =        %8.3lf   (%5.1lf percent) %10.6lf per launch",
                0.001 * CudaTimers.mlfn_softmax,
                100.0 * CudaTimers.mlfn_softmax / sum,
                0.001 * CudaTimers.mlfn_softmax / CudaTimers.mlfn_ncalls_softmax ) ;
      cudalog ( msg ) ;
      }

   if (CudaTimers.mlfn_ncalls_outdelta) {
      sprintf ( msg, "  Output delta =   %8.3lf   (%5.1lf percent) %10.6lf per launch",
                0.001 * CudaTimers.mlfn_outdelta,
                100.0 * CudaTimers.mlfn_outdelta / sum,
                0.001 * CudaTimers.mlfn_outdelta / CudaTimers.mlfn_ncalls_outdelta ) ;
      cudalog ( msg ) ;
      }

   if (CudaTimers.mlfn_ncalls_outgrad) {
      sprintf ( msg, "  Output grad =    %8.3lf   (%5.1lf percent) %10.6lf per launch",
                0.001 * CudaTimers.mlfn_outgrad,
                100.0 * CudaTimers.mlfn_outgrad / sum,
                0.001 * CudaTimers.mlfn_outgrad / CudaTimers.mlfn_ncalls_outgrad ) ;
      cudalog ( msg ) ;
      }

   for (ilayer=n_layers-2 ; ilayer>0 ; ilayer--) {
      if (CudaTimers.mlfn_ncalls_subgrad[ilayer-1]) {
         sprintf ( msg, "  Hid %2d grad =    %8.3lf   (%5.1lf percent) %10.6lf per launch", ilayer+1,
                   0.001 * CudaTimers.mlfn_subgrad[ilayer-1],
                   100.0 * CudaTimers.mlfn_subgrad[ilayer-1] / sum,
                   0.001 * CudaTimers.mlfn_subgrad[ilayer-1] / CudaTimers.mlfn_ncalls_subgrad[ilayer-1] ) ;
         cudalog ( msg ) ;
         }
      }

   if (CudaTimers.mlfn_ncalls_firstgrad) {
      sprintf ( msg, "  First grad =     %8.3lf   (%5.1lf percent) %10.6lf per launch",
                0.001 * CudaTimers.mlfn_firstgrad,
                100.0 * CudaTimers.mlfn_firstgrad / sum,
                0.001 * CudaTimers.mlfn_firstgrad / CudaTimers.mlfn_ncalls_firstgrad ) ;
      cudalog ( msg ) ;
      }

   if (CudaTimers.mlfn_ncalls_fetchgrad) {
      sprintf ( msg, "  Fetch grad =     %8.3lf   (%5.1lf percent) %10.6lf per launch",
                0.001 * CudaTimers.mlfn_fetchgrad,
                100.0 * CudaTimers.mlfn_fetchgrad / sum,
                0.001 * CudaTimers.mlfn_fetchgrad / CudaTimers.mlfn_ncalls_fetchgrad ) ;
      cudalog ( msg ) ;
      }

   if (classifier) {
      sprintf ( msg, "  Log likelihood = %8.3lf   (%5.1lf percent) %10.6lf per launch",
                0.001 * CudaTimers.mlfn_ll,
                100.0 * CudaTimers.mlfn_ll / sum,
                0.001 * CudaTimers.mlfn_ll / CudaTimers.mlfn_ncalls_ll ) ;
      cudalog ( msg ) ;
      }
   else {
      sprintf ( msg, "  MSE =            %8.3lf   (%5.1lf percent) %10.6lf per launch",
                0.001 * CudaTimers.mlfn_mse,
                100.0 * CudaTimers.mlfn_mse / sum,
                0.001 * CudaTimers.mlfn_mse / CudaTimers.mlfn_ncalls_mse ) ;
      cudalog ( msg ) ;
      }

   sprintf ( msg, "  Weight penalty = %8.3lf   (%5.1lf percent) %10.6lf per launch",
             0.001 * CudaTimers.mlfn_wpen,
             100.0 * CudaTimers.mlfn_wpen / sum,
             0.001 * CudaTimers.mlfn_wpen / CudaTimers.mlfn_ncalls_wpen ) ;
   cudalog ( msg ) ;

   MEMTEXT ( "CUDA mlfn_cuda_cleanup ending" ) ;
}
