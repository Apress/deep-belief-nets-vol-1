/******************************************************************************/
/*                                                                            */
/*  MLFN_CUDA - MLFN routines modified for CUDA processing                    */
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

#include "deep.rh"
#include "const.h"
#include "classes.h"
#include "extern.h"
#include "funcdefs.h"


/*
--------------------------------------------------------------------------------

   trial_error_cuda - Compute the mean square error for the entire training set

--------------------------------------------------------------------------------
*/

double Model::trial_error_cuda (
   int nc ,             // Number of cases
   double *input ,      // Input matrix, nc by Model::n_model_inputs
   double *target       // Target matrix, nc by ntarg
   )
{
   int i, ilayer, ineuron, ivar, ret_val, ibatch, n_in_batch, n_subsets, max_batch, istart, istop, n_done ;
   int n_prior, gradlen, nin_this_layer, timer ;
   double mse, *wptr ;
   char msg[256] ;

   assert ( n_all >= 2 ) ;  // Use CUDA only if at least one hidden layer

/*
   In order to prevent integer overflow in allocating memory for the gradient
   we compute the minimum number of batches needed to get each batch small enough.
*/

   gradlen = 0 ;
   n_prior = n_model_inputs ;
   for (i=0 ; i<n_all-1 ; i++) {
      gradlen += nhid_all[i] * (n_prior + 1) ;
      n_prior = nhid_all[i] ;
      }
   gradlen += ntarg * (n_prior + 1) ;
   assert ( gradlen == n_all_weights ) ;

   max_batch = MAXPOSNUM / (gradlen * sizeof(float)) ;  // Memory allocation size
   if (max_batch > 65535)                               // Grid dimension
      max_batch = 65535 ;
   n_subsets = nc / max_batch + 1 ;

   if (n_subsets < TrainParams.n_subsets)
      n_subsets = TrainParams.n_subsets ;

   else if (n_subsets > TrainParams.n_subsets  &&  ! mlfn_cuda_initialized) {
      sprintf ( msg, "MLFN CUDA increased n_subsets to %d", n_subsets ) ;
      MEMTEXT ( msg ) ;
      cudalog ( msg ) ;
      audit ( "" ) ;
      sprintf ( msg, "NOTE... Number of subsets had to be increased to %d", n_subsets ) ;
      audit ( msg ) ;
      }


/*
   Initialize CUDA device if not yet done for this session

   Programming WARNING... If ANY of the parameters in the call to mlfn_cuda_init change,
                          then mlfn_cuda_cleanup MUST be called and init redone!
*/

   if (! mlfn_cuda_initialized) {

      n_done = 0 ;         // Must find max batch size for cuda init
      for (ibatch=0 ; ibatch<n_subsets ; ibatch++) {
         n_in_batch = (nc - n_done) / (n_subsets - ibatch) ;   // Cases left to do / batches left to do
         if (ibatch == 0  ||  n_in_batch > max_batch)
            max_batch = n_in_batch ;
         n_done += n_in_batch ;
         }

      assert ( max_batch * sizeof(float) <= MAXPOSNUM / gradlen ) ;

      ret_val =  mlfn_cuda_init ( classifier , class_ids , nc , n_model_inputs , max_neurons , input ,
                                  ntarg , target , max_batch , n_all , nhid_all , msg ) ;

      if (ret_val == ERROR_INSUFFICIENT_MEMORY) {
         audit ( "" ) ;
         audit ( "ERROR... Host computer has insufficient memory" ) ;
         }
      if (ret_val == ERROR_CUDA_MEMORY) {
         audit ( "" ) ;
         audit ( "ERROR... CUDA device has insufficient memory" ) ;
         }
      if (ret_val == ERROR_CUDA_ERROR) {
         audit ( "" ) ;
         audit ( "ERROR... CUDA device had unexpected serious error" ) ;
         }
      if (ret_val) {
         audit ( "" ) ;
         audit ( "ERROR... Unrecoverable serious error... aborting" ) ;
         return -1.e40 ;
         }

      mlfn_cuda_initialized = 1 ;
      }


   if (cuda_weights_changed) {
      ++CudaTimers.mlfn_ncalls_weights ;
      timer = timeGetTime() ;
      ret_val = cuda_weights_to_device ( n_model_inputs , ntarg ,
                  n_all , nhid_all , weights_opt , final_layer_weights ) ;
      if (ret_val) {
         audit ( "" ) ;
         audit ( "ERROR - Serious CUDA error" ) ;
         return -1.e40 ;
         }
      CudaTimers.mlfn_weights += timeGetTime() - timer ;
      cuda_weights_changed = 0 ;
      }

   istart = 0 ;         // Batch start = training data start
   n_done = 0 ;         // Number of training cases done in this epoch so far

   for (ibatch=0 ; ibatch<n_subsets ; ibatch++) {
      n_in_batch = (nc - n_done) / (n_subsets - ibatch) ;   // Cases left to do / batches left to do
      istop = istart + n_in_batch ;                         // Stop just before this index

      for (ilayer=0 ; ilayer<n_all-1 ; ilayer++) {
         ++CudaTimers.mlfn_ncalls_hidden[ilayer] ;
         timer = timeGetTime() ;
         ret_val = cuda_hidden_activation ( istart , istop , nhid_all[ilayer] , ilayer ) ;
         if (ret_val) {
            audit ( "" ) ;
            sprintf ( msg, "ERROR - Serious CUDA error (1 - %d) in MLFN_CUDA.CPP trial_error_cuda", ilayer ) ;
            audit ( msg ) ;
            return -1.e40 ;
            }
         CudaTimers.mlfn_hidden[ilayer] += timeGetTime() - timer ;
         }

      ++CudaTimers.mlfn_ncalls_outact ;
      timer = timeGetTime() ;
      ret_val = cuda_output_activation ( istart , istop , nhid_all[n_all-2] , ntarg , n_all-2 ) ;
      if (ret_val) {
         audit ( "" ) ;
         audit ( "ERROR - Serious CUDA error (2) in MLFN_CUDA.CPP trial_error_cuda" ) ;
         return -1.e40 ;
         }
      CudaTimers.mlfn_outact += timeGetTime() - timer ;

      if (classifier) {
         ++CudaTimers.mlfn_ncalls_softmax ;
         timer = timeGetTime() ;
         ret_val = cuda_softmax ( istart , istop ) ;
         if (ret_val) {
            audit ( "" ) ;
            audit ( "ERROR - Serious CUDA error (3) in MLFN_CUDA.CPP trial_error_cuda" ) ;
            return -1.e40 ;
            }
         CudaTimers.mlfn_softmax += timeGetTime() - timer ;
         }

      n_done += n_in_batch ;
      istart = istop ;
      }  // For all batches

   if (classifier) {
      ++CudaTimers.mlfn_ncalls_ll ;
      timer = timeGetTime() ;
      ret_val = cuda_ll ( nc , &mse ) ;
      CudaTimers.mlfn_ll += timeGetTime() - timer ;
      mse /= ntarg ;
      }
   else {
      ++CudaTimers.mlfn_ncalls_mse ;
      timer = timeGetTime() ;
      ret_val = cuda_mse ( nc * ntarg , &mse ) ;
      CudaTimers.mlfn_mse += timeGetTime() - timer ;
      }

   if (ret_val) {
      audit ( "" ) ;
      audit ( "ERROR - Serious CUDA error (4) in MLFN_CUDA.CPP trial_error_cuda" ) ;
      return -1.e40 ;
      }


/*
   Deal with weight penalty
*/

   ++CudaTimers.mlfn_ncalls_wpen ;
   timer = timeGetTime() ;
   penalty = 0.0 ;
   nin_this_layer = n_model_inputs ;
   for (ilayer=0 ; ilayer<n_all-1 ; ilayer++) {  // Do all hidden layers
      for (ineuron=0 ; ineuron<nhid_all[ilayer] ; ineuron++) {
         wptr = weights_opt[ilayer]+ineuron*(nin_this_layer+1) ;  // Weights for this neuron in this layer
         for (ivar=0 ; ivar<nin_this_layer ; ivar++)              // Do not include bias
            penalty += wptr[ivar] * wptr[ivar] ;
         }
      nin_this_layer = nhid_all[ilayer] ;
      }

   for (ineuron=0 ; ineuron<ntarg ; ineuron++) {
      wptr = final_layer_weights + ineuron * n_final_layer_weights ;
      for (ivar=0 ; ivar<nin_this_layer ; ivar++)
         penalty += wptr[ivar] * wptr[ivar] ;
      }
   CudaTimers.mlfn_wpen += timeGetTime() - timer ;

   penalty *= TrainParams.wpen / n_all_weights ;
   return mse + penalty ;
}


/*
--------------------------------------------------------------------------------

   gradient_cuda - Compute the gradient for the entire training set

--------------------------------------------------------------------------------
*/

double Model::gradient_cuda (
   int nc ,             // Number of cases
   double *input ,      // Input matrix, nc by Model::n_model_inputs
   double *target ,     // Target matrix, nc by ntarg
   double *grad         // Complete gradient
   )
{
   int i, k, n, ilayer, ineuron, ivar, ret_val, ibatch, n_in_batch, n_subsets, istart, istop, n_done, max_batch ;
   int n_prior, gradlen, nin_this_layer, timer ;
   double mse, wpen, *wptr, *gptr ;
   char msg[256] ;

   assert ( n_all >= 2 ) ;  // Use CUDA only if at least one hidden layer

// Setup pointers to gradient for each layer
   gptr = grad ;  // CONJGRAD.CPP allocated this

   k = 0 ;
   for (ilayer=0 ; ilayer<n_all ; ilayer++) {
      grad_ptr[ilayer] = gptr ;

      if (ilayer == 0  &&  n_all == 1) {             // Direct input to output?
         n = ntarg * (n_model_inputs+1) ;            // This many inputs to each neuron in this layer
         gptr += n ;                                 // Not needed, but it illustrates the process
         k += n ;   // Can remove this when final assert is assured
         }

      else if (ilayer == 0) {                        // First hidden layer?
         n = nhid_all[ilayer] * (n_model_inputs+1) ; // This many inputs to each neuron in this layer
         gptr += n ;
         k += n ;   // Can remove this when final assert is assured
         }

      else if (ilayer < n_all-1) {                       // Subsequent hidden layer?
         n = nhid_all[ilayer] * (nhid_all[ilayer-1]+1) ; // This many inputs to each neuron in this layer
         gptr += n ;
         k += n ;   // Can remove this when final assert is assured
         }

      else {
         assert ( (nhid_all[ilayer-1]+1) == n_final_layer_weights ) ;
         n = ntarg * (nhid_all[ilayer-1]+1) ; // This many inputs to each neuron in this layer
         k += n ;   // Can remove this when final assert is assured
         }
      } // For all layers, including output

   assert ( k == n_all_weights ) ;

/*
   In order to prevent integer overflow in allocating memory for the gradient
   we compute the minimum number of batches needed to get each batch small enough.
*/

   gradlen = 0 ;
   n_prior = n_model_inputs ;
   for (i=0 ; i<n_all-1 ; i++) {   // Hidden layers
      gradlen += nhid_all[i] * (n_prior + 1) ;
      n_prior = nhid_all[i] ;
      }
   gradlen += ntarg * (n_prior + 1) ;    // Output layer
   assert ( gradlen == n_all_weights ) ;

   max_batch = MAXPOSNUM / (gradlen * sizeof(float)) ;  // Memory allocation size
   if (max_batch > 65535)                               // Grid dimension
      max_batch = 65535 ;
   n_subsets = nc / max_batch + 1 ;

   if (n_subsets < TrainParams.n_subsets)
      n_subsets = TrainParams.n_subsets ;

   else if (n_subsets > TrainParams.n_subsets  &&  ! mlfn_cuda_initialized) {
      sprintf ( msg, "WARNING... MLFN CUDA increased n_subsets to %d", n_subsets ) ;
      MEMTEXT ( msg ) ;
      cudalog ( msg ) ;
      audit ( "" ) ;
      sprintf ( msg, "NOTE... Number of batches had to be increased to %d", n_subsets ) ;
      audit ( msg ) ;
      }


/*
   Initialize CUDA device if not yet done for this session

   Programming WARNING... If ANY of the parameters in the call to mlfn_cuda_init change,
                          then mlfn_cuda_cleanup MUST be called and init redone!
*/

   if (! mlfn_cuda_initialized) {

      n_done = 0 ;         // Must find max batch size for cuda init
      for (ibatch=0 ; ibatch<n_subsets ; ibatch++) {
         n_in_batch = (nc - n_done) / (n_subsets - ibatch) ;   // Cases left to do / batches left to do
         if (ibatch == 0  ||  n_in_batch > max_batch)
            max_batch = n_in_batch ;
         n_done += n_in_batch ;
         }

      assert ( max_batch * sizeof(float) <= MAXPOSNUM / gradlen ) ;

      ret_val =  mlfn_cuda_init ( classifier , class_ids , nc , n_model_inputs , max_neurons , input ,
                                  ntarg , target , max_batch , n_all , nhid_all , msg ) ;

      if (ret_val == ERROR_INSUFFICIENT_MEMORY) {
         audit ( "" ) ;
         audit ( "ERROR... Host computer has insufficient memory" ) ;
         }
      if (ret_val == ERROR_CUDA_MEMORY) {
         audit ( "" ) ;
         audit ( "ERROR... CUDA device has insufficient memory" ) ;
         }
      if (ret_val == ERROR_CUDA_ERROR) {
         audit ( "" ) ;
         audit ( "ERROR... CUDA device had unexpected serious error" ) ;
         }
      if (ret_val) {
         audit ( "" ) ;
         audit ( "ERROR... Unrecoverable serious error... aborting" ) ;
         return -1.e40 ;
         }

      mlfn_cuda_initialized = 1 ;
      }


   if (cuda_weights_changed) {
      ++CudaTimers.mlfn_ncalls_weights ;
      timer = timeGetTime() ;
      ret_val = cuda_weights_to_device ( n_model_inputs , ntarg ,
                  n_all , nhid_all , weights_opt , final_layer_weights ) ;
      if (ret_val) {
         audit ( "" ) ;
         audit ( "ERROR - Serious CUDA error" ) ;
         return -1.e40 ;
         }
      CudaTimers.mlfn_weights += timeGetTime() - timer ;
      cuda_weights_changed = 0 ;
      }

/*
   Gradient computation starts here
*/

   for (i=0 ; i<n_all_weights ; i++)
      grad[i] = 0.0 ;

   istart = 0 ;         // Batch start = training data start
   n_done = 0 ;         // Number of training cases done in this epoch so far

   for (ibatch=0 ; ibatch<n_subsets ; ibatch++) {
      n_in_batch = (nc - n_done) / (n_subsets - ibatch) ;   // Cases left to do / batches left to do
      istop = istart + n_in_batch ;                         // Stop just before this index

/*
   Forward pass
*/

      for (ilayer=0 ; ilayer<n_all-1 ; ilayer++) {
         ++CudaTimers.mlfn_ncalls_hidden[ilayer] ;
         timer = timeGetTime() ;
         ret_val = cuda_hidden_activation ( istart , istop , nhid_all[ilayer] , ilayer ) ;
         if (ret_val) {
            audit ( "" ) ;
            sprintf ( msg, "ERROR - Serious CUDA error (1 - %d) in MLFN_CUDA.CPP gradient_cuda", ilayer ) ;
            audit ( msg ) ;
            return -1.e40 ;
            }
         CudaTimers.mlfn_hidden[ilayer] += timeGetTime() - timer ;
         }

      ++CudaTimers.mlfn_ncalls_outact ;
      timer = timeGetTime() ;
      ret_val = cuda_output_activation ( istart , istop , nhid_all[n_all-2] , ntarg , n_all-2 ) ;
      if (ret_val) {
         audit ( "" ) ;
         audit ( "ERROR - Serious CUDA error (2) in MLFN_CUDA.CPP gradient_cuda" ) ;
         return -1.e40 ;
         }
      CudaTimers.mlfn_outact += timeGetTime() - timer ;

      if (classifier) {
         ++CudaTimers.mlfn_ncalls_softmax ;
         timer = timeGetTime() ;
         ret_val = cuda_softmax ( istart , istop ) ;
         if (ret_val) {
            audit ( "" ) ;
            audit ( "ERROR - Serious CUDA error (3) in MLFN_CUDA.CPP trial_error_cuda" ) ;
            return -1.e40 ;
            }
         CudaTimers.mlfn_softmax += timeGetTime() - timer ;
         }

/*
   Backward pass
*/

      ++CudaTimers.mlfn_ncalls_outdelta ;
      timer = timeGetTime() ;
      ret_val = cuda_output_delta ( istart , istop , classifier , ntarg ) ;
      if (ret_val) {
         audit ( "" ) ;
         audit ( "ERROR - Serious CUDA error (4) in MLFN_CUDA.CPP gradient_cuda" ) ;
         return -1.e40 ;
         }
      CudaTimers.mlfn_outdelta += timeGetTime() - timer ;

      ++CudaTimers.mlfn_ncalls_outgrad ;
      timer = timeGetTime() ;
      ret_val = cuda_output_gradient ( n_in_batch , nhid_all[n_all-2] , n_all-2 , ntarg ) ;
      if (ret_val) {
         audit ( "" ) ;
         audit ( "ERROR - Serious CUDA error (5) in MLFN_CUDA.CPP gradient_cuda" ) ;
         return -1.e40 ;
         }
      CudaTimers.mlfn_outgrad += timeGetTime() - timer ;

      for (ilayer=n_all-2 ; ilayer>0 ; ilayer--) {
         ++CudaTimers.mlfn_ncalls_subgrad[ilayer-1] ;
         timer = timeGetTime() ;
         ret_val = cuda_subsequent_hidden_gradient ( n_in_batch , ilayer ,
                              nhid_all[ilayer] , nhid_all[ilayer-1] , ilayer==n_all-2 ) ;
         if (ret_val) {
            audit ( "" ) ;
            sprintf ( msg, "ERROR - Serious CUDA error (6 - %d) in MLFN_CUDA.CPP gradient_cuda", ilayer ) ;
            audit ( msg ) ;
            return -1.e40 ;
            }
         CudaTimers.mlfn_subgrad[ilayer-1] += timeGetTime() - timer ;
         }

      ++CudaTimers.mlfn_ncalls_firstgrad ;
      timer = timeGetTime() ;
      ret_val = cuda_first_hidden_gradient ( istart , istop , n_model_inputs , nhid_all[0] , n_all==2 ) ;
      if (ret_val) {
         audit ( "" ) ;
         audit ( "ERROR - Serious CUDA error (7) in MLFN_CUDA.CPP gradient_cuda" ) ;
         return -1.e40 ;
         }
      CudaTimers.mlfn_firstgrad += timeGetTime() - timer ;

      ++CudaTimers.mlfn_ncalls_fetchgrad ;
      timer = timeGetTime() ;
      ret_val = cuda_fetch_gradient ( n_in_batch , grad ) ;
      if (ret_val) {
         audit ( "" ) ;
         audit ( "ERROR - Serious CUDA error (8) in MLFN_CUDA.CPP gradient_cuda" ) ;
         return -1.e40 ;
         }
      CudaTimers.mlfn_fetchgrad += timeGetTime() - timer ;

      n_done += n_in_batch ;
      istart = istop ;
      }  // For all batches

   for (i=0 ; i<n_all_weights ; i++)
      grad[i] /= nc * ntarg ;


   if (classifier) {
      ++CudaTimers.mlfn_ncalls_ll ;
      timer = timeGetTime() ;
      ret_val = cuda_ll ( nc , &mse ) ;
      CudaTimers.mlfn_ll += timeGetTime() - timer ;
      mse /= ntarg ;  // cuda_ll() divided by n but not ntarg
      }
   else {
      ++CudaTimers.mlfn_ncalls_mse ;
      timer = timeGetTime() ;
      ret_val = cuda_mse ( nc * ntarg , &mse ) ;
      CudaTimers.mlfn_mse += timeGetTime() - timer ;
      }

   if (ret_val) {
      audit ( "" ) ;
      audit ( "ERROR - Serious CUDA error (9) in MLFN_CUDA.CPP gradient_cuda" ) ;
      return -1.e40 ;
      }


/*
   Deal with weight penalty
   First block of code does hidden layers, second does output layer
*/

   wpen = TrainParams.wpen / n_all_weights ;
   penalty = 0.0 ;
   nin_this_layer = n_model_inputs ;

   ++CudaTimers.mlfn_ncalls_wpen ;
   timer = timeGetTime() ;
   for (ilayer=0 ; ilayer<n_all-1 ; ilayer++) {  // Do all hidden layers
      for (ineuron=0 ; ineuron<nhid_all[ilayer] ; ineuron++) {
         wptr = weights_opt[ilayer] + ineuron*(nin_this_layer+1) ;  // Weights for this neuron in this layer
         gptr = grad_ptr[ilayer] + ineuron*(nin_this_layer+1) ;     // Ditto grad
         for (ivar=0 ; ivar<nin_this_layer ; ivar++) {              // Do not include bias
            penalty += wptr[ivar] * wptr[ivar] ;
            gptr[ivar] -= 2.0 * wpen * wptr[ivar] ;
            }
         }
      nin_this_layer = nhid_all[ilayer] ;
      }

   for (ineuron=0 ; ineuron<ntarg ; ineuron++) {
      wptr = final_layer_weights + ineuron * n_final_layer_weights ;
      gptr = grad_ptr[n_all-1] + ineuron * n_final_layer_weights ;
      for (ivar=0 ; ivar<nin_this_layer ; ivar++) {                 // Do not include bias
         penalty += wptr[ivar] * wptr[ivar] ;
         gptr[ivar] -= 2.0 * wpen * wptr[ivar] ;
         }
      }
   CudaTimers.mlfn_wpen += timeGetTime() - timer ;

   penalty *= wpen ;
   return mse + penalty ;
}