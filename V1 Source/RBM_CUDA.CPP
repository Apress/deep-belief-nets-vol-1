/******************************************************************************/
/*                                                                            */
/*  RBM_CUDA - RBM training via CUDA                                          */
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

#define IA 16807
#define IM 2147483647
#define AM (1.0 / IM)
#define IQ 127773
#define IR 2836

#define DEBUG 0


/*
--------------------------------------------------------------------------------

   rbm_cuda_wt_init() - Called from GREEDY.CPP to find good initial weights

--------------------------------------------------------------------------------
*/

double rbm_cuda_wt_init (
   int nc ,                // Number of cases
   int n_inputs ,          // Number of inputs
   int ncols ,             // Number of columns in data
   double *data ,          // Nc rows by ncols columns of 0-1 input data in first n_inputs cols
   int nhid ,              // Number of hidden neurons
   int n_rand ,            // Number of random weight sets to try
   int n_batches ,         // Number of batches per weight trial
   int *shuffle_index ,    // Work vector nc long
   double *w ,             // Computed weight matrix, nhid sets of n_inputs weights; max_threads sets
   double *in_bias ,       // Computed input bias vector; max_threads sets
   double *hid_bias ,      // Computed hidden bias vector; max_threads sets
   double *in_bias_best ,  // Work vector n_inputs long
   double *hid_bias_best , // Work vector nhid long
   double *w_best ,        // Work vector n_inputs * nhid long
   double *data_mean ,     // Work vector n_inputs long
   double *err_vec         // Work vector n_inputs long
   )

{
   int irand ;        // Index of random try for best starting weights
   int ivis ;         // Index within visible layer
   int ihid ;         // Index of hidden neuron
   double error ;     // Mean squared error for each epoch; sum of squared diffs between input and P[x=1|hidden layer]
                      // Or cross entropy loss
   double best_err ;  // Best error seen so far

   int i, ret_val ;
   int n_done, icase, ibatch, max_batch, n_in_batch, istart, istop ;
   double sum, wt, *dptr, diff ;
   char msg[4096] ;
#if DEBUG
   char msg2[256] ;
#endif

   user_pressed_escape () ;
   escape_key_pressed = 0 ;  // Allow subsequent operations

/*
   Find the mean of the data for each input.
   This is used to initialize visible bias terms to reasonable values.
*/

   for (ivis=0 ; ivis<n_inputs ; ivis++)
      data_mean[ivis] = 0.0 ;

   for (i=0 ; i<nc ; i++) {            // Pass through all cases, cumulating mean vector
      dptr = data + i * ncols ;        // Point to this case in the data
      for (ivis=0 ; ivis<n_inputs ; ivis++)
         data_mean[ivis] += dptr[ivis] ;
      }

   for (ivis=0 ; ivis<n_inputs ; ivis++) {
      data_mean[ivis] /= nc ;
      if (data_mean[ivis] < 1.e-8)
         data_mean[ivis] = 1.e-8 ;
      if (data_mean[ivis] > 1.0 - 1.e-8)
         data_mean[ivis] = 1.0 - 1.e-8 ;
      }

/*
   Initialize CUDA, including sending all data to device
   We need the maximum batch size
*/

   n_done = max_batch = 0 ;
   for (ibatch=0 ; ibatch<n_batches ; ibatch++) {  // An epoch is split into batches of training data
      n_in_batch = (nc - n_done) / (n_batches - ibatch) ;  // Cases left to do / batches left to do
      if (n_in_batch > max_batch)
         max_batch = n_in_batch ;
      n_done += n_in_batch ;
      }

   ret_val = rbm_cuda_init ( nc , ncols , n_inputs , nhid , 1 , 1 , max_batch , data ,
                             data_mean , in_bias , hid_bias , w , msg ) ;

   if (ret_val == ERROR_INSUFFICIENT_MEMORY) {
      audit ( "" ) ;
      audit ( "ERROR... Insufficent memory" ) ;
      return -1.0 ;
      }

   else if (ret_val == ERROR_CUDA_MEMORY) {
      audit ( "" ) ;
      audit ( "Warning... Insufficent video device memory.  Switching to host." ) ;
      audit ( msg ) ;
      return -1.0 ;
      }

   else if (ret_val == ERROR_CUDA_ERROR) {
      audit ( "" ) ;
      audit ( "Warning... Video device error.  Switching to host." ) ;
      audit ( "           This is an unexpected error which should never happen." ) ;
      audit ( "           Please contact the developer." ) ;
      audit ( msg ) ;
      return -1.0 ;
      }


/*
   Initialize the shuffle index, which will be used by fetch_vis1() to extract
   a random batch of cases from the full dataset and also for random number generation
*/

   for (icase=0 ; icase<nc ; icase++)
      shuffle_index[icase] = icase ;



/*
------------------------------------------------------------------------------------------------

   Main loop...
   Try some small weight vectors and choose as starter the one with minimum reproduction error.
   We also initialize all bias vectors to minus half of the weight sum for rough balance.

------------------------------------------------------------------------------------------------
*/

   best_err = 1.e40 ;

   for (irand=0 ; irand<n_rand ; irand++) { // Main loop processes all tries
      error = 0.0 ;

      if (irand == 0) {
         ret_val = cuda_shuffle_to_device ( nc , shuffle_index ) ;
         if (ret_val) {
            audit ( "ERROR... cuda_shuffle_to_device failed" ) ;
            audit ( "         Switching to host, but results may be compromised." ) ;
            return -1.0 ;
            }
         }

      // Generate the trial weight matrix and bias vectors

      diff = 4.0 * unifrand_fast() / sqrt ( sqrt ( (double) n_inputs * nhid ) ) ;

      for (ihid=0 ; ihid<nhid ; ihid++) {
         sum = 0.0 ;

         for (ivis=0 ; ivis<n_inputs ; ivis++) {   // Get all visible weights for this hidden neuron
            wt = diff * (unifrand_fast() - 0.5) ;  // This is symmetric with heavy tails
            w[ihid*n_inputs+ivis] = wt ;
            sum += data_mean[ivis] * wt ;          // We'll need this for this hidden neuron's bias
            }

         hid_bias[ihid] = -sum ;                   // Center the distribution
         } // For ihid


      for (ivis=0 ; ivis<n_inputs ; ivis++) {      // Also center the visible
         sum = 0.0 ;
         for (ihid=0 ; ihid<nhid ; ihid++)
            sum += w[ihid*n_inputs+ivis] ;            
         in_bias[ivis] = log ( data_mean[ivis] / (1.0 - data_mean[ivis]) ) - 0.5 * sum ;
         }

      ret_val = cuda_params_to_device ( n_inputs , nhid , in_bias , hid_bias , w ) ;
      if (ret_val) {
         audit ( "ERROR... cuda_params_to_device failed" ) ;
         audit ( "         Switching to host, but results may be compromised." ) ;
         return -1.0 ;
         }

/*
   Evaluate the reconstruction error for this trial weight set
*/

      n_done = istart = 0 ;
      for (ibatch=0 ; ibatch<n_batches ; ibatch++) {  // An epoch is split into batches of training data
         n_in_batch = (nc - n_done) / (n_batches - ibatch) ;  // Cases left to do / batches left to do
         istop = istart + n_in_batch ;

         // CUDA calls

         // Get visible1 from database
         ret_val = cuda_fetch_vis1 ( istart , istop , n_inputs , 1 , NULL ) ;
         if (ret_val) {
            audit ( "ERROR... cuda_fetch_vis1 failed" ) ;
            return -1.0 ;
            }

         // Compute hidden1 probability (no sampling)
         ret_val = cuda_vis_to_hid ( n_in_batch , nhid , NULL , NULL , NULL ) ;
         if (ret_val) {
            audit ( "ERROR... cuda_vis_to_hid failed" ) ;
            return -1.0 ;
            }

         ret_val = cuda_hid_to_vis_direct ( n_in_batch , n_inputs ) ;  // No sampling
         if (ret_val) {
            audit ( "ERROR... cuda_hid_to_vis_direct failed" ) ;
            return -1.0 ;
            }

         ret_val = cuda_recon_error ( n_inputs , n_in_batch , err_vec ) ;

         for (ivis=0 ; ivis<n_inputs ; ivis++)
            error += err_vec[ivis] ;   // Cumulates across epoch (all batches)

         istart = istop ;
         n_done += n_in_batch ;
         } // For all batches


      // If we just improved, save the best-so-far parameters

      if (error < best_err) {
         best_err = error ;
         for (ihid=0 ; ihid<nhid ; ihid++) {
            hid_bias_best[ihid] = hid_bias[ihid] ;
            for (ivis=0 ; ivis<n_inputs ; ivis++)
               w_best[ihid*n_inputs+ivis] = w[ihid*n_inputs+ivis] ;
            }
          for (ivis=0 ; ivis<n_inputs ; ivis++)
            in_bias_best[ivis] = in_bias[ivis] ;
         }

      if (escape_key_pressed  ||  user_pressed_escape ()) {
         user_pressed_escape () ;
         escape_key_pressed = 0 ;   // Allow subsequent opertations to continue
         audit ( "" ) ;
         audit ( "WARNING... User pressed ESCape!  Incomplete results" ) ;
         audit ( "" ) ;
         break ;
         }

      } // For all random tries

/*
   Copy the best parameters (in ?_best) into the weights.
   Since the error is stochastic, we cannot expect an exact match with what we will get
   on the first epoch, which uses the 'best' weights.  But they should usually be close.
*/

   for (ihid=0 ; ihid<nhid ; ihid++) {
      hid_bias[ihid] = hid_bias_best[ihid] ;
      for (ivis=0 ; ivis<n_inputs ; ivis++)
         w[ihid*n_inputs+ivis] = w_best[ihid*n_inputs+ivis] ;
      }

   for (ivis=0 ; ivis<n_inputs ; ivis++)
      in_bias[ivis] = in_bias_best[ivis] ;

   rbm_cuda_cleanup () ;

   return best_err / (nc * n_inputs) ;
}


/*
------------------------------------------------------------------------------------------------

   rbm_cuda() - Train RBM weights via CUDA

------------------------------------------------------------------------------------------------
*/

double rbm_cuda (
   int nc ,                  // Number of cases in complete dataset
   int ncols ,               // Number of columns in data
   double *data ,            // Nc rows by ncols columns of 0-1 input data in first n_inputs cols
   int n_inputs ,            // Number of inputs
   int nhid ,                // Number of hidden neurons
   int n_chain_start ,       // Starting length of Markov chain, generally 1
   int n_chain_end ,         // Ending length of Markov chain, generally 1 or a small number
   double n_chain_rate ,     // Exponential smoothing rate for epochs moving toward n_chain_end
   int mean_field ,          // Use mean field instead of random sampling?
   int greedy_mean_field ,   // Use mean field for greedy training?
   int n_batches ,           // Number of batches per epoch
   int max_epochs ,          // Maximum number of epochs
   int max_no_imp ,          // Converged if this many epochs with no ratio improvement
   double convergence_crit , // Convergence criterion for max inc / max weight
   double learning_rate ,    // Learning rate
   double start_momentum ,   // Learning momentum start value
   double end_momentum ,     // Learning momentum end value
   double weight_pen ,       // Weight penalty
   double sparsity_penalty , // Sparsity penalty
   double sparsity_target ,  // Sparsity target
   double *w ,               // Computed weight matrix, nhid sets of n_inputs weights
   double *in_bias ,         // Computed input bias vector
   double *hid_bias ,        // Computed hidden bias vector
   int *shuffle_index ,      // Work vector nc long
   double *data_mean ,       // Work vector n_inputs long
   double *err_vec           // Work vector n_inputs long
   )
{
   int i, j, k, i_epoch, icase, ivis, n_no_improvement, ret_val, timer ;
   int istart, istop, ibatch, n_done, n_in_batch, max_batch, ichain, randnum ;
   double error, best_err, max_inc, momentum, chain_length ;
   double dtemp, sum, len_this, len_prev, dot, smoothed_this, smoothed_ratio ;
   double smoothed_dot, max_weight, best_crit, most_recent_correct_error ;
   char msg[256] ;


   randnum = 1 ;

/*
   Find the mean of each input for sparsity penalty on weights
   The 'data' array MUST have n_inputs columns on the device to avoid wasting memory.
   But it will be called with data having ncols columns, with the required data
   being in the first n_inputs columns.
*/

   for (ivis=0 ; ivis<n_inputs ; ivis++)
      data_mean[ivis] = 0.0 ;

   for (icase=0 ; icase<nc ; icase++) {
      for (ivis=0 ; ivis<n_inputs ; ivis++)
         data_mean[ivis] += data[icase*ncols+ivis] ;
      }

   for (ivis=0 ; ivis<n_inputs ; ivis++) {
      data_mean[ivis] /= nc ;
      if (data_mean[ivis] < 1.e-8)
         data_mean[ivis] = 1.e-8 ;
      if (data_mean[ivis] > 1.0 - 1.e-8)
         data_mean[ivis] = 1.0 - 1.e-8 ;
      }


/*
   Initialize the shuffle index, which will be used by fetch_vis1() to extract
   a random batch of cases from the full dataset and also for random number generation
*/

   for (icase=0 ; icase<nc ; icase++)
      shuffle_index[icase] = icase ;

/*
   Initialize CUDA, including sending all data to device
   We need the maximum batch size
*/

   n_done = max_batch = 0 ;
   for (ibatch=0 ; ibatch<n_batches ; ibatch++) {  // An epoch is split into batches of training data
      n_in_batch = (nc - n_done) / (n_batches - ibatch) ;  // Cases left to do / batches left to do
      if (n_in_batch > max_batch)
         max_batch = n_in_batch ;
      n_done += n_in_batch ;
      }

   ret_val = rbm_cuda_init ( nc , ncols , n_inputs , nhid , mean_field , greedy_mean_field , max_batch , data ,
                             data_mean , in_bias , hid_bias , w , msg ) ;

   if (ret_val == ERROR_INSUFFICIENT_MEMORY) {
      audit ( "" ) ;
      audit ( "ERROR... Insufficent memory" ) ;
      return -1.0 ;
      }

   else if (ret_val == ERROR_CUDA_MEMORY) {
      audit ( "" ) ;
      audit ( "Warning... Insufficent video device memory.  Switching to host." ) ;
      audit ( msg ) ;
      return -1.0 ;
      }

   else if (ret_val == ERROR_CUDA_ERROR) {
      audit ( "" ) ;
      audit ( "Warning... Video device error.  Switching to host." ) ;
      audit ( "           This is an unexpected error which should never happen." ) ;
      audit ( "           Please contact the developer." ) ;
      audit ( msg ) ;
      return -1.0 ;
      }

   CudaTimers.rbm_ncalls = 0 ;
   CudaTimers.rbm_ncalls_chain = 0 ;
   CudaTimers.rbm_fetch = 0 ;
   CudaTimers.rbm_vis_to_hid = 0 ;
   CudaTimers.rbm_hid_to_vis = 0 ;
   CudaTimers.rbm_vis2_to_hid2 = 0 ;
   CudaTimers.rbm_sample_hid2 = 0 ;
   CudaTimers.rbm_recon = 0 ;
   CudaTimers.rbm_update_in_bias = 0 ;
   CudaTimers.rbm_update_hid_bias = 0 ;
   CudaTimers.rbm_update_w = 0 ;
   CudaTimers.rbm_transpose = 0 ;
   CudaTimers.rbm_max_inc = 0 ;
   CudaTimers.rbm_len_dot = 0 ;


/*
   Training starts here
*/

   momentum = start_momentum ;
   chain_length = n_chain_start ;
   n_no_improvement = 0 ;  // Counts failure of ratio to improve
   
   for (i_epoch=0 ; i_epoch<max_epochs ; i_epoch++) { // Each epoch is a complete pass through all training data

/*
   Shuffle the data so that if it has serial correlation, similar cases do not end up
   in the same batch.  It's also nice to vary the contents of each batch,
   epoch to epoch, for more diverse averaging.
*/

      i = nc ;                         // Number remaining to be shuffled
      while (i > 1) {                  // While at least 2 left to shuffle
         j = (int) (unifrand_fast () * i) ;
         if (j >= i)
            j = i - 1 ;
         k = shuffle_index[--i] ;
         shuffle_index[i] = shuffle_index[j] ;
         shuffle_index[j] = k ;
         }

      ret_val = cuda_shuffle_to_device ( nc , shuffle_index ) ;
      if (ret_val) {
         audit ( "ERROR... cuda_shuffle_to_device failed" ) ;
         audit ( "         Switching to host, but results may be compromised." ) ;
         return -1.0 ;
         }

/*
------------------------------------------------------------------------------------------------

   Batch loop

------------------------------------------------------------------------------------------------
*/

      istart = 0 ;         // Batch start = training data start
      n_done = 0 ;         // Number of training cases done in this epoch so far
      error = 0.0 ;        // Cumulates reconstruction error across epoch (sum of all batches)
      max_inc = 0.0 ;      // For testing convergence: increment relative to largest magnitude weight

      for (ibatch=0 ; ibatch<n_batches ; ibatch++) {  // An epoch is split into batches of training data
         n_in_batch = (nc - n_done) / (n_batches - ibatch) ;  // Cases left to do / batches left to do
         istop = istart + n_in_batch ;                // Stop just before this index

         ++CudaTimers.rbm_ncalls ;

         // Get visible1 from data array
         if (! greedy_mean_field) {
            k = randnum / IQ ;
            randnum = IA * (randnum - k * IQ) - IR * k ;
            if (randnum < 0)
               randnum += IM ;
            }

         timer = timeGetTime() ;
         ret_val = cuda_fetch_vis1 ( istart , istop , n_inputs , k , NULL ) ;
         if (ret_val) {
            audit ( "ERROR... cuda_fetch_vis1 failed" ) ;
            return -1.0 ;
            }
         CudaTimers.rbm_fetch += timeGetTime() - timer ;


         // Compute hidden1 probability (no sampling); also copy to hidden2 for MC chain
         timer = timeGetTime() ;
         ret_val = cuda_vis_to_hid ( n_in_batch , nhid , NULL , NULL , NULL ) ;
         if (ret_val) {
            audit ( "ERROR... cuda_vis_to_hid failed" ) ;
            return -1.0 ;
            }
         CudaTimers.rbm_vis_to_hid += timeGetTime() - timer ;


/*
   Markov chain
*/

         for (ichain=0 ; ichain<(int)(chain_length+0.5)  ; ichain++) {

            // Sample hidden2 into hidden_act
            k = randnum / IQ ;
            randnum = IA * (randnum - k * IQ) - IR * k ;
            if (randnum < 0)
               randnum += IM ;

            timer = timeGetTime() ;
            ret_val = cuda_sample_hidden2 ( n_in_batch , nhid , randnum , NULL ) ;
            if (ret_val) {
               audit ( "ERROR... cuda_sample_hidden2 failed" ) ;
               return -1.0 ;
               }
            CudaTimers.rbm_sample_hid2 += timeGetTime() - timer ;


            // Use hidden_act to get visible2, sampling visible2 if not mean_field
            k = randnum / IQ ;
            randnum = IA * (randnum - k * IQ) - IR * k ;
            if (randnum < 0)
               randnum += IM ;

            timer = timeGetTime() ;
            ret_val = cuda_hid_to_vis ( n_in_batch , n_inputs , randnum , NULL ) ;
            if (ret_val) {
               audit ( "ERROR... cuda_hid_to_vis failed" ) ;
               return -1.0 ;
               }
            CudaTimers.rbm_hid_to_vis += timeGetTime() - timer ;

            // Cumulate reconstruction error
            if (ichain == 0) {
               timer = timeGetTime() ;
               ret_val = cuda_recon_error ( n_inputs , n_in_batch , err_vec ) ;
               if (ret_val) {
                  audit ( "ERROR... cuda_recon_error failed" ) ;
                  return -1.0 ;
                  }
               CudaTimers.rbm_recon += timeGetTime() - timer ;
               }

            // Use visible2 (which is probabilities or samples per mean_field)
            // to get hidden2 probabilities (no sampling of hidden2)
            timer = timeGetTime() ;
            ret_val = cuda_vis2_to_hid2 ( n_in_batch , nhid , NULL ) ;
            if (ret_val) {
               audit ( "ERROR... cuda_vis2_to_hid2 failed" ) ;
               return -1.0 ;
               }
            CudaTimers.rbm_vis2_to_hid2 += timeGetTime() - timer ;

            ++CudaTimers.rbm_ncalls_chain ;
            } // For Markov chain

/*
   Update parameters, cumulate error, and keep track of max increment for convergence test later
*/


         timer = timeGetTime() ;
         ret_val = cuda_update_in_bias ( n_in_batch , n_inputs , learning_rate , momentum , NULL , NULL ) ;
         if (ret_val) {
            audit ( "ERROR... cuda_update_in_bias failed" ) ;
            return -1.0 ;
            }
         CudaTimers.rbm_update_in_bias += timeGetTime() - timer ;

         // Update hidden bias.  We'll need randnum if not mean_field to sample hidden1 into hidden_act.
         k = randnum / IQ ;
         randnum = IA * (randnum - k * IQ) - IR * k ;
         if (randnum < 0)
            randnum += IM ;

         timer = timeGetTime() ;
         ret_val = cuda_update_hid_bias ( n_in_batch , nhid , learning_rate , momentum ,
                                  randnum , sparsity_penalty , sparsity_target , NULL , NULL ) ;
         if (ret_val) {
            audit ( "ERROR... cuda_update_hid_bias failed" ) ;
            return -1.0 ;
            }
         CudaTimers.rbm_update_hid_bias += timeGetTime() - timer ;

         timer = timeGetTime() ;
         ret_val = cuda_update_weights ( n_in_batch , n_inputs , nhid , learning_rate , momentum , weight_pen ,
                                         sparsity_penalty , sparsity_target , NULL , NULL , NULL ) ;
         if (ret_val) {
            audit ( "ERROR... cuda_update_weights failed" ) ;
            return -1.0 ;
            }
         CudaTimers.rbm_update_w += timeGetTime() - timer ;

         timer = timeGetTime() ;
         ret_val = cuda_transpose ( n_inputs , nhid ) ;
         if (ret_val) {
            audit ( "ERROR... cuda_transpose failed" ) ;
            return -1.0 ;
            }
         CudaTimers.rbm_transpose += timeGetTime() - timer ;

         for (ivis=0 ; ivis<n_inputs ; ivis++)
            error += err_vec[ivis] ;   // Cumulates across epoch (all batches)

         timer = timeGetTime() ;
         ret_val = cuda_max_inc_w ( n_inputs * nhid , &dtemp , 1 ) ;
         CudaTimers.rbm_max_inc += timeGetTime() - timer ;
         if (ret_val) {
            audit ( "ERROR... cuda_max_inc_w failed" ) ;
            return -1.0 ;
            }

         if (dtemp > max_inc)
            max_inc = dtemp ;

         if (i_epoch  &&  (escape_key_pressed  ||  user_pressed_escape ()))
            break ;

/*
   Cumulate gradient (and previous) lengths and dot product for dynamic updating of learning rate
   The smoothed_? variables are purely for user display
*/

         if (i_epoch == 0  &&  ibatch == 0) {   // No previous gradient yet
            timer = timeGetTime() ;
            ret_val = cuda_len_dot ( n_inputs * nhid , &len_prev , &dot ) ;
            CudaTimers.rbm_len_dot += timeGetTime() - timer ;
            if (ret_val) {
               audit ( "ERROR... cuda_len_dot failed" ) ;
               return -1.0 ;
               }
            smoothed_this = sqrt ( len_prev / (nhid * n_inputs) ) ;
            smoothed_dot = 0.0 ;
            }

         else {
            timer = timeGetTime() ;
            ret_val = cuda_len_dot ( n_inputs * nhid , &len_this , &dot ) ;
            CudaTimers.rbm_len_dot += timeGetTime() - timer ;
            if (ret_val) {
               audit ( "ERROR... cuda_len_dot failed" ) ;
               return -1.0 ;
               }
            dot /= sqrt ( len_this * len_prev ) ;
            len_prev = len_this ;

            if (dot > 0.5)        // Heuristic threshold
               learning_rate *= 1.2 ;
            else if (dot > 0.3)
               learning_rate *= 1.1 ;
            else if (dot < -0.5)
               learning_rate /= 1.2 ;
            else if (dot < -0.3)
               learning_rate /= 1.1 ;
            if (learning_rate > 1.0)
               learning_rate = 1.0 ;
            if (learning_rate < 0.001)
               learning_rate = 0.001 ;

            if (fabs(dot) > 0.3)
               momentum /= 1.5 ;

            smoothed_this = 0.99 * smoothed_this + 0.01 * sqrt ( len_this / (nhid * n_inputs) ) ;
            smoothed_dot = 0.9 * smoothed_dot + 0.1 * dot ;
            }

         n_done += n_in_batch ;
         istart = istop ;

         } // For ibatch

/*
------------------------------------------------------------------------------------------------

   All batches of this epoch have ended.  Finish computations for this epoch.

   WARNING... If the user pressed ESCape during the batch loop, which is the
              most likely situation, the error will not be completely summed
              across all batches, the remaining batches having been skipped.
              Thus, the error now will be too small.

------------------------------------------------------------------------------------------------
*/

      if (i_epoch  &&  (escape_key_pressed  ||  user_pressed_escape ())) { // Make sure we get a complete epoch!
         user_pressed_escape () ;
         escape_key_pressed = 0 ;   // Allow subsequent opertations to continue
         audit ( "" ) ;
         audit ( "WARNING... User pressed ESCape!  Incomplete results" ) ;
         audit ( "" ) ;
         break ;
         }

      error /= nc * n_inputs ;
      most_recent_correct_error = error ; // Needed in case of user ESCape partway through epoch

      if (i_epoch == 0  ||  error < best_err)
         best_err = error ;  // Not currently used; may use it later.


/*
   Test for convergence: largest gradient across epoch relative to largest magnitude weight
*/

      timer = timeGetTime() ;
      ret_val = cuda_max_inc_w ( n_inputs * nhid , &max_weight , 0 ) ;
      CudaTimers.rbm_max_inc += timeGetTime() - timer ;
      if (ret_val) {
         audit ( "ERROR... cuda_max_inc_w failed" ) ;
         return -1.0 ;
         }

      if (max_inc / max_weight < convergence_crit)
         break ;


/*
   Test for convergence: Too many failures to improve
   When we get near convergence, the stochastic nature of the gradient calculation
   causes the update to wander aimlessly
*/

      if (i_epoch == 0  ||  max_inc / max_weight < best_crit) {
         best_crit = max_inc / max_weight ;
         n_no_improvement = 0 ;  // Number of epochs with no improvement
         }

      else {
         ++n_no_improvement ;
         if (n_no_improvement > max_no_imp)  // Test for convergence
            break ;
         }


      momentum = 0.99 * momentum + 0.01 * end_momentum ;
      chain_length = (1.0 - n_chain_rate) * chain_length + n_chain_rate * n_chain_end ;

      if (i_epoch == 0)
         smoothed_ratio = max_inc / max_weight ;
      else
         smoothed_ratio = 0.9 * smoothed_ratio + 0.1 * max_inc / max_weight ;

/*
   Prevent wild gyrations when near convergence
*/

      if (n_no_improvement > 50  &&  learning_rate > 0.03)
         learning_rate = 0.03 ;

      if (n_no_improvement > 100  &&  learning_rate > 0.02)
         learning_rate = 0.02 ;

      if (n_no_improvement > 150  &&  learning_rate > 0.01)
         learning_rate = 0.01 ;

      if (n_no_improvement > 200  &&  learning_rate > 0.005)
         learning_rate = 0.005 ;

      if (n_no_improvement > 250  &&  learning_rate > 0.002)
         learning_rate = 0.002 ;

      } // For i_epoch

   ret_val = cuda_params_from_device ( n_inputs , nhid , in_bias , hid_bias , w ) ;
   if (ret_val) {
      audit ( "ERROR... cuda_params_from_device failed" ) ;
      return -1.0 ;
      }


   rbm_cuda_cleanup () ;

/*
   Print CUDA timers
*/

   sum =  CudaTimers.rbm_fetch + CudaTimers.rbm_vis_to_hid + CudaTimers.rbm_hid_to_vis +
          CudaTimers.rbm_vis2_to_hid2 + CudaTimers.rbm_sample_hid2 + CudaTimers.rbm_recon +
          CudaTimers.rbm_update_in_bias + CudaTimers.rbm_update_hid_bias +
          CudaTimers.rbm_update_w + CudaTimers.rbm_transpose + CudaTimers.rbm_max_inc +
          CudaTimers.rbm_len_dot ;

   cudalog ( "" ) ;
   cudalog ( "" ) ;
   cudalog ( "RBM CUDA times in seconds: total, (percent), per launch" ) ;
   sprintf ( msg, "  Fetch batch data =   %8.3lf   (%5.1lf percent) %10.6lf per launch",
             0.001 * CudaTimers.rbm_fetch,
             100.0 * CudaTimers.rbm_fetch / sum,
             0.001 * CudaTimers.rbm_fetch / CudaTimers.rbm_ncalls ) ;
   cudalog ( msg ) ;
   sprintf ( msg, "  Visible to hidden1 = %8.3lf   (%5.1lf percent) %10.6lf per launch",
             0.001 * CudaTimers.rbm_vis_to_hid,
             100.0 * CudaTimers.rbm_vis_to_hid / sum,
             0.001 * CudaTimers.rbm_vis_to_hid / CudaTimers.rbm_ncalls ) ;
   cudalog ( msg ) ;
   sprintf ( msg, "  Hidden to visible2 = %8.3lf   (%5.1lf percent) %10.6lf per launch",
             0.001 * CudaTimers.rbm_hid_to_vis,
             100.0 * CudaTimers.rbm_hid_to_vis / sum,
             0.001 * CudaTimers.rbm_hid_to_vis / (CudaTimers.rbm_ncalls + CudaTimers.rbm_ncalls_chain) ) ;
   cudalog ( msg ) ;
   sprintf ( msg, "  Visible2 to hidden2 =%8.3lf   (%5.1lf percent) %10.6lf per launch",
             0.001 * CudaTimers.rbm_vis2_to_hid2,
             100.0 * CudaTimers.rbm_vis2_to_hid2 / sum,
             0.001 * CudaTimers.rbm_vis2_to_hid2 / (CudaTimers.rbm_ncalls + CudaTimers.rbm_ncalls_chain) ) ;
   cudalog ( msg ) ;
   sprintf ( msg, "  Sample hidden2 =     %8.3lf   (%5.1lf percent) %10.6lf per launch",
             0.001 * CudaTimers.rbm_sample_hid2,
             100.0 * CudaTimers.rbm_sample_hid2 / sum,
             0.001 * CudaTimers.rbm_sample_hid2 / (CudaTimers.rbm_ncalls_chain + 1.e-10)) ;
   cudalog ( msg ) ;
   sprintf ( msg, "  Reconstruction =     %8.3lf   (%5.1lf percent) %10.6lf per launch",
             0.001 * CudaTimers.rbm_recon,
             100.0 * CudaTimers.rbm_recon / sum,
             0.001 * CudaTimers.rbm_recon / CudaTimers.rbm_ncalls ) ;
   cudalog ( msg ) ;
   sprintf ( msg, "  Update input bias =  %8.3lf   (%5.1lf percent) %10.6lf per launch",
             0.001 * CudaTimers.rbm_update_in_bias,
             100.0 * CudaTimers.rbm_update_in_bias / sum,
             0.001 * CudaTimers.rbm_update_in_bias / CudaTimers.rbm_ncalls ) ;
   cudalog ( msg ) ;
   sprintf ( msg, "  Update hidden bias = %8.3lf   (%5.1lf percent) %10.6lf per launch",
             0.001 * CudaTimers.rbm_update_hid_bias,
             100.0 * CudaTimers.rbm_update_hid_bias / sum,
             0.001 * CudaTimers.rbm_update_hid_bias / CudaTimers.rbm_ncalls ) ;
   cudalog ( msg ) ;
   sprintf ( msg, "  Update weights =     %8.3lf   (%5.1lf percent) %10.6lf per launch",
             0.001 * CudaTimers.rbm_update_w,
             100.0 * CudaTimers.rbm_update_w / sum,
             0.001 * CudaTimers.rbm_update_w / CudaTimers.rbm_ncalls ) ;
   cudalog ( msg ) ;
   sprintf ( msg, "  Transpose =          %8.3lf   (%5.1lf percent) %10.6lf per launch",
             0.001 * CudaTimers.rbm_transpose,
             100.0 * CudaTimers.rbm_transpose / sum,
             0.001 * CudaTimers.rbm_transpose / CudaTimers.rbm_ncalls ) ;
   cudalog ( msg ) ;
   sprintf ( msg, "  Find max inc/w =     %8.3lf   (%5.1lf percent) %10.6lf per launch",
             0.001 * CudaTimers.rbm_max_inc,
             100.0 * CudaTimers.rbm_max_inc / sum,
             0.001 * CudaTimers.rbm_max_inc / (CudaTimers.rbm_ncalls + i_epoch) ) ;
   cudalog ( msg ) ;
   sprintf ( msg, "  Compute dot product =%8.3lf   (%5.1lf percent) %10.6lf per launch",
             0.001 * CudaTimers.rbm_len_dot,
             100.0 * CudaTimers.rbm_len_dot / sum,
             0.001 * CudaTimers.rbm_len_dot / CudaTimers.rbm_ncalls ) ;
   cudalog ( msg ) ;

   return most_recent_correct_error ;
}