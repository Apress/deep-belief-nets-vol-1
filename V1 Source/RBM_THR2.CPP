/******************************************************************************/
/*                                                                            */
/*  RBM_THR2 - Restricted Boltzman Machine trains a single hidden layer       */
/*                                                                            */
/*  This is the main routine which refines the weights                        */
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
------------------------------------------------------------------------------------------------

   Threaded routine that cumulates error and gradient for a sub-batch

------------------------------------------------------------------------------------------------
*/

#define IA 16807
#define IM 2147483647
#define AM (1.0 / IM)
#define IQ 127773
#define IR 2836

static void rbm2_threaded (
   int istart ,            // First case in this batch
   int istop ,             // One past last case
   int ncols ,             // Number of columns in data
   int n_inputs ,          // Number of inputs
   double *data ,          // 'Training cases' rows by ncols columns of input data; 0-1
   int nhid ,              // Number of hidden neurons
   int n_chain ,           // Length of Markov chain
   int mean_field ,        // Use mean field instead of random sampling?
   int greedy_mean_field , // Use mean field for greedy training?
   double *w ,             // Weight matrix, nhid sets of n_inputs weights
   double *in_bias ,       // Input bias vector
   double *hid_bias ,      // Hidden bias vector
   int *shuffle_index ,    // For addressing shuffled data
   double *visible1 ,      // Work vector n_inputs long
   double *visible2 ,      // Work vector n_inputs long
   double *hidden1 ,       // Work vector nhid long
   double *hidden2 ,       // Work vector nhid long
   double *hidden_act ,    // Work vector nhid long
   double *in_bias_grad ,  // Cumulate gradient here
   double *hid_bias_grad , // Cumulate gradient here
   double *w_grad ,        // Cumulate gradient here
   double *hid_on_frac ,   // Cumulate fraction of time each hidden neuron is on
   double *error           // Cumulates reconstruction criterion
   )

{
   int k, randnum, icase, ivis, ihid, ichain ;
   double sum, *wptr, *dptr, P, Q, frand ;

   randnum = (istop + shuffle_index[0]) % IM ;
   if (randnum == 0)
      randnum = 1 ;


/*
   Zero the arrays that will cumulate gradient and error for this batch
*/

   for (ihid=0 ; ihid<nhid ; ihid++) {
      hid_bias_grad[ihid] = 0.0 ;
      hid_on_frac[ihid] = 0.0 ;
      for (ivis=0 ; ivis<n_inputs ; ivis++)
         w_grad[ihid*n_inputs+ivis] = 0.0 ;
      }

   for (ivis=0 ; ivis<n_inputs ; ivis++)
      in_bias_grad[ivis] = 0.0 ;

   *error = 0.0 ;

      
/*
   Loop over input cases (each a vector) in this batch.

   If this model is being greedily trained AND its input is a prior model's
   hidden probabilities AND the user has chosen to not use mean field
   then we must sample the inputs.
   All of these factors have been taken into account by the caller.
*/

   for (icase=istart ; icase<istop ; icase++) {
      dptr = data + shuffle_index[icase] * ncols ;  // Point to this case in the data
      for (ivis=0 ; ivis<n_inputs ; ivis++)
         visible1[ivis] = dptr[ivis] ;

      if (! greedy_mean_field) {
         for (ivis=0 ; ivis<n_inputs ; ivis++) {
            k = randnum / IQ ;
            randnum = IA * (randnum - k * IQ) - IR * k ;
            if (randnum < 0)
               randnum += IM ;
            frand = AM * randnum ;
            visible1[ivis] = (frand < visible1[ivis])  ?  1.0 : 0.0 ;
            }
         }

/*
   For each hidden neuron, compute Q[h=1|visible1]
   The positive (data) term will be visible1 * hidden1
*/

      for (ihid=0 ; ihid<nhid ; ihid++) {
         wptr = w + ihid * n_inputs ;        // Weight vector for this neuron
         sum = hid_bias[ihid] ;
         for (ivis=0 ; ivis<n_inputs ; ivis++)
            sum += wptr[ivis] * visible1[ivis] ;
         Q = 1.0 / (1.0 + exp(-sum)) ;       // Probability
         hidden1[ihid] = hidden2[ihid] = Q ; // We'll need hidden2 for CD-k loop below
         hid_on_frac[ihid] += Q ;            // Need this for sparsity penalty
         }

#if RECON_ERR_DIRECT
      // Compute the reconstruction error the deterministic but expensive way
      for (ivis=0 ; ivis<n_inputs ; ivis++) {
         sum = in_bias[ivis] ;
         for (ihid=0 ; ihid<nhid ; ihid++)
            sum += w[ihid*n_inputs+ivis] * hidden1[ihid] ;
         P = 1.0 / (1.0 + exp(-sum)) ;
#if RECON_ERR_XENT
         *error -= visible1[ivis] * log(P+1.e-10) + (1.0 - visible1[ivis]) * log(1.0-P+1.e-10) ;
#else
         double diff = visible1[ivis] - P ;
         *error += diff * diff ;
#endif
         }
#endif

/*
   Continue the Markov chain
*/

      for (ichain=0 ; ichain<n_chain ; ichain++) {

         // Sample Q[h|x] to get next (binary) hidden layer.

         for (ihid=0 ; ihid<nhid ; ihid++) {
            k = randnum / IQ ;
            randnum = IA * (randnum - k * IQ) - IR * k ;
            if (randnum < 0)
               randnum += IM ;
            frand = AM * randnum ;
            hidden_act[ihid] = (frand < hidden2[ihid])  ?  1.0 : 0.0 ;
            }

         // For each visible neuron, compute P[x=1|hidden layer] and then
         // sample (if not mean_field) its value as x2

         for (ivis=0 ; ivis<n_inputs ; ivis++) {
            sum = in_bias[ivis] ;
            for (ihid=0 ; ihid<nhid ; ihid++)
               sum += w[ihid*n_inputs+ivis] * hidden_act[ihid] ;
            P = 1.0 / (1.0 + exp(-sum)) ;                   // This is the probability

#if ! RECON_ERR_DIRECT
            // Compute the reconstruction error the stochastic but fast way
            if (ichain == 0) {
#if RECON_ERR_XENT
               *error -= visible1[ivis] * log(P+1.e-10) + (1.0-visible1[ivis]) * log(1.0-P+1.e-10) ;
#else
               double diff = visible1[ivis] - P ;
               *error += diff * diff ;
#endif
               }
#endif

            if (mean_field)
               visible2[ivis] = P ;
            else {
               k = randnum / IQ ;
               randnum = IA * (randnum - k * IQ) - IR * k ;
               if (randnum < 0)
                  randnum += IM ;
               frand = AM * randnum ;
               visible2[ivis] = (frand < P)  ?  1.0 : 0.0 ;  // Sample the activation
               }
            } // For each visible neuron, computing its probability and sampling if not mean_field


         // For each hidden neuron, compute Q[h=1|visible2]

         for (ihid=0 ; ihid<nhid ; ihid++) {
            wptr = w + ihid * n_inputs ;      // Weight vector for this neuron
            sum = hid_bias[ihid] ;
            for (ivis=0 ; ivis<n_inputs ; ivis++)
               sum += wptr[ivis] * visible2[ivis] ;
            hidden2[ihid] = 1.0 / (1.0 + exp(-sum)) ;
            }
         } // For Markov chain

/*
   cumulate negative gradient for weights and bias terms in this batch
*/

      for (ihid=0 ; ihid<nhid ; ihid++) {

         if (mean_field) {
            hid_bias_grad[ihid] += hidden1[ihid] - hidden2[ihid] ;
            for (ivis=0 ; ivis<n_inputs ; ivis++)
               w_grad[ihid*n_inputs+ivis] += hidden1[ihid] * visible1[ivis] - hidden2[ihid] * visible2[ivis] ;
            }

         else {
            k = randnum / IQ ;
            randnum = IA * (randnum - k * IQ) - IR * k ;
            if (randnum < 0)
               randnum += IM ;
            frand = AM * randnum ;
            hidden_act[ihid] = (frand < hidden1[ihid])  ?  1.0 : 0.0 ;
            hid_bias_grad[ihid] += hidden_act[ihid] - hidden2[ihid] ;
            for (ivis=0 ; ivis<n_inputs ; ivis++)
               w_grad[ihid*n_inputs+ivis] += hidden_act[ihid] * visible1[ivis] - hidden2[ihid] * visible2[ivis] ;
            }
         }

      for (ivis=0 ; ivis<n_inputs ; ivis++)
         in_bias_grad[ivis] += visible1[ivis] - visible2[ivis] ;

      } // For each case in this batch
}



/*
--------------------------------------------------------------------------------

   Thread stuff...
      Structure for passing information to/from threaded code
      Threaded code called by the main subroutine

--------------------------------------------------------------------------------
*/

typedef struct {
   int istart ;            // First case in this batch
   int istop ;             // One past last case
   int ncols ;             // Number of columns in data
   int n_inputs ;          // Number of inputs
   double *data ;          // 'Training cases' rows by ncols columns of input data; 0-1
   int nhid ;              // Number of hidden neurons
   int n_chain ;           // Length of Markov chain; typically 1
   int mean_field ;        // Use mean field instead of random sampling?
   int greedy_mean_field ; // Use mean field for greedy training?
   double *w ;             // Weight matrix; nhid sets of n_inputs weights
   double *in_bias ;       // Input bias vector
   double *hid_bias ;      // Hidden bias vector
   int *shuffle_index ;    // For addressing shuffled data
   double *visible1 ;      // Work vector n_inputs long
   double *visible2 ;      // Work vector n_inputs long
   double *hidden1 ;       // Work vector nhid long
   double *hidden2 ;       // Work vector nhid long
   double *hidden_act ;    // Work vector nhid long
   double *in_bias_grad ;  // Cumulates gradient here
   double *hid_bias_grad ; // Cumulates gradient here
   double *w_grad ;        // Cumulates gradient here
   double *hid_on_frac ;   // Cumulates fraction of time each hidden neuron is on
   double *error ;         // Cumulates MSE
} RBM_THR2_PARAMS ;

static unsigned int __stdcall rbm2_wrapper ( LPVOID dp )
{
rbm2_threaded (
                          ((RBM_THR2_PARAMS *) dp)->istart ,
                          ((RBM_THR2_PARAMS *) dp)->istop ,
                          ((RBM_THR2_PARAMS *) dp)->ncols ,
                          ((RBM_THR2_PARAMS *) dp)->n_inputs ,
                          ((RBM_THR2_PARAMS *) dp)->data ,
                          ((RBM_THR2_PARAMS *) dp)->nhid ,
                          ((RBM_THR2_PARAMS *) dp)->n_chain ,
                          ((RBM_THR2_PARAMS *) dp)->mean_field ,
                          ((RBM_THR2_PARAMS *) dp)->greedy_mean_field ,
                          ((RBM_THR2_PARAMS *) dp)->w ,
                          ((RBM_THR2_PARAMS *) dp)->in_bias ,
                          ((RBM_THR2_PARAMS *) dp)->hid_bias ,
                          ((RBM_THR2_PARAMS *) dp)->shuffle_index ,
                          ((RBM_THR2_PARAMS *) dp)->visible1 ,
                          ((RBM_THR2_PARAMS *) dp)->visible2 ,
                          ((RBM_THR2_PARAMS *) dp)->hidden1 ,
                          ((RBM_THR2_PARAMS *) dp)->hidden2 ,
                          ((RBM_THR2_PARAMS *) dp)->hidden_act ,
                          ((RBM_THR2_PARAMS *) dp)->in_bias_grad ,
                          ((RBM_THR2_PARAMS *) dp)->hid_bias_grad ,
                          ((RBM_THR2_PARAMS *) dp)->w_grad ,
                          ((RBM_THR2_PARAMS *) dp)->hid_on_frac ,
                          ((RBM_THR2_PARAMS *) dp)->error  ) ;
   return 0 ;
}


/*
------------------------------------------------------------------------------------------------

   Main routine called from greedy()

------------------------------------------------------------------------------------------------
*/


double rbm_thr2 (
   int nc ,                  // Number of training cases
   int ncols ,               // Number of columns in data
   double *data ,            // Nc rows by ncols columns of input data; 0-1
   int n_inputs ,            // Number of inputs
   int nhid ,                // Number of hidden neurons
   int max_neurons ,         // Maximum number of neurons in any layer, as well as nin
   int n_chain_start ,       // Starting length of Markov chain, generally 1
   int n_chain_end ,         // Ending length of Markov chain, generally 1 or a small number
   double n_chain_rate ,     // Exponential smoothing rate for epochs moving toward n_chain_end
   int mean_field ,          // Use mean field instead of random sampling?
   int greedy_mean_field ,   // Use mean field for greedy training?
   int n_batches ,           // Number of batches per epoch
   int max_epochs ,          // Maximum number of epochs
   int max_no_improvement ,  // Converged if this many epochs with no ratio improvement
   double convergence_crit , // Convergence criterion for max inc / max weight
   double learning_rate ,    // Learning rate
   double start_momentum ,   // Learning momentum start value
   double end_momentum ,     // Learning momentum end value
   double weight_penalty ,   // Weight penalty
   double sparsity_penalty , // Sparsity penalty
   double sparsity_target ,  // Sparsity target
   double *w ,               // Computed weight matrix, nhid sets of n_inputs weights
   double *in_bias ,         // Computed input bias vector
   double *hid_bias ,        // Computed hidden bias vector
   int *shuffle_index ,      // Work vector nc long
   double *data_mean ,       // Work vector n_inputs long
   double *visible1 ,        // Work vector n_inputs * max_threads long
   double *visible2 ,        // Work vector n_inputs * max_threads long
   double *hidden1 ,         // Work vector nhid * max_threads long
   double *hidden2 ,         // Work vector nhid * max_threads long
   double *hidden_act ,      // Work vector nhid * max_threads long
   double *hid_on_frac ,     // Work vector nhid * max_threads long
   double *hid_on_smoothed , // Work vector nhid long
   double *in_bias_inc ,     // Work vector n_inputs long
   double *hid_bias_inc ,    // Work vector nhid long
   double *w_inc ,           // Work vector n_inputs * nhid long
   double *in_bias_grad ,    // Work vector n_inputs * max_threads long
   double *hid_bias_grad ,   // Work vector nhid * max_threads long
   double *w_grad ,          // Work vector n_inputs * nhid * max_threads long
   double *w_prev            // Work vector n_inputs * nhid long
   )

{
   int i_epoch ;      // Each epoch is a complete pass through all training data
   int n_threads ;    // Each batch is broken into this many threads
   int ivis ;         // Index within visible layer
   int ihid ;         // Index of hidden neuron
   int istart ;       // Index in dataset of first batch case
   int istop ;        // And one past last batch case
   int jstart ;       // Offset in batch of first thread case
   int jstop ;        // And one past last thread case
   int n_in_batch ;   // Number of training cases in the batch being processed
   int n_in_thread ;  // Number of training cases in the thread being processed
   int ibatch ;       // Batch number being processed
   int ithread ;      // Thread number being processed
   int n_done ;       // Number of training cases done in this epoch so far
   int nt_done ;      // Number of training cases done in this batch so far
   int n_no_improvement ; // Number of consecutive times convergence crit failed to improve
   double chain_length ; // Chain length, which may be exponentially smoothed upwards
   double error ;     // Mean squared error for each epoch; sum of squared diffs between input and P[x=1|hidden layer]
   double best_err ;  // Best error seen so far

   int i, j, k, ret_val ;

   double *dptr, momentum, max_inc, max_weight, error_vec[MAX_THREADS], best_crit ;
   double sp_pen, x_this, x_prev, len_this, len_prev, dot, smoothed_this, smoothed_ratio, smoothed_dot ;
   double most_recent_correct_error ;
   char msg[4096] ;
   RBM_THR2_PARAMS params[MAX_THREADS] ;
   HANDLE threads[MAX_THREADS] ;

/*
   Find the mean of the data for each input.
   This is used for sparsity targeting in the weights
*/

   for (ivis=0 ; ivis<n_inputs ; ivis++)
      data_mean[ivis] = 0.0 ;

   for (i=0 ; i<nc ; i++) {          // Pass through all cases, cumulating mean vector
      dptr = data + i * ncols ;      // Point to this case in the data
      for (ivis=0 ; ivis<n_inputs ; ivis++)
         data_mean[ivis] += dptr[ivis] ;
      }

   for (ivis=0 ; ivis<n_inputs ; ivis++)
      data_mean[ivis] /= nc ;

/*
   Initialize parameters that will not change
*/

   for (i=0 ; i<max_threads ; i++) {
      params[i].mean_field = mean_field ;
      params[i].greedy_mean_field = greedy_mean_field ;
      params[i].n_inputs = n_inputs ;
      params[i].ncols = ncols ;
      params[i].nhid = nhid ;
      params[i].data = data ;
      params[i].in_bias = in_bias ;
      params[i].hid_bias = hid_bias ;
      params[i].w = w ;
      params[i].shuffle_index = shuffle_index ;
      params[i].visible1 = visible1 + i * max_neurons ;
      params[i].visible2 = visible2 + i * max_neurons ;
      params[i].hidden1 = hidden1 + i * max_neurons ;
      params[i].hidden2 = hidden2 + i * max_neurons ;
      params[i].hidden_act = hidden_act + i * max_neurons ;
      params[i].in_bias_grad = in_bias_grad + i * max_neurons ;
      params[i].hid_bias_grad = hid_bias_grad + i * max_neurons ;
      params[i].hid_on_frac = hid_on_frac + i * max_neurons ;
      params[i].w_grad = w_grad + i * nhid * n_inputs ;
      params[i].error = error_vec + i ;
      }

/*
   Initialize the parameter increments to zero for momentum.
   Also initialize the smoothed hid_on_frac to 0.5.
*/

   for (ihid=0 ; ihid<nhid ; ihid++) {
      hid_bias_inc[ihid] = 0.0 ;
      hid_on_smoothed[ihid] = 0.5 ;
      for (ivis=0 ; ivis<n_inputs ; ivis++)
         w_inc[ihid*n_inputs+ivis] = 0.0 ;
      }

   for (ivis=0 ; ivis<n_inputs ; ivis++)
      in_bias_inc[ivis] = 0.0 ;

/*
------------------------------------------------------------------------------------------------

   Outermost loop is epochs, with each epoch being a complete pass through all training data
   Just inside this loop is the batch loop.  Training cases are processed in batches,
   with updates being averaged across the batch and applied when the batch is complete.
   This averaging is good because it smooths the path taken by following the gradient.
   Also, in large problems the cost of updating is significant, so it would be prohibitively
   expensive to update after every case.

------------------------------------------------------------------------------------------------
*/

   // We'll shuffle before each epoch, so initialize indices
   for (i=0 ; i<nc ; i++)
      shuffle_index[i] = i ;

   momentum = start_momentum ;
   n_no_improvement = 0 ;       // Counts failure of ratio to improve
   chain_length = n_chain_start ;
   
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

/*
------------------------------------------------------------------------------------------------

   Batch loop

------------------------------------------------------------------------------------------------
*/

      istart = 0 ;         // Batch start = training data start
      n_done = 0 ;         // Number of training cases done in this epoch so far
      error = 0.0 ;        // Cumulates reproduction error

      max_inc = 0.0 ;      // For testing convergence: increment relative to largest magnitude weight

      for (ibatch=0 ; ibatch<n_batches ; ibatch++) {  // An epoch is split into batches of training data
         n_in_batch = (nc - n_done) / (n_batches - ibatch) ;  // Cases left to do / batches left to do
         istop = istart + n_in_batch ;                // Stop just before this index

/*
------------------------------------------------------------------------------------------------

   Thread loop that breaks up this single batch

------------------------------------------------------------------------------------------------
*/

         n_threads = max_threads ;                              // Try to use as many as possible
         while (n_threads > 1  &&  n_in_batch / n_threads < 10) // But due to overhead make each do significant work
            --n_threads ;                                       // The choice of constant is difficult

         jstart = 0 ;                      // Thread within this batch
         nt_done = 0 ;                     // Number in this batch done

/*
   Start the threads
*/

         for (ithread=0 ; ithread<n_threads ; ithread++) {
            n_in_thread = (n_in_batch - nt_done) / (n_threads - ithread) ;
            jstop = jstart + n_in_thread ;

            // Set the pointers that vary with the batch or epoch

            params[ithread].istart = istart + jstart ;
            params[ithread].istop = istart + jstop ;
            params[ithread].n_chain = (int) (chain_length + 0.5) ; // Fixed throughout each epoch

            threads[ithread] = (HANDLE) _beginthreadex ( NULL , 0 , rbm2_wrapper , &params[ithread] , 0 , NULL ) ;
            if (threads[ithread] == NULL) {
               audit ( "Internal ERROR: bad thread creation in RBM_THR2" ) ;
               for (i=0 ; i<n_threads ; i++) {
                  if (threads[i] != NULL)
                     CloseHandle ( threads[i] ) ;
                  }
               return -1.e40 ;
               }

            nt_done += n_in_thread ;
            jstart = jstop ;
            } // For all threads in this batch

/*
   Wait for threads to finish
*/

         ret_val = WaitForMultipleObjects ( n_threads , threads , TRUE , 1200000 ) ;
         if (ret_val == WAIT_TIMEOUT  ||  ret_val == WAIT_FAILED  ||  ret_val < 0  ||  ret_val >= n_threads) {
            sprintf ( msg, "INTERNAL ERROR!!!  Thread wait failed (%d) in RBM_THR1.CPP", ret_val ) ;
            audit ( msg ) ;
            MEMTEXT ( msg ) ;
            if (ret_val == WAIT_TIMEOUT)
               audit ( "Timeout waiting for computation to finish; problem too large" ) ;
            return -1.e40 ;
            }

/*
   Pool gradient, error, and hid_on_frac from all threads
*/

         CloseHandle ( threads[0] ) ;
         for (ithread=1 ; ithread<n_threads ; ithread++) {
            for (ihid=0 ; ihid<nhid ; ihid++) {
               hid_bias_grad[ihid] += (params[ithread].hid_bias_grad)[ihid] ;
               hid_on_frac[ihid] += (params[ithread].hid_on_frac)[ihid] ;
               for (ivis=0 ; ivis<n_inputs ; ivis++)
                  w_grad[ihid*n_inputs+ivis] += (params[ithread].w_grad)[ihid*n_inputs+ivis] ;
               }
            for (ivis=0 ; ivis<n_inputs ; ivis++)
               in_bias_grad[ivis] += (params[ithread].in_bias_grad)[ivis] ;
            error_vec[0] += error_vec[ithread] ;
            CloseHandle ( threads[ithread] ) ;
            }

/*
------------------------------------------------------------------------------------------------

   A single batch has ended

------------------------------------------------------------------------------------------------
*/


/*
   Update error, smoothed on fraction, weights, and bias terms as of this batch.
*/


         error += error_vec[0] ;

         for (ihid=0 ; ihid<nhid ; ihid++) {
            hid_on_frac[ihid] /= n_in_batch ;
            hid_on_smoothed[ihid] = 0.95 * hid_on_smoothed[ihid] + 0.05 * hid_on_frac[ihid] ;
            sp_pen = sparsity_penalty * (hid_on_smoothed[ihid] - sparsity_target) ;
            if (hid_on_smoothed[ihid] < 0.01)
               sp_pen += 0.5 * (hid_on_smoothed[ihid] - 0.01) ;       // 0.5 is heuristic
            if (hid_on_smoothed[ihid] > 0.99)
               sp_pen += 0.5 * (hid_on_smoothed[ihid] - 0.99) ;
            hid_bias_inc[ihid] = momentum * hid_bias_inc[ihid] +
                                 learning_rate * (hid_bias_grad[ihid] / n_in_batch - sp_pen) ;
            hid_bias[ihid] += hid_bias_inc[ihid] ;

            for (ivis=0 ; ivis<n_inputs ; ivis++) {
               w_grad[ihid*n_inputs+ivis] /= n_in_batch ;   // Negative gradient pooled across this batch
               w_grad[ihid*n_inputs+ivis] -= weight_penalty * w[ihid*n_inputs+ivis] ; // Penalize large weights
               w_grad[ihid*n_inputs+ivis] -= data_mean[ivis] * sp_pen ; // Penalize poor sparsity
               w_inc[ihid*n_inputs+ivis] = momentum * w_inc[ihid*n_inputs+ivis] +
                                           learning_rate * w_grad[ihid*n_inputs+ivis] ;
               w[ihid*n_inputs+ivis] += w_inc[ihid*n_inputs+ivis] ;

               if (fabs(w_inc[ihid*n_inputs+ivis]) > max_inc)   // Will be used to test for convergence at end of epoch
                  max_inc = fabs(w_inc[ihid*n_inputs+ivis]) ;
               }
            } // For ihid

         for (ivis=0 ; ivis<n_inputs ; ivis++) {
            in_bias_inc[ivis] = momentum * in_bias_inc[ivis] +
                                learning_rate * in_bias_grad[ivis] / n_in_batch ;
            in_bias[ivis] += in_bias_inc[ivis] ;
            }

         if (i_epoch  &&  (escape_key_pressed  ||  user_pressed_escape ()))
            break ;

/*
   Cumulate gradient (and previous) lengths and dot product for dynamic updating of learning rate
   The two smoothed_? variables are purely for user display
*/

         if (i_epoch == 0  &&  ibatch == 0) {
            len_this = 0.0 ;
            for (ihid=0 ; ihid<nhid ; ihid++) {
               for (ivis=0 ; ivis<n_inputs ; ivis++) {
                  x_this = w_grad[ihid*n_inputs+ivis] ;
                  w_prev[ihid*n_inputs+ivis] = x_this ;
                  len_this += x_this * x_this ;
                  }
               }
            len_prev = len_this ;
            smoothed_this = sqrt ( len_this / (nhid * n_inputs) ) ;
            smoothed_dot = 0.0 ;
            }

         else {   
            len_this = dot = 0.0 ;
            for (ihid=0 ; ihid<nhid ; ihid++) {
               for (ivis=0 ; ivis<n_inputs ; ivis++) {
                  x_this = w_grad[ihid*n_inputs+ivis] ;
                  x_prev = w_prev[ihid*n_inputs+ivis] ;
                  w_prev[ihid*n_inputs+ivis] = x_this ;
                  len_this += x_this * x_this ;
                  dot += x_this * x_prev ;
                  }
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

         } // For each batch

/*
------------------------------------------------------------------------------------------------

   All batches have ended.  Finish computations for this epoch.

   WARNING... If the user pressed ESCape during the batch loop, which is the
              most likely situation, the error will not be completely summed
              across all batches, the remaining batches having been skipped.
              Thus, the error now will be too small.

------------------------------------------------------------------------------------------------
*/

      if (i_epoch  &&  (escape_key_pressed  ||  user_pressed_escape ())) {
         user_pressed_escape () ;
         escape_key_pressed = 0 ;   // Allow subsequent opertations to continue
         audit ( "" ) ;
         audit ( "WARNING... User pressed ESCape!  Incomplete results" ) ;
         audit ( "" ) ;
         break ;
         }

      error /= nc * n_inputs ;
      most_recent_correct_error = error ;

      if (i_epoch == 0  ||  error < best_err)
         best_err = error ;


/*
   Test for convergence: largest weight increment across epoch relative to largest magnitude weight
*/

      max_weight = 0.0 ;
      for (ihid=0 ; ihid<nhid ; ihid++) {
         for (ivis=0 ; ivis<n_inputs ; ivis++) {
            if (fabs(w[ihid*n_inputs+ivis]) > max_weight)
               max_weight = fabs(w[ihid*n_inputs+ivis]) ;
            }
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
         if (n_no_improvement > max_no_improvement)  // Test for convergence
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

      } // For each epoch

   return most_recent_correct_error ;
}