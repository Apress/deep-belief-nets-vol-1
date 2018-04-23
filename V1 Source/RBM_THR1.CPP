/******************************************************************************/
/*                                                                            */
/*  RBM_THR1 - Restricted Boltzman Machine trains a single hidden layer       */
/*                                                                            */
/*  This is the first half: find good initial weights                         */
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

   Workhorse routine that computes the criterion (reproduction error)
   for a weight matrix

--------------------------------------------------------------------------------
*/

static double rbm1_threaded (
   int nc ,                // Number of cases
   int n_inputs ,          // Number of inputs
   int max_neurons ,       // Maximum number of neurons in any layer, as well as nin
   double *data ,          // Nc rows by max_neurons columns (n_inputs used) of input data; 0-1
   int nhid ,              // Number of hidden neurons
   double *w ,             // Computed weight matrix, nhid sets of n_inputs weights
   double *in_bias ,       // Computed input bias vector
   double *hid_bias ,      // Computed hidden bias vector
   double *visible1 ,      // Work vector n_inputs long
   double *hidden1         // Work vector nhid long
   )
{
   int icase, ihid, ivis ;
   double error, sum, *wptr, *dptr, P ;

   error = 0.0 ;  // Will cumulate reconstruction error, which is our criterion for best parameters here

   for (icase=0 ; icase<nc ; icase++) {    // Pass through all cases, cumulating error
      dptr = data + icase * max_neurons ;  // Point to this case in the data
      for (ivis=0 ; ivis<n_inputs ; ivis++)
         visible1[ivis] = dptr[ivis] ;

      // For each hidden neuron, compute Q[h=1|visible1].  Do not sample.

      for (ihid=0 ; ihid<nhid ; ihid++) {

         wptr = w + ihid * n_inputs ;      // Weight vector for this neuron
         sum = hid_bias[ihid] ;
         for (ivis=0 ; ivis<n_inputs ; ivis++)
            sum += wptr[ivis] * visible1[ivis] ;
         hidden1[ihid] = 1.0 / (1.0 + exp(-sum)) ;
         }

      // For each visible neuron, compute P[x=1|hidden layer]
      // and then find reconstruction error

      for (ivis=0 ; ivis<n_inputs ; ivis++) {
         sum = in_bias[ivis] ;
         for (ihid=0 ; ihid<nhid ; ihid++)
            sum += w[ihid*n_inputs+ivis] * hidden1[ihid] ;
         P = 1.0 / (1.0 + exp(-sum)) ;
#if RECON_ERR_XENT
         error -= visible1[ivis] * log(P+1.e-10) + (1.0 - visible1[ivis]) * log(1.0-P+1.e-10) ;
#else
         double diff ;
         diff = visible1[ivis] - P ;
         error += diff * diff ;
#endif
         }

      } // For icase

   return error ;
}


/*
--------------------------------------------------------------------------------

   Thread stuff...
      Structure for passing information to/from threaded code
      Threaded code called by the main subroutine

--------------------------------------------------------------------------------
*/

typedef struct {
   int nc ;                // Number of cases
   int n_inputs ;          // Number of inputs
   int max_neurons ;       // Maximum number of neurons in any layer, including input
   double *data ;          // Nc rows by max_neurons columns of input data; 0-1
   int nhid ;              // Number of hidden neurons
   double *w ;             // Weight matrix; nhid sets of n_inputs weights
   double *in_bias ;       // Input bias vector
   double *hid_bias ;      // Hidden bias vector
   double *visible1 ;      // Work vector n_inputs long
   double *hidden1 ;       // Work vector nhid long
   double crit ;           // Computed criterion returned here
} RBM_THR1_PARAMS ;

static unsigned int __stdcall rbm1_wrapper ( LPVOID dp )
{
   ((RBM_THR1_PARAMS *) dp)->crit = rbm1_threaded (
                          ((RBM_THR1_PARAMS *) dp)->nc ,
                          ((RBM_THR1_PARAMS *) dp)->n_inputs ,
                          ((RBM_THR1_PARAMS *) dp)->max_neurons ,
                          ((RBM_THR1_PARAMS *) dp)->data ,
                          ((RBM_THR1_PARAMS *) dp)->nhid ,
                          ((RBM_THR1_PARAMS *) dp)->w ,
                          ((RBM_THR1_PARAMS *) dp)->in_bias ,
                          ((RBM_THR1_PARAMS *) dp)->hid_bias ,
                          ((RBM_THR1_PARAMS *) dp)->visible1 ,
                          ((RBM_THR1_PARAMS *) dp)->hidden1 ) ;
   return 0 ;
}


/*
--------------------------------------------------------------------------------

   Main routine called from greedy()

--------------------------------------------------------------------------------
*/

double rbm_thr1 (
   int nc ,                // Number of cases
   int n_inputs ,          // Number of inputs
   int max_neurons ,       // Maximum number of neurons in any layer, including input
   double *data ,          // Nc rows by max_neurons columns of input data; 0-1
   int nhid ,              // Number of hidden neurons
   double *w ,             // Returned weight matrix, nhid sets of n_inputs weights; max_threads sets
   double *in_bias ,       // Returned input bias vector; max_threads sets
   double *hid_bias ,      // Returned hidden bias vector; max_threads sets
   double *visible1 ,      // Work vector n_inputs long; max_threads sets
   double *hidden1 ,       // Work vector nhid long; max_threads sets
   double *in_bias_best ,  // Work vector n_inputs long
   double *hid_bias_best , // Work vector nhid long
   double *w_best ,        // Work vector n_inputs * nhid long
   double *data_mean       // Work vector n_inputs long
   )

{
   int irand, ivis, ihid ;
   int i, k, n_rand, n_threads, empty_slot, ret_val ;
   double error, best_err ;
   double sum, wt, *dptr, *wptr, *hid_bias_ptr, *in_bias_ptr, diff ;
   char msg[4096] ;
   RBM_THR1_PARAMS params[MAX_THREADS] ;
   HANDLE threads[MAX_THREADS] ;

   user_pressed_escape () ;
   escape_key_pressed = 0 ;  // Allow subsequent operations

/*
   Find the mean of the data for each input.
   This is used to initialize visible bias terms to reasonable values.
*/

   for (ivis=0 ; ivis<n_inputs ; ivis++)
      data_mean[ivis] = 0.0 ;

   for (i=0 ; i<nc ; i++) {            // Pass through all cases, cumulating mean vector
      dptr = data + i * max_neurons ;  // Point to this case in the data
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
   Get the training parameters from the global storage
   Initialize parameters that will not change for threads.
*/

   n_rand = TrainParams.n_rand ;

   for (i=0 ; i<max_threads ; i++) {
      params[i].nc = nc ;
      params[i].n_inputs = n_inputs ;
      params[i].max_neurons = max_neurons ;
      params[i].nhid = nhid ;
      params[i].data = data ;
      params[i].visible1 = visible1 + i * max_neurons ;
      params[i].hidden1 = hidden1 + i * max_neurons ;
      params[i].w = w + i * nhid * n_inputs ;
      params[i].hid_bias = hid_bias + i * max_neurons ;
      params[i].in_bias = in_bias + i * max_neurons ;
      }


/*
------------------------------------------------------------------------------------------------

   Try some small weight vectors and choose as starter the one with minimum reconstruction error.
   We also initialize all bias vectors to minus half of the weight sum for rough balance.

------------------------------------------------------------------------------------------------
*/

   n_threads = 0 ;                    // Counts threads that are active
   for (i=0 ; i<max_threads ; i++)
      threads[i] = NULL ;

   irand = 0 ;        // Index of try
   empty_slot = -1 ;  // After full, will identify the thread that just completed
   best_err = 1.e40 ;

   for (;;) {         // Main thread loop processes all tries

/*
   Handle user ESCape
*/

      if (irand  &&  (escape_key_pressed  ||  user_pressed_escape ())) { // Make sure at least one tried
         user_pressed_escape () ;
         escape_key_pressed = 0 ;  // Allow subsequent operations
         for (i=0, k=0 ; i<max_threads ; i++) {
            if (threads[i] != NULL)
               threads[k++] = threads[i] ;
            }
         ret_val = WaitForMultipleObjects ( n_threads , threads , TRUE , 12000000 ) ;
         if (ret_val == WAIT_TIMEOUT)
            audit ( "Timeout waiting for computation to finish; problem too large" ) ;
         sprintf ( msg, "RBM_THR1.CPP: User abort; n_threads=%d  k=%d  Wait retval=%d", n_threads, k, ret_val ) ;
         MEMTEXT ( msg ) ;
         for (i=0 ; i<n_threads ; i++)
            CloseHandle ( threads[i] ) ;
         audit ( "" ) ;
         audit ( "WARNING: User pressed ESCape during initial search for RBM starting weights" ) ;
         audit ( "         Results may be substandard" ) ;
         return best_err ;  // Let greedy() continue with whatever we have so far
         }

/*
   Start a new thread if we still have work to do
*/

      if (irand < n_rand) {     // If there are still some to do
         if (empty_slot < 0)    // Negative while we are initially filling the queue
            k = n_threads ;
         else
            k = empty_slot ;

         // Generate the trial weight matrix and bias vectors

         wptr = params[k].w ;
         hid_bias_ptr = params[k].hid_bias ;
         in_bias_ptr = params[k].in_bias ;
         
         diff = 4.0 * unifrand_fast() / sqrt ( sqrt ( (double) n_inputs * nhid ) ) ;

         for (ihid=0 ; ihid<nhid ; ihid++) {
            sum = 0.0 ;

            for (ivis=0 ; ivis<n_inputs ; ivis++) {   // Get all visible weights for this hidden neuron
               wt = diff * (unifrand_fast() - 0.5) ;  // This is symmetric with heavy-ish tails
               wptr[ihid*n_inputs+ivis] = wt ;
               sum += data_mean[ivis] * wt ;          // We'll need this for this hidden neuron's bias
               }

            hid_bias_ptr[ihid] = -sum ;               // Center the distribution
            } // For ihid


         for (ivis=0 ; ivis<n_inputs ; ivis++) {      // Also center the visible
            sum = 0.0 ;
            for (ihid=0 ; ihid<nhid ; ihid++)
               sum += wptr[ihid*n_inputs+ivis] ;            
            in_bias_ptr[ivis] = log ( data_mean[ivis] / (1.0 - data_mean[ivis]) ) - 0.5 * sum ;
            }

         // Start the thread for this trial

         threads[k] = (HANDLE) _beginthreadex ( NULL , 0 , rbm1_wrapper , &params[k] , 0 , NULL ) ;
         if (threads[k] == NULL) {
            audit ( "Internal ERROR: bad thread creation in RBM_THR1" ) ;
            for (i=0 ; i<n_threads ; i++) {
               if (threads[i] != NULL)
                  CloseHandle ( threads[i] ) ;
               }
            return -best_err ;  // Signal greedy() that a catastrophic error occurred
            }
         ++n_threads ;
         ++irand ;
         } // if (irand < n_rand)

      if (n_threads == 0)  // Are we done?
         break ;

/*
   Handle full suite of threads running and more threads to add as soon as some are done.
   Wait for just one thread to finish.
*/

      if (n_threads == max_threads  &&  irand < n_rand) {
         ret_val = WaitForMultipleObjects ( n_threads , threads , FALSE , 12000000 ) ;
         if (ret_val == WAIT_TIMEOUT  ||  ret_val == WAIT_FAILED  ||  ret_val < 0  ||  ret_val >= n_threads) {
            sprintf ( msg, "INTERNAL ERROR!!!  Thread wait 1 failed (%d) in RBM_THR1", ret_val ) ;
            audit ( msg ) ;
            MEMTEXT ( msg ) ;
            if (ret_val == WAIT_TIMEOUT)
               audit ( "Timeout waiting for computation to finish; problem too large" ) ;
            return -best_err ;  // Signal greedy() that a catastrophic error occurred
            }

         error = params[ret_val].crit ;

         // If we just improved, save the best-so-far parameters

         if (error < best_err) {
            best_err = error ;
            for (ihid=0 ; ihid<nhid ; ihid++) {
               hid_bias_best[ihid] = params[ret_val].hid_bias[ihid] ;
               for (ivis=0 ; ivis<n_inputs ; ivis++)
                  w_best[ihid*n_inputs+ivis] = params[ret_val].w[ihid*n_inputs+ivis] ;
               }

            for (ivis=0 ; ivis<n_inputs ; ivis++)
               in_bias_best[ivis] = params[ret_val].in_bias[ivis] ;
            }

#if RECON_ERR_XENT
         sprintf ( msg, "%d of %d  XENT=%7.4lf  Best=%7.4lf",
                   irand-max_threads+1, n_rand, error / (n_inputs * nc),
                   best_err / (n_inputs * nc) ) ;
#else
         sprintf ( msg, "%d of %d  RMS Err=%7.4lf  Best=%7.4lf",
                   irand-max_threads+1, n_rand, sqrt ( error / (n_inputs * nc) ),
                   sqrt ( best_err / (n_inputs * nc) ) ) ;
#endif

         empty_slot = ret_val ;
         CloseHandle ( threads[empty_slot] ) ;
         threads[empty_slot] = NULL ;
         --n_threads ;
         }

/*
   Handle all work has been started and now we are just waiting for threads to finish
*/

      else if (irand == n_rand) {
         ret_val = WaitForMultipleObjects ( n_threads , threads , TRUE , 1200000 ) ;
         if (ret_val == WAIT_TIMEOUT  ||  ret_val == WAIT_FAILED  ||  ret_val < 0  ||  ret_val >= n_threads) {
            sprintf ( msg, "INTERNAL ERROR!!!  Thread wait 2 failed (%d) in RBM_THR1.CPP", ret_val ) ;
            audit ( msg ) ;
            MEMTEXT ( msg ) ;
            if (ret_val == WAIT_TIMEOUT)
               audit ( "Timeout waiting for computation to finish; problem too large" ) ;
            return -best_err ;  // Signal greedy() that a catastrophic error occurred
            }

         for (i=0 ; i<n_threads ; i++) {

            error = params[i].crit ;

            // If we just improved, save the best-so-far parameters

            if (error < best_err) {
               for (ihid=0 ; ihid<nhid ; ihid++) {
                  hid_bias_best[ihid] = params[i].hid_bias[ihid] ;
                  best_err = error ;
                  for (ivis=0 ; ivis<n_inputs ; ivis++)
                     w_best[ihid*n_inputs+ivis] = params[i].w[ihid*n_inputs+ivis] ;
                  }
   
               for (ivis=0 ; ivis<n_inputs ; ivis++)
                  in_bias_best[ivis] = params[i].in_bias[ivis] ;
               }

            CloseHandle ( threads[i] ) ;

#if RECON_ERR_XENT
            sprintf ( msg, "%d of %d  XENT=%7.4lf  Best=%7.4lf",
                      n_rand-n_threads+i+1, n_rand, error / (n_inputs * nc),
                      best_err / (n_inputs * nc) ) ;
#else
            sprintf ( msg, "%d of %d  RMS Err=%7.4lf  Best=%7.4lf",
                      n_rand-n_threads+i+1, n_rand, sqrt ( error / (n_inputs * nc) ),
                      sqrt ( best_err / (n_inputs * nc) ) ) ;
#endif
            } // For i, processing all threads just returned
         break ;
         } // Waiting for final threads to finish
      } // Endless loop which threads computation of criterion for all random tries

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

   return best_err / (nc * n_inputs) ;
}