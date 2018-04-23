/******************************************************************************/
/*                                                                            */
/*  MLFN_THR - MLFN routines modified for threading                           */
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

   Local routine to compute activation

--------------------------------------------------------------------------------
*/


void activity (
   double *input ,   // This neuron's input vector, ninputs long
   double *coefs ,   // Weight vector, ninputs+1 long (bias is at end)
   double *output ,  // Achieved activation of this neuron
   int ninputs ,     // Number of inputs
   int outlin        // Activation function is identity if nonzero, else logistic
   )
{
   double sum ;

   sum = dotprod ( ninputs , input , coefs ) ;
   sum += coefs[ninputs] ;      // Bias term

   if (outlin)
      *output = sum ;
    else
      *output = 1.0 / (1.0 + exp(-sum)) ;
}


/*
--------------------------------------------------------------------------------

   trial - Compute the output for a given input by evaluating network
           This is the Model version, callable from outside here.

--------------------------------------------------------------------------------
*/

void Model::trial ( double *input )
{
   int i, ilayer ;
   double sum ;

   for (ilayer=0 ; ilayer<n_all ; ilayer++) {

      if (ilayer == 0  &&  n_all == 1) {                 // Direct input to output?
         for (i=0 ; i<ntarg ; i++)
            activity ( input , final_layer_weights+i*(n_model_inputs+1) , outputs+i , n_model_inputs , 1 ) ;
         }

      else if (ilayer == 0) {                            // First hidden layer?
         for (i=0 ; i<nhid_all[ilayer] ; i++)
            activity ( input , weights_opt[ilayer]+i*(n_model_inputs+1) , hid_act[ilayer]+i , n_model_inputs , 0 ) ;
         }

      else if (ilayer < n_all-1) {                       // Subsequent hidden layer?
         for (i=0 ; i<nhid_all[ilayer] ; i++)
            activity ( hid_act[ilayer-1] , weights_opt[ilayer]+i*(nhid_all[ilayer-1]+1) ,
                       hid_act[ilayer]+i , nhid_all[ilayer-1] , 0 );
         }

      else {                                             // Final layer
         for (i=0 ; i<ntarg ; i++)
            activity ( hid_act[ilayer-1] , final_layer_weights+i*(nhid_all[ilayer-1]+1) ,
                       outputs+i , nhid_all[ilayer-1] , 1 );
         }
      }

   if (classifier) {  // Classifier is always SoftMax
      sum = 0.0 ;
      for (i=0 ; i<ntarg ; i++) {
         if (outputs[i] < 300.0)
            outputs[i] = exp ( outputs[i] ) ;
         else
            outputs[i] = exp ( 300.0 ) ;
         sum += outputs[i] ;
         }
      for (i=0 ; i<ntarg ; i++)
         outputs[i] /= sum ;
      }
}


/*
--------------------------------------------------------------------------------

   trial_thr - Compute the output for a given input by evaluating network
               This is a strictly local version for threading.

--------------------------------------------------------------------------------
*/

static void trial_thr (
   double *input ,                 // Input vector n_model_inputs long
   int n_all ,                     // Number of layers, including output, not including input
   int n_model_inputs ,            // Number of inputs to the model
   double *outputs ,               // Output vector of the model
   int ntarg ,                     // Number of outputs
   int *nhid_all ,                 // nhid_all[i] is the number of hidden neurons in hidden layer i
   double *weights_opt[] ,         // weights_opt[i] points to the weight vector for hidden layer i
   double *hid_act[] ,             // hid_act[i] points to the vector of activations of hidden layer i
   double *final_layer_weights ,   // Weights of final layer
   int classifier                  // If nonzero use SoftMax output; else use linear output
   )
{
   int i, ilayer ;
   double sum ;

   for (ilayer=0 ; ilayer<n_all ; ilayer++) {

      if (ilayer == 0  &&  n_all == 1) {        // Direct input to output?
         for (i=0 ; i<ntarg ; i++)
            activity ( input , final_layer_weights+i*(n_model_inputs+1) , outputs+i , n_model_inputs , 1 ) ;
         }

      else if (ilayer == 0) {                   // First hidden layer?
         for (i=0 ; i<nhid_all[ilayer] ; i++)
            activity ( input , weights_opt[ilayer]+i*(n_model_inputs+1) , hid_act[ilayer]+i , n_model_inputs , 0 ) ;
         }

      else if (ilayer < n_all-1) {              // Subsequent hidden layer?
         for (i=0 ; i<nhid_all[ilayer] ; i++)
            activity ( hid_act[ilayer-1] , weights_opt[ilayer]+i*(nhid_all[ilayer-1]+1) ,
                       hid_act[ilayer]+i , nhid_all[ilayer-1] , 0 );
         }

      else {                                    // Final layer
         for (i=0 ; i<ntarg ; i++)
            activity ( hid_act[ilayer-1] , final_layer_weights+i*(nhid_all[ilayer-1]+1) ,
                       outputs+i , nhid_all[ilayer-1] , 1 );
         }
      }

   if (classifier) {  // Classifier is always SoftMax
      sum = 0.0 ;
      for (i=0 ; i<ntarg ; i++) {  // For all outputs
         if (outputs[i] < 300.0)
            outputs[i] = exp ( outputs[i] ) ;
         else
            outputs[i] = exp ( 300.0 ) ;
         sum += outputs[i] ;
         }
      for (i=0 ; i<ntarg ; i++)
         outputs[i] /= sum ;
      }
}


/*
------------------------------------------------------------------------------------------------

   Threaded routine that cumulates error for a batch

------------------------------------------------------------------------------------------------
*/

static double batch_error (
   int istart ,                    // Index of starting case in input matrix
   int istop ,                     // And one past last case
   int max_neurons ,               // Number of columns in input matrix; max exceed n_model_inputs
   double *input ,                 // Input matrix; each case is max_neurons long
   int n_all ,                     // Number of layers, including output, not including input
   int n_model_inputs ,            // Number of inputs to the model; Input matrix may have more columns
   double *outputs ,               // Output vector of the model; used as work vector here
   int ntarg ,                     // Number of outputs
   int *nhid_all ,                 // nhid_all[i] is the number of hidden neurons in hidden layer i
   double *weights_opt[] ,         // weights_opt[i] points to the weight vector for hidden layer i
   double *hid_act[] ,             // hid_act[i] points to the vector of activations of hidden layer i
   double *final_layer_weights ,   // Weights of final layer
   double *targets ,               // Target matrix; each case is ntarg long
   int classifier                  // If nonzero use SoftMax output; else use linear output
   )
{
   int i, icase, imax ;
   double err, tot_err, *dptr, diff, tmax ;

   tot_err = 0.0 ;  // Total error will be cumulated here

   for (icase=istart ; icase<istop ; icase++) {  // Do all samples

      dptr = input + icase * max_neurons ; // Point to this sample
      trial_thr ( dptr , n_all , n_model_inputs , outputs ,  ntarg , nhid_all ,
                  weights_opt , hid_act , final_layer_weights , classifier ) ;
      err = 0.0 ;

      dptr = targets + icase * ntarg ;

      if (classifier) {               // SoftMax
         tmax = -1.e30 ;
         for (i=0 ; i<ntarg ; i++) {  // Find the true class as that having max target
            if (*dptr > tmax) {
               imax = i ;
               tmax = *dptr ;
               }
            ++dptr ;
            }
         err = -log ( outputs[imax] + 1.e-30 ) ;
         }

      else {
         for (i=0 ; i<ntarg ; i++) {
            diff = *dptr++ - outputs[i] ;
            err += diff * diff ;
            }
         }

      tot_err += err ;
      } // for all cases

   return tot_err ;
}


/*
--------------------------------------------------------------------------------

   batch_gradient - Cumulate the gradient for a given subset of inputs

   Note: grad is all gradients as a vector, and grad_ptr[ilayer] points to
         the entry in grad that is for the first weight in a layer

--------------------------------------------------------------------------------
*/


static double batch_gradient (
   int istart ,                    // Index of starting case in input matrix
   int istop ,                     // And one past last case
   double *input ,                 // Input matrix; each case is max_neurons long
   double *targets ,               // Target matrix; each case is ntarg long
   int n_all ,                     // Number of layers, including output, not including input
   int n_all_weights ,             // Total number of weights, including final layer and all bias terms
   int n_model_inputs ,            // Number of inputs to the model; Input matrix may have more columns
   double *outputs ,               // Output vector of the model; used as work vector here
   int ntarg ,                     // Number of outputs
   int *nhid_all ,                 // nhid_all[i] is the number of hidden neurons in hidden layer i
   double *weights_opt[] ,         // weights_opt[i] points to the weight vector for hidden layer i
   double *hid_act[] ,             // hid_act[i] points to the vector of activations of hidden layer i
   int max_neurons ,               // Number of columns in input matrix; may exceed n_model_inputs
   double *this_delta ,            // Delta for the current layer
   double *prior_delta ,           // And saved for use in the prior (next to be processed) layer
   double **grad_ptr ,             // grad_ptr[i] points to gradient for layer i
   double *final_layer_weights ,   // Weights of final layer
   double *grad ,                  // All computed gradients, strung out as a single long vector
   int classifier                  // If nonzero use SoftMax output; else use linear output
   )
{
   int i, j, icase, ilayer, nprev, nthis, nnext, imax ;
   double diff, *dptr, error, *targ_ptr, *prevact, *gradptr, delta, *nextcoefs, tmax ;

   for (i=0 ; i<n_all_weights ; i++)  // Zero gradient for summing
      grad[i] = 0.0 ;                 // All layers are strung together here

   error = 0.0 ;  // Will cumulate total error here

   for (icase=istart ; icase<istop ; icase++) {

      dptr = input + icase * max_neurons ; // Point to this sample
      trial_thr ( dptr , n_all , n_model_inputs , outputs ,  ntarg , nhid_all ,
                  weights_opt , hid_act , final_layer_weights , classifier ) ;

      targ_ptr = targets + icase * ntarg ;

      if (classifier) {               // SoftMax
         tmax = -1.e30 ;
         for (i=0 ; i<ntarg ; i++) {  // Find the true class as that having max target
            if (targ_ptr[i] > tmax) { // To save a small amount of time we could precompute this
               imax = i ;
               tmax = targ_ptr[i] ;
               }
            this_delta[i] = targ_ptr[i] - outputs[i] ; // Neg deriv of cross entropy wrt input (logit) i
            }
         error -= log ( outputs[imax] + 1.e-30 ) ;
         }

      else {
         for (i=0 ; i<ntarg ; i++) {
            diff = outputs[i] - targ_ptr[i] ;
            error += diff * diff ;
            this_delta[i] = -2.0 * diff ; // Neg deriv of squared error wrt input to neuron i
            }
         }

/*
   Cumulate output gradient
*/

      if (n_all == 1) {                           // No hidden layer
         nprev = n_model_inputs ;                 // Number of inputs to the output layer
         prevact = input + icase * max_neurons ;  // Point to this sample
         }
      else {
         nprev = nhid_all[n_all-2] ;        // n_all-2 is the last hidden layer
         prevact = hid_act[n_all-2] ;       // Point to layer feeding the output layer
         }
      gradptr = grad_ptr[n_all-1] ;         // Point to output gradient in grand gradient vector
      for (i=0 ; i<ntarg ; i++) {           // For all output neurons
         delta = this_delta[i] ;            // Neg deriv of criterion wrt logit
         for (j=0 ; j<nprev ; j++)
            *gradptr++ += delta * prevact[j] ; // Cumulate for all training cases
         *gradptr++ += delta ;              // Bias activation is always 1
         }

      nnext = ntarg ;                       // Prepare for moving back one layer
      nextcoefs = final_layer_weights ;

/*
   Cumulate hidden gradients
*/

      for (ilayer=n_all-2 ; ilayer>=0 ; ilayer--) {   // For each hidden layer, working backwards
         nthis = nhid_all[ilayer] ;        // Number of neurons in this hidden layer
         gradptr = grad_ptr[ilayer] ;      // Point to gradient for this layer
         for (i=0 ; i<nthis ; i++) {       // For each neuron in this layer
            delta = 0.0 ;
            for (j=0 ; j<nnext ; j++)
               delta += this_delta[j] * nextcoefs[j*(nthis+1)+i] ;
            delta *= hid_act[ilayer][i] * (1.0 - hid_act[ilayer][i]) ;  // Derivative
            prior_delta[i] = delta ;                    // Save it for the next layer back
            if (ilayer == 0) {                          // First hidden layer?
               prevact = input + icase * max_neurons ;  // Point to this sample
               for (j=0 ; j<n_model_inputs ; j++)
                  *gradptr++ += delta * prevact[j] ;
               }
            else {      // There is at least one more hidden layer prior to this one
               prevact = hid_act[ilayer-1] ;
               for (j=0 ; j<nhid_all[ilayer-1] ; j++)
                  *gradptr++ += delta * prevact[j] ;
               }
            *gradptr++ += delta ;   // Bias activation is always 1
            }  // For all neurons in this hidden layer

         for (i=0 ; i<nthis ; i++)           // These will be delta for the next layer back
            this_delta[i] = prior_delta[i] ;

         nnext = nhid_all[ilayer] ;          // Prepare for the next layer back
         nextcoefs = weights_opt[ilayer] ;
         }  // For all layers, working backwards

      } // for all cases

   return error ;  // MSE or negative log likelihood
}


/*
--------------------------------------------------------------------------------

   Thread stuff...
      Structure for passing information to/from threaded code
      Threaded code is called by the main subroutine

--------------------------------------------------------------------------------
*/

typedef struct {
   int istart ;            // First case in this batch
   int istop ;             // One past last case
   int classifier ;
   int max_neurons ;
   int n_all ;
   int n_model_inputs ;
   int ntarg ;
   int *nhid_all ;
   double *input ;
   double *outputs ;
   double **weights_opt ;
   double **hid_act ;
   double *final_layer_weights ;
   double *target ;
   double error ;
} ERR_THR_PARAMS ;

static unsigned int __stdcall batch_error_wrapper ( LPVOID dp )
{
((ERR_THR_PARAMS *) dp)->error = batch_error (
                          ((ERR_THR_PARAMS *) dp)->istart ,
                          ((ERR_THR_PARAMS *) dp)->istop ,
                          ((ERR_THR_PARAMS *) dp)->max_neurons ,
                          ((ERR_THR_PARAMS *) dp)->input ,
                          ((ERR_THR_PARAMS *) dp)->n_all ,
                          ((ERR_THR_PARAMS *) dp)->n_model_inputs ,
                          ((ERR_THR_PARAMS *) dp)->outputs ,
                          ((ERR_THR_PARAMS *) dp)->ntarg ,
                          ((ERR_THR_PARAMS *) dp)->nhid_all ,
                          ((ERR_THR_PARAMS *) dp)->weights_opt ,
                          ((ERR_THR_PARAMS *) dp)->hid_act ,
                          ((ERR_THR_PARAMS *) dp)->final_layer_weights ,
                          ((ERR_THR_PARAMS *) dp)->target ,
                          ((ERR_THR_PARAMS *) dp)->classifier ) ;
   return 0 ;
}


typedef struct {
   int istart ;            // First case in this batch
   int istop ;             // One past last case
   int classifier ;
   int n_all ;
   int n_all_weights ;
   int n_model_inputs ;
   int ntarg ;
   int *nhid_all ;
   int max_neurons ;
   double *input ;
   double *targets ;
   double *outputs ;
   double **weights_opt ;
   double **hid_act ;
   double *this_delta ;
   double *prior_delta ;
   double **grad_ptr ;
   double *final_layer_weights ;
   double *grad ;
   double error ;
} GRAD_THR_PARAMS ;

static unsigned int __stdcall batch_gradient_wrapper ( LPVOID dp )
{
((GRAD_THR_PARAMS *) dp)->error = batch_gradient (
                          ((GRAD_THR_PARAMS *) dp)->istart ,
                          ((GRAD_THR_PARAMS *) dp)->istop ,
                          ((GRAD_THR_PARAMS *) dp)->input ,
                          ((GRAD_THR_PARAMS *) dp)->targets ,
                          ((GRAD_THR_PARAMS *) dp)->n_all ,
                          ((GRAD_THR_PARAMS *) dp)->n_all_weights ,
                          ((GRAD_THR_PARAMS *) dp)->n_model_inputs ,
                          ((GRAD_THR_PARAMS *) dp)->outputs ,
                          ((GRAD_THR_PARAMS *) dp)->ntarg ,
                          ((GRAD_THR_PARAMS *) dp)->nhid_all ,
                          ((GRAD_THR_PARAMS *) dp)->weights_opt ,
                          ((GRAD_THR_PARAMS *) dp)->hid_act ,
                          ((GRAD_THR_PARAMS *) dp)->max_neurons ,
                          ((GRAD_THR_PARAMS *) dp)->this_delta ,
                          ((GRAD_THR_PARAMS *) dp)->prior_delta ,
                          ((GRAD_THR_PARAMS *) dp)->grad_ptr ,
                          ((GRAD_THR_PARAMS *) dp)->final_layer_weights ,
                          ((GRAD_THR_PARAMS *) dp)->grad ,
                          ((GRAD_THR_PARAMS *) dp)->classifier ) ;
   return 0 ;
}

/*
--------------------------------------------------------------------------------

   gradient() - Gradient for entire model

--------------------------------------------------------------------------------
*/

double Model::gradient_thr (
   int nc ,              // Number of cases
   double *input ,       // Inputs, nc rows and max_neurons columns, of which the first n_model_inputs are used
   double *target ,      // Targets, nc rows and ntarg columns
   double *grad          // Concatenated gradient vector, which is computed here
   )
{
   int i, j, ilayer, ineuron, ivar, n, istart, istop, n_done, ithread ;
   int n_in_batch, n_threads, ret_val, nin_this_layer ;
   int k=0 ;   // Can remove this when final assert is assured
   double error, *wptr, *gptr, factor, *hid_act_ptr[MAX_THREADS][MAX_LAYERS], *grad_ptr_ptr[MAX_THREADS][MAX_LAYERS] ;
   double wpen ;
   char msg[256] ;
   GRAD_THR_PARAMS params[MAX_THREADS] ;
   HANDLE threads[MAX_THREADS] ;

   wpen = TrainParams.wpen / n_all_weights ;

/*
   Compute length of grad vector and gradient positions in it.
   If I ever make grad a permanent member of the model, rather than a temporary
   allocated in CONJGRAD.CPP, I can do this in the constructor.  But it's very fast.
*/

   gptr = grad ;  // CONJGRAD.CPP allocated this n_all_weights * max_threads long

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

   for (i=0 ; i<max_threads ; i++) {
      params[i].input = input ;
      params[i].targets = targets ;
      params[i].n_all = n_all ;
      params[i].n_all_weights = n_all_weights ;
      params[i].n_model_inputs = n_model_inputs ;
      params[i].ntarg = ntarg ;
      params[i].nhid_all = nhid_all ;
      params[i].max_neurons = max_neurons ;
      params[i].weights_opt = weights_opt ;
      params[i].final_layer_weights = final_layer_weights ;

      params[i].this_delta = this_layer + i * max_neurons ;
      params[i].prior_delta = prior_layer + i * max_neurons ;
      params[i].outputs = outputs + i * ntarg ;
      params[i].grad = grad + i * n_all_weights ;
      for (j=0 ; j<n_all ; j++) {
         hid_act_ptr[i][j] = hid_act[j] + i * max_neurons ;
         grad_ptr_ptr[i][j] = grad_ptr[j] + i * n_all_weights ;
         }
      params[i].hid_act = hid_act_ptr[i] ;
      params[i].grad_ptr = grad_ptr_ptr[i] ;
      params[i].classifier = classifier ;
      }

/*
------------------------------------------------------------------------------------------------

   Batch loop uses a different thread for each batch

------------------------------------------------------------------------------------------------
*/

   n_threads = max_threads ;    // Try to use as many as possible
   if (nc / n_threads < 100)    // But because threads have overhead
      n_threads = 1 ;           // Avoid using them if the batch is small

   istart = 0 ;         // Batch start = training data start
   n_done = 0 ;         // Number of training cases done in this epoch so far

   for (ithread=0 ; ithread<n_threads ; ithread++) {
      n_in_batch = (nc - n_done) / (n_threads - ithread) ;  // Cases left to do / batches left to do
      istop = istart + n_in_batch ;                         // Stop just before this index

      // Set the pointers that vary with the batch

      params[ithread].istart = istart ;
      params[ithread].istop = istop ;

      threads[ithread] = (HANDLE) _beginthreadex ( NULL , 0 , batch_gradient_wrapper , &params[ithread] , 0 , NULL ) ;
      if (threads[ithread] == NULL) {
         audit ( "Internal ERROR: bad thread creation in MLFN_THR" ) ;
         for (i=0 ; i<n_threads ; i++) {
            if (threads[i] != NULL)
               CloseHandle ( threads[i] ) ;
            }
         return -1.e40 ;
         }

      n_done += n_in_batch ;
      istart = istop ;
      } // For all threads / batches

/*
   Wait for threads to finish, and then cumulate all results into [0]
*/

   ret_val = WaitForMultipleObjects ( n_threads , threads , TRUE , 1200000 ) ;
   if (ret_val == WAIT_TIMEOUT  ||  ret_val == WAIT_FAILED  ||  ret_val < 0  ||  ret_val >= n_threads) {
      sprintf ( msg, "INTERNAL ERROR!!!  Thread wait 1 failed (%d) in MLFN_THR.CPP", ret_val ) ;
      audit ( msg ) ;
      MEMTEXT ( msg ) ;
      if (ret_val == WAIT_TIMEOUT)
         audit ( "Timeout waiting for computation to finish; problem too large" ) ;
      return -1.e40 ;
      }

   CloseHandle ( threads[0] ) ;
   for (ithread=1 ; ithread<n_threads ; ithread++) {
      params[0].error += params[ithread].error ;
      for (i=0 ; i<n_all_weights ; i++)
         params[0].grad[i] += params[ithread].grad[i] ;
      CloseHandle ( threads[ithread] ) ;
      }


/*
   Find the mean per presentation.  Also, compensate for nout if that was
   not done implicitly in the error computation.
*/

   factor = 1.0 / (nc * ntarg) ;

   error = factor * params[0].error ;

   for (i=0 ; i<n_all_weights ; i++)
      grad[i] = factor * params[0].grad[i] ;   // Note that grad and params[0].grad are the same!


/*
   Deal with weight penalty
   First block of code does hidden layers, second does output layer
*/

   penalty = 0.0 ;
   nin_this_layer = n_model_inputs ;
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

   penalty *= wpen ;
   return error + penalty ;
}


/*
--------------------------------------------------------------------------------

   trial_error_thr - Compute the mean square error for the entire training set

--------------------------------------------------------------------------------
*/

double Model::trial_error_thr (
   int nc ,
   double *input ,
   double *target
   )
{
   int i, j, ineuron, ivar, ithread, n_threads, n_in_batch, n_done, istart, istop, ret_val ;
   int ilayer, nin_this_layer ;
   double error, *wptr, *hid_act_ptr[MAX_THREADS][MAX_LAYERS], wpen ;
   char msg[256] ;
   ERR_THR_PARAMS params[MAX_THREADS] ;
   HANDLE threads[MAX_THREADS] ;

   wpen = TrainParams.wpen / n_all_weights ;

/*
   Initialize parameters that will not change for threads.
*/

   for (i=0 ; i<max_threads ; i++) {
      params[i].ntarg = ntarg ;
      params[i].nhid_all = nhid_all ;
      params[i].max_neurons = max_neurons ;
      params[i].n_all = n_all ;
      params[i].n_model_inputs = n_model_inputs ;
      params[i].input = input ;
      params[i].weights_opt = weights_opt ;
      params[i].final_layer_weights = final_layer_weights ;
      params[i].target = target ;
      params[i].outputs = outputs + i * ntarg ;
      for (j=0 ; j<n_all ; j++)
         hid_act_ptr[i][j] = hid_act[j] + i * max_neurons ;
      params[i].hid_act = hid_act_ptr[i] ;
      params[i].classifier = classifier ;
      }


/*
------------------------------------------------------------------------------------------------

   Batch loop uses a different thread for each batch

------------------------------------------------------------------------------------------------
*/

   n_threads = max_threads ;    // Try to use as many as possible
   if (nc / n_threads < 100)    // But because threads have overhead
      n_threads = 1 ;           // Avoid using them if the batch is small

   istart = 0 ;         // Batch start = training data start
   n_done = 0 ;         // Number of training cases done in this epoch so far

   for (ithread=0 ; ithread<n_threads ; ithread++) {
      n_in_batch = (nc - n_done) / (n_threads - ithread) ;  // Cases left to do / batches left to do
      istop = istart + n_in_batch ;                         // Stop just before this index

      // Set the pointers that vary with the batch

      params[ithread].istart = istart ;
      params[ithread].istop = istop ;

      threads[ithread] = (HANDLE) _beginthreadex ( NULL , 0 , batch_error_wrapper , &params[ithread] , 0 , NULL ) ;
      if (threads[ithread] == NULL) {
         audit ( "Internal ERROR: bad thread creation in MLFN_THR" ) ;
         for (i=0 ; i<n_threads ; i++) {
            if (threads[i] != NULL)
               CloseHandle ( threads[i] ) ;
            }
         return -1.e40 ;
         }

      n_done += n_in_batch ;
      istart = istop ;
      } // For all threads / batches

/*
   Wait for threads to finish
*/

   ret_val = WaitForMultipleObjects ( n_threads , threads , TRUE , 1200000 ) ;
   if (ret_val == WAIT_TIMEOUT  ||  ret_val == WAIT_FAILED  ||  ret_val < 0  ||  ret_val >= n_threads) {
      sprintf ( msg, "INTERNAL ERROR!!!  Thread wait 2 failed (%d) in MLFN_THR.CPP", ret_val ) ;
      audit ( msg ) ;
      MEMTEXT ( msg ) ;
      if (ret_val == WAIT_TIMEOUT)
         audit ( "Timeout waiting for computation to finish; problem too large" ) ;
      return -1.e40 ;
      }

   error = 0.0 ;        // Cumulates squared reproduction error or negative log likelihood (for classifier)
   for (ithread=0 ; ithread<n_threads ; ithread++) {
      error += params[ithread].error ;
      CloseHandle ( threads[ithread] ) ;
      }


   error /= nc * ntarg ;


/*
   Deal with weight penalty
*/

   penalty = 0.0 ;
   nin_this_layer = n_model_inputs ;
   for (ilayer=0 ; ilayer<n_all-1 ; ilayer++) {  // Do all hidden layers
      for (ineuron=0 ; ineuron<nhid_all[ilayer] ; ineuron++) {
         wptr = weights_opt[ilayer]+ineuron*(nin_this_layer+1) ;  // Weights for this neuron in this layer
         for (ivar=0 ; ivar<nin_this_layer ; ivar++)
            penalty += wptr[ivar] * wptr[ivar] ;
         }
      nin_this_layer = nhid_all[ilayer] ;
      }

   for (ineuron=0 ; ineuron<ntarg ; ineuron++) {
      wptr = final_layer_weights + ineuron * n_final_layer_weights ;
      for (ivar=0 ; ivar<nin_this_layer ; ivar++)
         penalty += wptr[ivar] * wptr[ivar] ;
      }

   penalty *= wpen ;
   return error + penalty ;
}