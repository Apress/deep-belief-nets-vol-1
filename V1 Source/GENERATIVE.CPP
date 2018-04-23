/************************************************************/
/*                                                          */
/*  GENERATIVE - Display generative samples                 */
/*                                                          */
/*  Computation fragment only; no display                   */
/*                                                          */
/************************************************************/


class GenerativeChild {

public:
   GenerativeChild ( int first_case , int nrows , int ncols , int nchain ) ;
   ~GenerativeChild () ;

   int ok ;
   int first_case ;
   int nrows ;
   int ncols ;
   int nchain ;
   DIBimage *dib ;       /* The image is here */
} ;


/*
--------------------------------------------------------------------------------

   Workhorse routine that computes a single generative sample

--------------------------------------------------------------------------------
*/

static void gen_threaded (
   int nvis ,                // Number of inputs to the first (bottom) layer
   int max_neurons ,         // Maximum number of neurons in any layer, as well as nvis
   int n_unsup ,             // Number of unsupervised layers
   int *nhid_unsup ,         // N_unsup vector containing the number of hidden neurons in each layer
   double **weights_unsup ,  // N_unsup pointers to weight matrices, each being nhid sets of nvis weights
   double *in_bias ,         // Input bias vectors; n_unsup sets of max_neurons each
   double *hid_bias ,        // Hidden bias vectors; n_unsup sets of max_neurons each
   int nchain ,              // Length of Gibbs chain, 0 to return raw data
   int input_vis ,           // Start with visible (as opposed to hidden)?
   double *workvec1 ,        // Work vector max_neurons long, also inputs starting case if input_vis
   double *workvec2 ,        // Work vector max_neurons long, also inputs starting hidden if ! input_vis
   unsigned char *image      // Computed image, 0-255 returned here
   )
{
   int i, k, ichain, ivis, nin, ihid, nhid, i_layer, randnum ;
   double *vis_layer, *hid_layer, *w, *wptr, *ibptr, *hbptr, sum, Q, frand ;

   vis_layer = workvec1 ;
   hid_layer = workvec2 ;
      
   // Either a training set image is in workvec1 (input_vis),
   // or a hidden weight vector is in workvec2 (! input_vis).

   if (nchain == 0) {   // User wants original image?  This overrides input_vis.
      for (i=0 ; i<nvis ; i++)
         image[i] = (unsigned char) (255.9999 * vis_layer[i]) ; 
      return ;
      }

   if (input_vis) {

      randnum = 1 ;               // Get a somewhat random seed
      for (i=0 ; i<nvis ; i++) {
         if (vis_layer[i] > 0.5)
            ++randnum ;
         }
            
   // Propagate up until we reach the RBM

      nin = nvis ;
      for (i_layer=0 ; i_layer<n_unsup-1 ; i_layer++) {
         nhid = nhid_unsup[i_layer] ;
         w = weights_unsup[i_layer] ;
         hbptr = hid_bias + i_layer * max_neurons ;
          for (ihid=0 ; ihid<nhid ; ihid++) {
            wptr = w + ihid * nin ;          // Weight vector for this neuron
            sum = hbptr[ihid] ;              // This hidden neuron's bias
            for (ivis=0 ; ivis<nin ; ivis++)
               sum += wptr[ivis] * vis_layer[ivis] ;
            hid_layer[ihid] = 1.0 / (1.0 + exp(-sum)) ;
            }
         nin = nhid ;
         if (vis_layer == workvec1) {
            vis_layer = workvec2 ;
            hid_layer = workvec1 ;
            }
         else {
            vis_layer = workvec1 ;
            hid_layer = workvec2 ;
            }
         } // For i_layer, propagating up until the RBM
      } // If input_vis

   else { // Not input_vis, so user is inputting hidden layer of RBM
      randnum = 1 ;               // Get a somewhat random seed
      for (i=0 ; i<nhid_unsup[n_unsup-1] ; i++) {
         if (hid_layer[i] > 0.5)
            ++randnum ;
         }

      if (n_unsup == 1)
         nin = nvis ;
      else
         nin = nhid_unsup[n_unsup-2] ;
      } // If not input_vis


   // Gibbs chain in the RBM

   nhid = nhid_unsup[n_unsup-1] ;
   w = weights_unsup[n_unsup-1] ;
   hbptr = hid_bias + (n_unsup-1) * max_neurons ;
   ibptr = in_bias + (n_unsup-1) * max_neurons ;

   for (ichain=0 ; ichain<nchain ; ichain++) {

      if (ichain  ||  input_vis) {           // Skip first visible-to-hidden if user inputs hidden
         for (ihid=0 ; ihid<nhid ; ihid++) { // Visible to hidden, with sampling
            wptr = w + ihid * nin ;          // Weight vector for this neuron
            sum = hbptr[ihid] ;              // This hidden neuron's bias
            for (ivis=0 ; ivis<nin ; ivis++)
               sum += wptr[ivis] * vis_layer[ivis] ;
            Q = 1.0 / (1.0 + exp(-sum)) ;
            k = randnum / IQ ;
            randnum = IA * (randnum - k * IQ) - IR * k ;
            if (randnum < 0)
               randnum += IM ;
            frand = AM * randnum ;
            hid_layer[ihid] = (frand < Q) ? 1.0 : 0.0 ;
            }
         }
   
      for (ivis=0 ; ivis<nin ; ivis++) {   // Hidden to visible, without sampling
         sum = ibptr[ivis] ;
         for (ihid=0 ; ihid<nhid ; ihid++)
            sum += w[ihid*nin+ivis] * hid_layer[ihid] ;
         vis_layer[ivis] = 1.0 / (1.0 + exp(-sum)) ;
         }

      if (escape_key_pressed)
         break ;

      } // For ichain

   // The Gibbs chain is complete.  Work back down to the input.

   for (i_layer=n_unsup-2 ; i_layer>=0 ; i_layer--) {
      nhid = nin ;
      assert ( nhid == nhid_unsup[i_layer] ) ;
      if (i_layer == 0)
         nin = nvis ;
      else
         nin = nhid_unsup[i_layer-1] ;
      w = weights_unsup[i_layer] ;
      ibptr = in_bias + i_layer * max_neurons ;

      if (vis_layer == workvec1) {
         vis_layer = workvec2 ;
         hid_layer = workvec1 ;
         }
      else {
         vis_layer = workvec1 ;
         hid_layer = workvec2 ;
         }

      for (ivis=0 ; ivis<nin ; ivis++) {   // Hidden to visible, without sampling
         sum = ibptr[ivis] ;
         for (ihid=0 ; ihid<nhid ; ihid++)
            sum += w[ihid*nin+ivis] * hid_layer[ihid] ;
         vis_layer[ivis] = 1.0 / (1.0 + exp(-sum)) ;
         }
      } // For i_layer, propagating down until the data input

   for (i=0 ; i<nvis ; i++)
      image[i] = (unsigned char) (255.9999 * vis_layer[i]) ; 
}


/*
--------------------------------------------------------------------------------

   Thread stuff...
      Structure for passing information to/from threaded code
      Threaded code called by the main subroutine

--------------------------------------------------------------------------------
*/

typedef struct {
   int nvis ;                // Number of inputs to the first (bottom) layer
   int max_neurons ;         // Maximum number of neurons in any layer, as well as nin
   int n_unsup ;             // Number of unsupervised layers to greedily train
   int *nhid_unsup ;         // N_unsup vector containing the number of hidden neurons in each layer
   double **weights_unsup ;  // N_unsup pointers to computed weight matrices, each being nhid sets of n_inputs weights
   double *in_bias ;         // Input bias vectors; n_unsup sets of max_neurons each
   double *hid_bias ;        // Hidden bias vectors; n_unsup sets of max_neurons each
   int nchain ;              // Length of Gibbs chain, 0 to return raw data
   int input_vis ;           // Start with visible (as opposed to hidden)?
   double *workvec1 ;        // Work vector max_neurons long, also inputs starting case
   double *workvec2 ;        // Work vector max_neurons long
   unsigned char *image ;    // Computed image, 0-255 returned here
} RBM_GENER_PARAMS ;

static unsigned int __stdcall gen_wrapper ( LPVOID dp )
{
   gen_threaded (
       ((RBM_GENER_PARAMS *) dp)->nvis ,
       ((RBM_GENER_PARAMS *) dp)->max_neurons ,
       ((RBM_GENER_PARAMS *) dp)->n_unsup ,
       ((RBM_GENER_PARAMS *) dp)->nhid_unsup ,
       ((RBM_GENER_PARAMS *) dp)->weights_unsup ,
       ((RBM_GENER_PARAMS *) dp)->in_bias ,
       ((RBM_GENER_PARAMS *) dp)->hid_bias ,
       ((RBM_GENER_PARAMS *) dp)->nchain ,
       ((RBM_GENER_PARAMS *) dp)->input_vis ,
       ((RBM_GENER_PARAMS *) dp)->workvec1 ,
       ((RBM_GENER_PARAMS *) dp)->workvec2 ,
       ((RBM_GENER_PARAMS *) dp)->image ) ;
   return 0 ;
}


/*
--------------------------------------------------------------------------------

   Child members

--------------------------------------------------------------------------------
*/

GenerativeChild::GenerativeChild ( int c_first_case , int c_nrows , int c_ncols , int c_nchain  )
{
   int i, k, irow, icol, nr, nc, irnum, icnum, ir, ic, nvis, icase, ret_val, n_threads ;
   int image_number, data_index, save_data_index, empty_slot ;
   double *inptr, *workvec1, *workvec2 ;
   char msg[256] ;
   unsigned char *raw_image, *data, *dptr ;
   RBM_GENER_PARAMS params[MAX_THREADS] ;
   HANDLE threads[MAX_THREADS] ;

   first_case = c_first_case ;
   nrows = c_nrows ;  // These refer to the grid of images displayed
   ncols = c_ncols ;
   nchain = c_nchain ;

   nvis = model->n_data_inputs ;

/*
   Allocate memory
*/

   raw_image = NULL ;
   data = NULL ;
   dib = NULL ;

   nr = MNIST_rows * nrows + (nrows-1) * 3 + 2 * 2 ;
   nc = MNIST_cols * ncols + (ncols-1) * 3 + 2 * 2 ;

   ok = 1 ;

   raw_image = (unsigned char *) MALLOC ( 3 * nr * nc ) ;
   data = (unsigned char *) MALLOC ( nrows * ncols * nvis * sizeof(unsigned char) ) ;
   workvec1 = (double *) MALLOC ( model->max_neurons * max_threads * sizeof(double) ) ;
   workvec2 = (double *) MALLOC ( model->max_neurons * max_threads * sizeof(double) ) ;

   if (raw_image == NULL  ||  data == NULL  ||  workvec1 == NULL  ||  workvec2 == NULL) {
      if (raw_image != NULL)
         FREE ( raw_image ) ;
      if (data != NULL)
         FREE ( data ) ;
      if (workvec1 != NULL)
         FREE ( workvec1 ) ;
      if (workvec2 != NULL)
         FREE ( workvec2 ) ;
      ok = 0 ;
      audit ( "" ) ;
      audit ( "ERROR... Insufficient memory to display generative samples" ) ;
      return ;
      }

/*
   Initialize parameters that will not change for threads.
*/

   for (i=0 ; i<max_threads ; i++) {
      params[i].nvis = model->n_data_inputs ;
      params[i].max_neurons = model->max_neurons ;
      params[i].n_unsup = model->n_unsup ;
      params[i].nhid_unsup = model->nhid_unsup ;
      params[i].weights_unsup = model->weights_unsup ;
      params[i].in_bias = model->in_bias ;
      params[i].hid_bias = model->hid_bias ;
      params[i].nchain = nchain ;
      params[i].input_vis = (first_case > 0) ;
      params[i].workvec1 = workvec1 + i * model->max_neurons ;
      params[i].workvec2 = workvec2 + i * model->max_neurons ;
      }

/*
   Compute the generated images
*/

   n_threads = 0 ;                    // Counts threads that are active
   for (i=0 ; i<max_threads ; i++)
      threads[i] = NULL ;

   image_number = 0 ; // Index of generated image (nrows*ncols of them)
   empty_slot = -1 ;  // After full, will identify the thread that just completed

   for (;;) {         // Main thread loop processes all images

/*
   Handle user ESCape
*/

      if (escape_key_pressed  ||  user_pressed_escape ()) {
         audit ( "" ) ;
         audit ( "WARNING: User pressed ESCape during generative sampling" ) ;
         MEMTEXT ( "GENERATIVE.CPP: ESCape detected" ) ;
         user_pressed_escape () ;
         for (i=0, k=0 ; i<max_threads ; i++) {
            if (threads[i] != NULL)
               threads[k++] = threads[i] ;
            }
         ret_val = WaitForMultipleObjects ( n_threads , threads , TRUE , 1200000 ) ;
         if (ret_val == WAIT_TIMEOUT)
            audit ( "Timeout waiting for generative computation user ESCape" ) ;
         sprintf ( msg, "GENERATIVE.CPP: User abort; n_threads=%d  k=%d  Wait retval=%d", n_threads, k, ret_val ) ;
         MEMTEXT ( msg ) ;
         for (i=0 ; i<n_threads ; i++)
            CloseHandle ( threads[i] ) ;
         ok = 0 ;
         escape_key_pressed = 0 ;
         return ;
         }

/*
   Start a new thread if we still have work to do
*/

      if (image_number < nrows*ncols) { // If there are still some to do

         if (empty_slot < 0)    // Negative while we are initially filling the queue
            k = n_threads ;
         else
            k = empty_slot ;

         if (first_case > 0) {  // We start with a visible layer from training set
            icase = (first_case + image_number - 1) % n_cases ;
            inptr = database + icase * n_vars ;     // Point to this case in the database
            for (i=0 ; i<nvis ; i++) {              // Put starting case in workvec1
               if (TrainParams.binary_input)
                  params[k].workvec1[i] = (inptr[model->inputs[i]] > model->in_mean[i]) ? 1.0 : 0.0 ;
               else {
                  params[k].workvec1[i] = (inptr[model->inputs[i]] - model->in_min[i]) / (model->in_max[i] - model->in_min[i]) ;
                  assert ( workvec1[i] >= 0.0 ) ;
                  assert ( workvec1[i] <= 1.0 ) ;
                  }
               }
            }

         else {  // We start with a random top hidden layer (the RBM)
            for (i=0 ; i<model->nhid_unsup[model->n_unsup-1] ; i++)
               params[k].workvec2[i] = (unifrand_fast() >= 0.5)  ?  1.0 : 0.0 ;
            }

         params[k].image = data + image_number * nvis ;

         threads[k] = (HANDLE) _beginthreadex ( NULL , 0 , gen_wrapper , &params[k] , 0 , NULL ) ;
         if (threads[k] == NULL) {
            audit ( "Internal ERROR: bad thread creation in GENERATIVE.CPP" ) ;
            for (i=0 ; i<n_threads ; i++) {
               if (threads[i] != NULL)
                  CloseHandle ( threads[i] ) ;
               }
            ok = 0 ;
            return ;
            }
         ++n_threads ;
         ++image_number ;
         } // if (image_number < nrows*ncols)

      if (n_threads == 0)  // Are we done?
         break ;

/*
   Handle full suite of threads running and more threads to add as soon as some are done.
   Wait for just one thread to finish.
*/

      if (n_threads == max_threads  &&  image_number < nrows*ncols) {
         ret_val = WaitForMultipleObjects ( n_threads , threads , FALSE , 1200000 ) ;
         if (ret_val == WAIT_TIMEOUT  ||  ret_val == WAIT_FAILED  ||  ret_val < 0  ||  ret_val >= n_threads) {
            sprintf ( msg, "INTERNAL ERROR!!!  Thread wait 1 failed (%d) in GENERATIVE", ret_val ) ;
            audit ( msg ) ;
            MEMTEXT ( msg ) ;
            if (ret_val == WAIT_TIMEOUT)
               audit ( "Timeout waiting for generative computation to finish; problem too large" ) ;
            ok = 0 ;
            return ;
            }

         empty_slot = ret_val ;
         CloseHandle ( threads[empty_slot] ) ;
         threads[empty_slot] = NULL ;
         --n_threads ;
         }

/*
   Handle all work has been started and now we are just waiting for threads to finish
*/

      else if (image_number == nrows*ncols) {
         ret_val = WaitForMultipleObjects ( n_threads , threads , TRUE , 1200000 ) ;
         if (ret_val == WAIT_TIMEOUT  ||  ret_val == WAIT_FAILED  ||  ret_val < 0  ||  ret_val >= n_threads) {
            sprintf ( msg, "INTERNAL ERROR!!!  Thread wait 2 failed (%d) in GENERATIVE", ret_val ) ;
            audit ( msg ) ;
            MEMTEXT ( msg ) ;
            if (ret_val == WAIT_TIMEOUT)
               audit ( "Timeout waiting for generative computation to finish; problem too large" ) ;
            ok = 0 ;
            return ;
            }

         for (i=0 ; i<n_threads ; i++)
            CloseHandle ( threads[i] ) ;

         break ;
         } // Waiting for final threads to finish
      } // Endless loop which threads computation of criterion for all random tries

/*
   All computation is finished.  Build the display.
*/

// Display as desired
}
