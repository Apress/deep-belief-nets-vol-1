The code in the book has been simplified by the removal
of memory allocation, error checking, et cetera,
in order to provide the clearest presentation.

Most users would like examples of reasonable ways
to handle these issues.  Also, few readers would want
to type in those long subroutines.

For these reasons I am making available mostly or
entirely complete subroutines in this download.

However, because these routines are part of a large program,
they necessarily contain numerous references to external
routines.  These references must be replaced by the user
with routines appropriate to the program the user
is constructing.  This should not be difficult, as these
external references are just to things like memory allocation
and checking if the user has pressed the ESCape key.

Also, this code includes numerous references to a Model class
whose declaration is not supplied.  This is because the
complete declaration is large and complex.  But the
references to Model members are all straightforward,
so the reader should have no difficulty adapting these
references to his/her own Model class.