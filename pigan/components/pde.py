import tensorflow as tf

class PDE():
    def __init__(self, scale_factor=1):
        self._scale_factor = scale_factor

    def evaluate_loss(self, terms, tape):
        mse = tf.keras.losses.MeanSquaredError()
        real_output = self.calculate_residual(terms,tape)
        expected_output = tf.zeros_like(real_output) 
        return mse(expected_output,real_output)

    def calculate_residual(self, terms, tape):
        """
        Calculates the PDE from Equations 26 and 27 in the paper

        Parameters
        ----------
        terms : dict
            Dictionary containing variables that will be used to evaluate the 
            PDE constraint.

        tape : tf.GradientTape
            Gradient tape instance that has kept watch of the generator inputs.

        Returns
        -------
        f : tf.Tensor
            PDE evaluation.
        """
        nu = 0.3

        X = terms['X']
        u = terms['u']
        E = terms['E']
        ux, uy = tf.split(u, num_or_size_splits=2, axis=1)

        ux_dx, ux_dy = self.get_partial_derivatives(ux, X, tape)
        uy_dx, uy_dy = self.get_partial_derivatives(uy, X, tape)
    
        ux_dxx, ux_dxy = self.get_partial_derivatives(ux_dx, X, tape)
        ux_dyx, ux_dyy = self.get_partial_derivatives(ux_dy, X, tape)

        uy_dxx, uy_dxy = self.get_partial_derivatives(uy_dx, X, tape)
        uy_dyx, uy_dyy = self.get_partial_derivatives(uy_dy, X, tape)

        # HACK: rescale the displacements; safer to do individually like this?
        ux, uy = ux * self._scale_factor, uy * self._scale_factor

        ux_dx, ux_dy = ux_dx * self._scale_factor, ux_dy * self._scale_factor
        uy_dx, uy_dy = uy_dx, uy_dy * self._scale_factor
                                     
        ux_dxx, ux_dxy = ux_dxx* self._scale_factor, ux_dxy * self._scale_factor
        ux_dyx, ux_dyy = ux_dyx* self._scale_factor, ux_dyy * self._scale_factor

        uy_dxx, uy_dxy = uy_dxx* self._scale_factor, uy_dxy * self._scale_factor
        uy_dyx, uy_dyy = uy_dyx* self._scale_factor, uy_dyy * self._scale_factor
        # END HACK
        
        dE_dx, dE_dy = self.get_partial_derivatives(E,X,tape)
        
        #Equation 26
        f_1 = dE_dx * (ux_dx + nu * uy_dy) + E * (ux_dxx + nu * uy_dyx) + (
                    1. - nu) / 2. * (
                          (dE_dy * (ux_dy + uy_dx)) + (E * (ux_dyy + uy_dxy)))
        #Equation 27
        f_2 = (1. - nu) / 2. * ((dE_dx * (ux_dy + uy_dx)) + (
                    E * (ux_dyx + uy_dxx))) + dE_dy * (
                          nu * ux_dx + uy_dy) + E * (nu * ux_dxy + uy_dyy)

        f = tf.concat((f_1, f_2), axis=0)
        return f

    def get_partial_derivatives(self,f, X, tape):
        return tf.split(tape.gradient(f, X), num_or_size_splits=2, axis=1)
    
