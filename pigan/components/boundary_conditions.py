import tensorflow as tf


class BoundaryConditions():

    def __init__(self, boundary_conditions, noise_sampler, scale_factor = 1):

        self.boundary_conditions = boundary_conditions
        self.noise_sampler = noise_sampler
        self.mse = tf.keras.losses.MeanSquaredError()

        self._scale_factor = scale_factor
    
    def evaluate_loss(self, generator_u, generator_E, tape):
        '''
        Combines boundary losses from top, bottom and right boundaries
        '''
        nu = 0.3
        num_bc = 100

        bc_right_loss = self._sigma_boundary_loss(generator_u,
                                                  generator_E, tape,
                                                  nu, num_bc)
        bc_top_loss = self._top_boundary_loss(generator_u, generator_E,
                                              tape, nu, num_bc)
        bc_bottom_loss = self._bottom_boundary_loss(generator_u, 
                                                    generator_E,
                                                    tape, nu, num_bc)
        bc_left_loss = self._u_boundary_loss(generator_u,generator_E, tape, 
                                             nu, num_bc)

        bc_loss = bc_right_loss + bc_top_loss + bc_bottom_loss + bc_left_loss

        return bc_loss

    def _u_boundary_loss(self, generator_u, generator_E, tape, nu, num_bc):
        '''
        Combines output from equations 21 and 22 to calculate
        boundary condition on the left boundary of the domain
        '''
        
        X_ux_bc = self.boundary_conditions['X_u_bc']
        n_ux_sens = X_ux_bc.shape[0]
        ux_bc = tf.zeros((n_ux_sens, 1))
        ux_expected_vals = tf.tile(ux_bc, [num_bc, 1])
        noise_u_bc = self.noise_sampler.sample_noise(num_sensors=n_ux_sens,
                                                     batch_size=num_bc)
        X_ux_bc_tile = tf.tile(X_ux_bc, [num_bc, 1])
        ux_bc_inputs = tf.concat([X_ux_bc_tile, noise_u_bc], axis=1)
        generated_u = generator_u(ux_bc_inputs, training=True)
        generated_ux_bc, generated_uy_bc = tf.split(generated_u, 
                                                    num_or_size_splits=2, 
                                                    axis=1)
        
        tile_indicies=tf.range(0,len(generated_uy_bc),n_ux_sens)
        generated_uy_bc_at_origin=tf.gather(generated_uy_bc,tile_indicies)
        uy_expected_vals=tf.zeros_like(generated_uy_bc_at_origin)
        
        u_boundary_evaluation=tf.concat((generated_ux_bc,
                                         generated_uy_bc_at_origin),0)
        expected_vals=tf.concat((ux_expected_vals,uy_expected_vals),0)
        loss= self.mse(u_boundary_evaluation,expected_vals)
        return loss

    def _sigma_boundary_loss(self, generator_u, generator_E, tape, nu, 
                                   num_bc):
        '''
        Combines output from equations 28 and 29 to calculate
        boundary condition on the right boundary of the domain
        '''

        X_sigma_bc = self.boundary_conditions['X_sigma_bc']
        n_sig_sens = X_sigma_bc.shape[0]
        sigma_bc = self.boundary_conditions['sigma_bc']
        sigma_bc = tf.tile(sigma_bc, [num_bc, 1])
        #zero_bc = self.boundary_conditions['sigma_zero']
        #zero_bc = tf.tile(zero_bc, [num_bc, 1])
        zero_bc = tf.zeros_like(sigma_bc)

        noise_sigma_bc = self.noise_sampler.sample_noise(
                num_sensors=n_sig_sens, batch_size=num_bc)
        X_sigma_bc_tile = tf.tile(X_sigma_bc, [num_bc, 1])
        sigma_bc_inputs= tf.concat([X_sigma_bc_tile, noise_sigma_bc], axis=1)
        generated_u_sigma_bc = generator_u(sigma_bc_inputs, training=True)
        generated_E_sigma_bc = generator_E(sigma_bc_inputs, training=True)

        dux_dx, dux_dy, duy_dx, duy_dy = self._get_partial_derivatives(
                                                       generated_u_sigma_bc,
                                                       X_sigma_bc_tile, tape)

        #Compute applied traction BC:
        gen_sigma_xx = self._get_sigma_xx(dux_dx, duy_dy, 
                                          generated_E_sigma_bc, nu)
        sigma_xx_loss = self.mse(sigma_bc, gen_sigma_xx)

        #Compute zero shear stress (free) BC:
        gen_sigma_xy = self._get_sigma_xy(dux_dy, duy_dx,
                                          generated_E_sigma_bc, nu)
        sigma_xy_loss = self.mse(zero_bc, gen_sigma_xy)

        bc_right_loss = sigma_xy_loss + sigma_xx_loss
        return bc_right_loss

    def _top_boundary_loss(self, generator_u, generator_E, tape, nu,
                                 num_bc):
        '''
        Returns boundary loss on top boundary of domain
        '''
        X_sigma_bc = self.boundary_conditions['X_sigma_hi']

        zero_bc = tf.zeros((X_sigma_bc.shape[0], 1))
        zero_bc = tf.tile(zero_bc, [num_bc, 1])

        gen_sigma_yy, gen_sigma_xy = self._get_sigma_yy_xy_on_boundary(
                                                            generator_u,
                                                            generator_E,
                                                            tape, nu, num_bc,
                                                            X_sigma_bc)
        sigma_yy_loss = self.mse(zero_bc, gen_sigma_yy)
        sigma_xy_loss = self.mse(zero_bc, gen_sigma_xy)

        bc_top_loss = sigma_xy_loss + sigma_yy_loss
        return bc_top_loss

    def _bottom_boundary_loss(self, generator_u, generator_E, tape, nu,
                                    num_bc):
        '''
        Returns boundary loss on bottom boundary of domain
        '''
        X_sigma_bc = self.boundary_conditions['X_sigma_lo']
        zero_bc = tf.zeros((X_sigma_bc.shape[0], 1))
        zero_bc = tf.tile(zero_bc, [num_bc, 1])

        gen_sigma_yy, gen_sigma_xy = self._get_sigma_yy_xy_on_boundary(
                                                            generator_u,
                                                            generator_E,
                                                            tape, nu, num_bc,
                                                            X_sigma_bc)
        sigma_yy_loss = self.mse(zero_bc, gen_sigma_yy)
        sigma_xy_loss = self.mse(zero_bc, gen_sigma_xy)

        bc_bottom_loss = sigma_xy_loss + sigma_yy_loss
        return bc_bottom_loss

    def _get_sigma_yy_xy_on_boundary(self, generator_u, generator_E,
                                     tape, nu, num_bc, X_bc):
        '''
        Returns boundary calculations from equations 30 and 31 for the top or
        the bottom of the domain.
        '''
        n_sig_sens = X_bc.shape[0]
        noise_sigma_bc = self.noise_sampler.sample_noise(
                num_sensors=n_sig_sens, batch_size=num_bc)
        X_sigma_bc_tile = tf.tile(X_bc, [num_bc, 1])
        sigma_bc_inputs= tf.concat([X_sigma_bc_tile, noise_sigma_bc], axis=1)
        generated_u_sigma_bc = generator_u(sigma_bc_inputs, training=True)
        generated_E_sigma_bc = generator_E(sigma_bc_inputs, training=True)

        dux_dx, dux_dy, duy_dx, duy_dy = self._get_partial_derivatives(
                                                       generated_u_sigma_bc,
                                                       X_sigma_bc_tile, tape)

        gen_sigma_yy = self._get_sigma_yy(dux_dx, duy_dy, 
                                          generated_E_sigma_bc, nu)
        gen_sigma_xy = self._get_sigma_xy(dux_dy, duy_dx,
                                          generated_E_sigma_bc, nu)
        return gen_sigma_yy, gen_sigma_xy

    def _get_partial_derivatives(self, gen_u, xy_coords, tape):
        '''
        Returns all first order partial derivatives 
        '''
        gen_ux, gen_uy = tf.split(gen_u, num_or_size_splits=2, axis=1)

        grad_ux = tape.gradient(gen_ux, xy_coords)
        grad_uy = tape.gradient(gen_uy, xy_coords)
        dux_dx, dux_dy = tf.split(grad_ux, num_or_size_splits=2, axis=1)
        duy_dx, duy_dy = tf.split(grad_uy, num_or_size_splits=2, axis=1)

        # HACK: rescale displacements
        dux_dx, dux_dy = dux_dx * self._scale_factor, dux_dy* self._scale_factor
        duy_dx, duy_dy = duy_dx * self._scale_factor, duy_dy* self._scale_factor
        # END HACK

        return dux_dx, dux_dy, duy_dx, duy_dy

    def _get_sigma_xx(self, dux_dx, duy_dy, E, nu):
        '''
        Implements Equation 28
        
        '''

        ux_dx_plus_nu_uy_dy = dux_dx + nu * duy_dy
        sigma_xx = (1. / (1. - nu ** 2)) * E * ux_dx_plus_nu_uy_dy
        return sigma_xx

    def _get_sigma_yy(self, dux_dx, duy_dy, E, nu):
        '''
        Implements Equation 30
        '''

        nu_ux_dx_plus_uy_dy = nu * dux_dx + duy_dy
        sigma_yy = (1. / (1. - nu ** 2)) * E * nu_ux_dx_plus_uy_dy
        return sigma_yy

    def _get_sigma_xy(self, dux_dy, duy_dx, E, nu):
        '''
        Implements Equation 29 and 31
        '''
        
        factor = 1. + nu
        sigma_xy = (1/factor) * E * 0.5 * (dux_dy + duy_dx)
        return sigma_xy
