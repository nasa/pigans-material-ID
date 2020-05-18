from __future__ import print_function

import numpy as np
from dolfin import *
from scipy import interpolate as scipy_interp


class StiffnessInterp(UserExpression):
    def __init__(self, interpolator, **kwargs):
        self._interp = interpolator
        super().__init__(**kwargs)

    def eval(self, value, x):
        value[0] = self._interp(x[0], x[1])

    def value_shape(self):
        return ()


def compute_disps_on_grid(E_grid_data, disp_sensor_coords, L=1.,
                          w=0.2, Nx=200, Ny=40, nu=0.3, traction=0.1,
                          output_paraview=False):
    '''
    E_grid_data: #grid pts x 3 array with {x_pts, y_pts, E_vals}
    disp_sensor_coords: # sensors x 2 array with x/y coordinates of u sensor loc
    L: length
    w: width
    Nx: number of elements in x direction
    Ny: ...
    nu: poisson ratio
    traction: applied traction value on right side

    returns u_x, u_y  - arrays with x/y components of displacement evaluated
        at the 2d grid defined by disp_x/y_sensors
    '''

    mesh = RectangleMesh(Point(0., 0.), Point(L, w), Nx, Ny, "crossed")

    interp = scipy_interp.LinearNDInterpolator(E_grid_data[:, 0:2],
                                               E_grid_data[:, 2])
    E = StiffnessInterp(interp)

    D = FunctionSpace(mesh, "DG", 0)
    E_field = interpolate(E, D)
    nu = Constant(nu)

    def eps(v):
        return sym(grad(v))

    # https://en.wikipedia.org/wiki/Hooke%27s_law#Plane_stress:
    def sigma(v):
        return E_field / (1. - nu ** 2) * (
                (1. - nu) * eps(v) + nu * tr(eps(v)) * Identity(2))

    V = VectorFunctionSpace(mesh, 'Lagrange', degree=2)
    du = TrialFunction(V)
    u_ = TestFunction(V)
    a = inner(sigma(du), eps(u_)) * dx

    # Handle boundary conditions
    # Dirichlet displacement BC on the left side of beam:
    def left(x, on_boundary):
        return near(x[0], 0.)

    def bottomleftcorner(x, on_boundary):
        return (near(x[0], 0.) and near(x[1], 0.))

    bc1 = DirichletBC(V.sub(0), Constant(0.), left)
    bc2 = DirichletBC(V, Constant((0., 0.)), bottomleftcorner,
                      method='pointwise')
    bcs = [bc1, bc2]

    # Traction boundary condition on the right side of beam:
    class Right(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], L) and on_boundary

    # Define traction:
    T = Constant((traction, 0.0))
    facets = MeshFunction("size_t", mesh, 1)
    facets.set_all(0)
    Right().mark(facets, 1)
    ds = Measure('ds')(subdomain_data=facets)

    # Define right hand side of variational form
    l = dot(T, u_) * ds(1)

    # Solve problem
    u = Function(V, name="Displacement")
    set_log_level(50)
    solve(a == l, u, bcs)

    if output_paraview:
        file_results = XDMFFile("elasticity_results.xdmf")
        file_results.parameters["flush_output"] = True
        file_results.parameters["functions_share_mesh"] = True
        file_results.write(u, 0.)

    u_x = np.zeros(len(disp_sensor_coords))
    u_y = np.zeros(len(disp_sensor_coords))

    for i, coord in enumerate(disp_sensor_coords):
        p = Point(coord[0], coord[1])
        disp = u(p)
        u_x[i] = disp[0]
        u_y[i] = disp[1]

    return u_x, u_y
