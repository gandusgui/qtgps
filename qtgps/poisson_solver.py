from .solvers.implicit_poisson_solver_periodic import implicit_poisson_solver_periodic as solve_unconstrained
from .solvers.implicit_poisson_solver_mixed_periodic import implicit_poisson_solver_mixed_periodic as solve_dbcs

class PoissonSolver:

    def __init__(self, max_iter=30, omega=1, tol=1e-8):
        self.max_iter = max_iter
        self.omega = omega
        self.tol = tol
        self._neumann_directions = None
        self._dirichlet_directions = None
        self._uptodate = False

    def set_grid_descriptor(self, rsg):
        self.rsg = rsg

    def set_dielectric_descriptor(self, dielectric):
        assert dielectric.rsg is self.rsg
        self.dielectric = dielectric

    def _set(self, **kwargs):
        for key in kwargs.keys():
            if key in ['neumann_directions', 'dirichlet_directions']:
                my_key = '_' + key
                if getattr(self, my_key) != kwargs[key]:
                    setattr(self, my_key, kwargs[key])
                    self._uptodate = False
            else:
                raise ValueError('None valid key {}'.format(key))
        if not hasattr(self, 'fsg'):
            self._uptodate = False

    def solve(self, rho, dbcs=None, neumann_directions=None,
              dirichlet_directions=None):
        #Update uptodate, if necessary
        self._set(neumann_directions=neumann_directions,
               dirichlet_directions=dirichlet_directions)
        #Boundary conditions have changed or fsg never initialized?
        if not self._uptodate:
            self.fsg = self.rsg.get_reciprocal(neumann_directions,
                                      dirichlet_directions)

        args = [self.rsg, self.fsg, self.dielectric,
                self.tol, self.omega, self.max_iter, rho]

        if dbcs is None:
            V = solve_unconstrained(*args)

        else:
            V = solve_dbcs(dbcs, *args)

        return V
