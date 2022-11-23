import numpy as np


def trigonometric_function(dielectric, rho_elec, rho_min, rho_max):
    # https://aip.scitation.org/doi/pdf/10.1063/1.3676407
    twopi = 2 * pi
    f = np.log(dielectric) / twopi
    ln_rho_max = np.log(rho_max)
    ln_rho_min = np.log(rho_min)
    q = twopi / (ln_rho_max - ln_rho_min)
    x = np.log(rho_elec)
    y = q * (ln_rho_max - x)
    t = f * (y - np.sin(y))
    return t


class DielectricGrid:
    def __init__(
        self,
        rsg,
        rho_elec=None,
        dielectric=1.0,
        rho_min=1e-4,
        rho_max=3.5e-3,
        rho_dependent=False,
    ):

        self.rsg = rsg
        self.dielectric = dielectric
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.rho_dependent = rho_dependent

        self.eps_elec = rsg.zeros()  # self.dielectric * np.ones(self.size)
        self.update(rho_elec)

    @property
    def eps(self):
        return self.eps_elec

    def update(self, rho_elec=None):

        # if rho_elec is not None:
        self.rho_elec = rho_elec
        self.update_dielectric()
        self.update_gradient()

    def update_dielectric(self):

        if self.rho_dependent is False:
            self.eps_elec[:] = self.dielectric

        else:
            tol = 1e-12
            diff = self.rho_max - self.rho_min

            ind_less = self.rho_elec < self.rho_min
            ind_greater = self.rho_elec > self.rho_max
            ind_in = ~(ind_less | ind_greater)

            self.eps_elec[ind_greater] = 1.0
            self.eps_elec[ind_less] = self.dielectric

            if diff > tol:
                t = trigonometric_function(
                    self.dielectric, self.rho_elec[ind_in], self.rho_min, self.rho_max
                )
                self.eps_elec[ind_in] = np.exp(t)

            else:
                self.eps_elec[ind_in] = 1.0

    def update_gradient(self):

        rsg = self.rsg
        dx = rsg.dx
        dy = rsg.dy
        dz = rsg.dz

        if self.rho_dependent is False:
            deps_dx = rsg.zeros()
            deps_dy = rsg.zeros()
            deps_dz = rsg.zeros()
            ln_deps_dx = rsg.zeros()
            ln_deps_dy = rsg.zeros()
            ln_deps_dz = rsg.zeros()

        else:

            deps_dx, deps_dy, deps_dz = np.grad(self.eps_elec, dx, dy, dz)
            ln_deps_dx, ln_deps_dy, ln_deps_dz = np.gradient(
                np.log(self.eps_elec), dx, dy, dz
            )

        self.deps_dx = deps_dx
        self.deps_dy = deps_dy
        self.deps_dz = deps_dz
        self.ln_deps_dx = deps_dx
        self.ln_deps_dy = deps_dy
        self.ln_deps_dz = deps_dz
