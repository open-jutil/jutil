import numpy as np
import numpy.linalg as la
import cg
from lnsrch import lnsrch


def getChiSquareProbability(chisq, N):
    import scipy.special
    result = 1.0
    if chisq > 0.0:
        n_h = 0.5 * N
        chisq_h = 0.5 * chisq
        if chisq_h < n_h:
            result = 1.0 - scipy.special.gammainc(n_h, chisq_h)
        else:
            result = scipy.special.gammaincc(n_h, chisq_h)
    return 100.0 * result


class Minimizer(object):
    def __init__(self, stepper):
        self._stepper = stepper
        self.conv_min_costfunction_gradient = 0
        self.conv_discrepancy_principle_tau = 0
        self.conv_min_costfunction_reduction = 0
        self.conv_min_normalized_stepsize = 0
        self.conv_max_iteration = 10


    def printInfo(self, it, J, disq, normb):
        print "it= {it} / chi^2/m= {chisq} (meas= {chisqm} / apr= {chisqa} ) / d_i^2/n= {disq} / |J'|= {normb} / Q= {prob}".format(
                it=it, chisq=J.chisq, chisqm=J.chisq_m,
                chisqa=J.chisq_a, disq=disq, normb=normb,
                prob=getChiSquareProbability(J.chisq * J.m, J.m))


    def __call__(self, J, x_0):
        x_i = x_0.copy()

        J.init(x_i)

        if hasattr(self._stepper, "init"):
            self._stepper.init()

        disq  = 0.0
        it = 0
        while True:
            if hasattr(J, "updateJacobian"):
                J.updateJacobian(x_i)
            kernel_recomp = False
            last_kernel_recomp = it

            b = -J.jac(x_i)

            if la.norm(b) / J.m < self.conv_min_costfunction_gradient:
                print "Convergence criteria reached (dfmin)"
                break

            self.printInfo(it, J, disq, la.norm(b))

            chisq_old = J.chisq
            x_step = self._stepper(J, b, x_i)
            if np.any(np.isnan(x_step)):
                raise RuntimeError("Retrieval failed (x_step is NaN)! " + repr(x_step))

            x_i += x_step
            it += 1

            chisq = J.chisq
            assert chisq <= chisq_old

            # normalize step size in state space
            disq = np.dot(x_step, b) / J.n

            # Discrepancy principle
            dp_reached = (np.sqrt(J.chisq_m * J.m) <
                          self.conv_discrepancy_principle_tau *
                          np.sqrt(J.m))

            # Convergence test based on reduction of cost function...
            fmin_reached = 100. * abs(1. - chisq / chisq_old) < self.conv_min_costfunction_reduction
            # Convergence test on normalized step size
            dmin_reached = disq < self.conv_min_normalized_stepsize
            maxit_reached = it >= self.conv_max_iteration

            if dmin_reached or fmin_reached or dp_reached or maxit_reached:
                print "Convergence criteria reached. {dp}{fmin}{dmin}{maxit}".format(
                        dp="(dp)" if dp_reached else "",
                        fmin="(fmin)" if fmin_reached else "",
                        dmin="(dmin)" if dmin_reached else "",
                        maxit="(maxit)" if maxit_reached else "")
                break
        b = -J.jac(x_i)
        self.printInfo(it, J, disq, la.norm(b))

        return x_i


class LevenbergMarquardtStepper(object):
    def __init__(self, lmpar, factor,
                 cg_max_it=-1, cg_err_rel=1e-20, cg_err_abs=1e-20):
        self._lmpar_init = lmpar
        self._lmpar = self.self._lmpar_init
        self._factor = factor
        self._cg_max_it = cg_max_it
        self._cg_err_rel = cg_err_rel
        self._cg_err_abs = cg_err_abs

    def init():
        self._lmpar = self.self._lmpar_init

    def __call__(self, J, b, x_i):
        chisq_old = J.chisq
        while True:
            # Solve J''(x_i) x_step = J'(x_i)
            x_step = cg.conj_grad_solve(cg.CostFunctionCGWrapper(J, x_i, lmpar=self._lmpar), b,
                                        self._cg_max_it, self._cg_err_abs, self._cg_err_rel)
            print x_i, x_step
            x_new = x_i + x_step
            chisq = J(x_new)
            print x_i.shape, x_step.shape
            if chisq > chisq_old:
                self._lmpar *= self._factor
                if self.lmpar_ > 1e30:
                    raise RuntimeError("Retrieval failed (levenberg marquardt parameter too large)! i" + repr(self._lmpar))
                print "Increasing lmpar to {} ({}>{})".format(self._lmpar, chisq, chisq_old)
            else:
                self._lmpar /= self._factor
                print "Decreasing lmpar to {} ({}<{})".format(self._lmpar, chisq, chisq_old)

                return x_step


class SteepestDescentStepper(object):
    def __init__(self):
        pass

    def __call__(self, J, b, x_i):
        chisq_old = J.chisq
        check, chisq, x_new = lnsrch(x_i, chisq_old, -b, b, J)
        x_step = x_new - x_i

        return x_step


class ScaledSteepestDescentStepper(object):
    def __init__(self):
        pass

    def __call__(self, J, b, x_i):
        chisq_old = J.chisq
        direc = (np.dot(b, b) / np.dot(J.hess_dot(x_i, b), b)) * b
        check, chisq, x_new = lnsrch(x_i, chisq_old, -b, direc, J)

        x_step = x_new - x_i

        return x_step


class GaussNewtonStepper(object):
    def __init__(self, cg_max_it=-1, cg_err_rel=1e-20, cg_err_abs=1e-20):
        self._cg_max_it = cg_max_it
        self._cg_err_rel = cg_err_rel
        self._cg_err_abs = cg_err_abs

    def __call__(self, J, b, x_i):
        chisq_old = J.chisq
        # Solve J''(x_i) x_step = J'(x_i)
        x_step = cg.conj_grad_solve(cg.CostFunctionCGWrapper(J, x_i), b,
                                    self._cg_max_it, self._cg_err_abs, self._cg_err_rel)
        print J(x_i + x_step), J.jac(x_i + x_step)
        check, chisq, x_new = lnsrch(x_i, chisq_old, -b, x_step, J)
        x_step = x_new - x_i
        print J(x_i + x_step), J.jac(x_i + x_step)

        return x_step


class TruncatedQuasiNewtonStepper(object):
    def __init__(self, conv_rel, factor, cg_max_it=-1):
        self._conv_rel_init = conv_rel
        self._conv_rel = self._conv_rel_init
        self._factor = factor
        self._cg_max_it = cg_max_it

    def init():
        self._conv_rel = self._conv_rel_init

    def __call__(self, J, b, x_i):
        chisq_old = J.chisq
        assert chisq_old > 0

        err_rels = [self._conv_rel]
        while err_rels[-1] < 1:
            err_rels.append(min(err_rels[-1] * self._factor, 1.0))

        x_steps = cg.conj_grad_solve(cg.CostFunctionCGWrapper(J, x_i), b, self._cg_max_it, 1e-20, err_rels)
        for i, x_step in enumerate(x_steps):
            x_new = x_i + x_step
            chisq = J(x_new)
            if chisq > chisq_old and i + 1 < len(x_steps):
                continue

            if chisq > chisq_old and i + 1 == len(x_steps):
                print "  CG steps exhausted. Employing line search."
                print -b, x_step
                check, chisq, x_new = lnsrch(x_i, chisq_old, -b, x_step, J)
                x_step = x_new - x_i
                self._conv_rel = 1. / self._factor
            else:
                self._conv_rel = err_rels[i] / self._factor
            break
        print "  Decreasing reltol to {} ({}<{})".format(self._conv_rel, chisq, chisq_old)
        return x_step

