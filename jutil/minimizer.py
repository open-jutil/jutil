import numpy as np
import numpy.linalg as la
from jutil.lnsrch import lnsrch
from jutil.operator import CostFunctionOperator
import jutil.cg as cg


def get_chi_square_probability(chisq, N):
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
        self._conv = {
            "min_costfunction_gradient": 0,
            "discrepancy_principle_tau": 0,
            "min_costfunction_reduction": 0,
            "min_normalized_stepsize": 0,
            "max_iteration": 10,
        }

    def update_tolerances(self, tol):
        assert all([key in self._conv for key in tol]), tol
        self._conv.update(tol)


    def _print_info(self, it, J, disq, normb):
        print "it= {it} / chi^2/m= {chisq} (meas= {chisqm} / apr= {chisqa} ) / d_i^2/n= {disq} / |J'|= {normb} / Q= {prob}".format(
            it=it, chisq=J.chisq, chisqm=J.chisq_m,
            chisqa=J.chisq_a, disq=disq, normb=normb,
            prob=get_chi_square_probability(J.chisq * J.m, J.m))

    def __call__(self, J, x_0):
        print self._conv
        x_i = x_0.copy()

        J.init(x_i)

        if hasattr(self._stepper, "init"):
            self._stepper.init()

        disq = 0.0
        it = 0
        converged = {}
        while True:
            if hasattr(J, "updateJacobian"):
                J.updateJacobian(x_i)

            b = -J.jac(x_i)

            if la.norm(b) <= self._conv["min_costfunction_gradient"]:
                print "Convergence criteria reached [\"min_costfunction_gradient\"]"
                break

            self._print_info(it, J, disq, la.norm(b))

            chisq_old = J.chisq
            x_step = self._stepper(J, b, x_i)
            if np.any(np.isnan(x_step)):
                raise RuntimeError("Retrieval failed (x_step is NaN)! " + repr(x_step))

            x_i += x_step
            it += 1

            chisq = J.chisq
            assert chisq <= chisq_old, chisq_old - chisq

            # normalize step size in state space
            disq = np.dot(x_step, b) / J.n

            # Discrepancy principle
            converged["discrepancy_principle_tau"] = (J.chisq_m < self._conv["discrepancy_principle_tau"] ** 2)

            # Convergence test based on reduction of cost function...
            converged["min_costfunction_reduction"] = 100. * abs(1. - chisq / chisq_old) <= self._conv["min_costfunction_reduction"]
            # Convergence test on normalized step size
            converged["min_normalized_stepsize"] = disq <= self._conv["min_normalized_stepsize"]
            converged["max_iteration"]= it >= self._conv["max_iteration"]

            if any(converged.values()):
                print "Convergence criteria reached. " + str([x for x in converged if converged[x]])
                break
        b = -J.jac(x_i)
        self._print_info(it, J, disq, la.norm(b))

        return x_i


class LevenbergMarquardtAbstractBase(object):
    def __init__(self, lmpar=1, factor=10,
                 cg_max_iter=-1, cg_tol_rel=1e-20, cg_tol_abs=1e-20):
        self._lmpar_init = lmpar
        self._lmpar = self._lmpar_init
        self._lmpar_factor = factor
        self._cg_max_iter = cg_max_iter
        self._cg_tol_rel = cg_tol_rel
        self._cg_tol_abs = cg_tol_abs

    def init(self):
        self._lmpar = self._lmpar_init


class LevenbergMarquardtReductionStepper(LevenbergMarquardtAbstractBase):
    def __call__(self, J, b, x_i):
        chisq_old = J.chisq
        while True:
            x_step = cg.conj_grad_solve(
                CostFunctionOperator(J, x_i, lmpar=self._lmpar), b,
                max_iter=self._cg_max_iter, abs_tol=self._cg_tol_abs, rel_tol=self._cg_tol_rel)
            x_new = x_i + x_step
            chisq = J(x_new)
            if chisq > chisq_old:
                self._lmpar *= self._lmpar_factor
                if self._lmpar > 1e30:
                    raise RuntimeError("Retrieval failed (levenberg marquardt parameter too large)! i" + repr(self._lmpar))
                print "Increasing lmpar to {} ({}>{})".format(self._lmpar, chisq, chisq_old)
            else:
                self._lmpar /= self._lmpar_factor
                print "Decreasing lmpar to {} ({}<{})".format(self._lmpar, chisq, chisq_old)

                return x_step


class LevenbergMarquardtPredictorStepper(LevenbergMarquardtAbstractBase):
    def __call__(self, J, b, x_i):
        chisq_old = J.chisq
        while True:
            x_step = cg.conj_grad_solve(
                CostFunctionOperator(J, x_i, lmpar=self._lmpar), b,
                max_iter=self._cg_max_iter, abs_tol=self._cg_tol_abs, rel_tol=self._cg_tol_rel)
            x_new = x_i + x_step

            delta_chisq_pred = - np.dot(b, x_step) + 0.5 * np.dot(x_step, J.hess_dot(x_i, x_step))
            delta_chisq = J(x_new) - chisq_old
            if delta_chisq != 0:
                chisq_factor = delta_chisq_pred / delta_chisq
            else:
                chisq_factor = np.Inf
            if chisq_factor < 0.25:
                self._lmpar *= self._lmpar_factor
                if self._lmpar > 1e30:
                    raise RuntimeError("Retrieval failed (levenberg marquardt parameter too large)! i" + repr(self._lmpar))
                print "Increasing lmpar to {} ({}<0.25)".format(self._lmpar, chisq_factor)
            else:
                if chisq_factor > 0.5:
                    self._lmpar /= self._lmpar_factor
                    print "Decreasing lmpar to {} ({}>0.5)".format(self._lmpar, chisq_factor)
            if delta_chisq <= 0:
                return x_step


class SteepestDescentStepper(object):
    def __init__(self, preconditioner=None):
        if preconditioner:
            self._preconditioner = preconditioner
        else:
            self._preconditioner = lambda x: x

    def __call__(self, J, b, x_i):
        chisq_old = J.chisq
        _, _, x_new = lnsrch(x_i, chisq_old, -b, self._preconditioner(b), J)
        return x_new - x_i


class CauchyPointSteepestDescentStepper(object):
    def __init__(self):
        pass

    def __call__(self, J, b, x_i):
        chisq_old = J.chisq
        # This computes the optimal distance along the steepest descent
        # direction
        direc = (np.dot(b, b) / np.dot(J.hess_dot(x_i, b), b)) * b
        _, _, x_new = lnsrch(x_i, chisq_old, -b, direc, J)

        x_step = x_new - x_i

        return x_step


class GaussNewtonStepper(object):
    def __init__(self, cg_max_iter=-1, cg_tol_rel=1e-20, cg_tol_abs=1e-20):
        self._cg_max_iter = cg_max_iter
        self._cg_tol_rel = cg_tol_rel
        self._cg_tol_abs = cg_tol_abs

    def init(self):
        pass

    def __call__(self, J, b, x_i):
        chisq_old = J.chisq
        # Solve J''(x_i) x_step = J'(x_i)
        x_step = cg.conj_grad_solve(
            CostFunctionOperator(J, x_i), b,
            max_iter=self._cg_max_iter, abs_tol=self._cg_tol_abs, rel_tol=self._cg_tol_rel)
        if np.dot(x_step, b) == 0:
            return np.zeros_like(x_step)
        _, _, x_new = lnsrch(x_i, chisq_old, -b, x_step, J)
        x_step = x_new - x_i

        return x_step


class TruncatedCGQuasiNewtonStepper(object):
    def __init__(self, cg_max_iter=-1, cg_tol_abs=1e-20):
        self._cg_max_iter = cg_max_iter
        self._cg_tol_abs = cg_tol_abs

    def init(self):
        pass

    def __call__(self, J, b, x_i):
        chisq_old = J.chisq
        eps = min(0.5, np.sqrt(la.norm(b)))

        x_step = cg.conj_grad_solve(
            CostFunctionOperator(J, x_i), b,
            max_iter=self._cg_max_iter, abs_tol=self._cg_tol_abs, rel_tol=eps)
        _, _, x_new = lnsrch(x_i, chisq_old, -b, x_step, J)
        x_step = x_new - x_i

        return x_step


class TrustRegionTruncatedCGQuasiNewtonStepper(object):
    """
    \todo newton requires dampening (p - 1) for l_p normed cost functions?
    """
    def __init__(self, conv_rel=1e-4, factor=10, cg_max_iter=-1):
        self._conv_rel_init = conv_rel
        self._conv_rel = self._conv_rel_init
        self._factor = factor
        self._cg_max_iter = cg_max_iter

    def _get_err_rels(self):
        result = [self._conv_rel]
        while result[-1] < 1:
            result.append(min(result[-1] * self._factor, 1.0))
        return result

    def init(self):
        self._conv_rel = self._conv_rel_init

    def __call__(self, J, b, x_i):
        chisq_old = J.chisq

        err_rels = self._get_err_rels()
        x_steps = cg.conj_grad_solve(
            CostFunctionOperator(J, x_i), b,
            max_iter=self._cg_max_iter, rel_tol=err_rels, abs_tol=0, verbose=True)
        for i, x_step in enumerate(x_steps):
            x_new = x_i + x_step
            chisq = J(x_new)
            if chisq > chisq_old and i + 1 < len(x_steps):
                continue

            if chisq > chisq_old and i + 1 == len(x_steps):
                print "  CG steps exhausted. Employing line search."
                _, chisq, x_new = lnsrch(x_i, chisq_old, -b, x_step, J)
                x_step = x_new - x_i
                self._conv_rel = 1. / self._factor
            else:
                self._conv_rel = err_rels[i] / self._factor
            break
        print "  Decreasing reltol to {} ({}<{})".format(self._conv_rel, chisq, chisq_old)
        return x_step


def minimize(J, x0, method="TrustRegionTruncatedCGQuasiNewton", options={}, tol={}):
    """
    Front-end for JUTIL non-linear minimization.

    J: CostFunction
    x0: inital guess
    method: String determining method
    options: Additional parameters for the chosen method
    tol: convergence options for the Outer loop

    Supported methods are
    * SteepestDescent
    * CauchyPointSteepestDescent
    * LevenbergMarquardtReduction
    * LevenbergMarquardtPredictor
    * GaussNewton
    * TruncatedCGQuasiNewton
    * TrustRegionTruncatedCGQuasiNewton(default)

    """
    current_module = __import__(__name__)
    try:
        meth = globals()[method + "Stepper"]
    except KeyError:
        raise ValueError("Method {} unknown.".format(method))
    mini = Minimizer(meth(**options))
    mini.update_tolerances(tol)
    return mini(J, x0)


def scipy_minimize(J, x0, method=None, options=None, tol=None):
    """
    Wrapper around scipy.optimize. Tested are "BFGS", "Newton-CG", "CG", and "trust-ncg".
    """
    def print_info(x_i):
        print "it= {it} / chi^2/m= {chisq} (meas= {chisqm} / apr= {chisqa} ) / |J'|= {normb}".format(
            it=print_info.it, chisq=J.chisq, chisqm=J.chisq_m,
            chisqa=J.chisq_a, normb=la.norm(J.jac(x_i)))
        print_info.it += 1
    print_info.it = 0

    if x0 is None:
        x0 = np.zeros(J.n)
    import scipy.optimize as sopt
    J.init(x0)
    return sopt.minimize(J.__call__, x0, jac=J.jac, hessp=J.hess_dot,
                         method=method, tol=tol, options=options,
                         callback=print_info)

