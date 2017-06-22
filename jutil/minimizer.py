#
#
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Ungermann
#

import numpy as np
import numpy.linalg as la
from jutil.lnsrch import lnsrch
from jutil.operator import CostFunctionOperator
from jutil.preconditioner import CostFunctionPreconditioner
from jutil.costfunction import CountingCostFunction, WrapperCostFunction
import jutil.cg as cg
import logging


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


def _print_info(log, it, J, disq, normb):
    chisq_str, disq_str, qstr = "", "", ""
    if hasattr(J, "chisq_m") and hasattr(J, "chisq_a")\
            and J.chisq_m is not None and J.chisq_a is not None:
        chisq_str = " (meas= {chisqm:>10.7f} / apr= {chisqa:>10.7f} )".format(chisqm=J.chisq_m,
                                                                              chisqa=J.chisq_a)
    if disq and not np.isnan(disq):
        disq_str = " / d_i^2/n= {disq:>10.7f}".format(disq=disq)
    if hasattr(J, "m"):
        qstr = " / Q= {:>10.7f}".format(get_chi_square_probability(J.chisq * J.m, J.m))
    chisq_str_complete = 'chi^2/m= {chisq:>10.7f}{chisq_str}{disq_str}'.format(
        chisq=J.chisq, chisq_str=chisq_str, disq_str=disq_str)
    log.info("it= {it:>5} / {chisq_str_complete} / |J'|= {normb:>10.7f}{qstr}".format(
        it=it, chisq_str_complete=chisq_str_complete, normb=normb, qstr=qstr))


class OptimizeResult(dict):
    """
    Dictionary wrapper to return the minimum of a costfunction and ancillary information
    about the minimisation. Modelled after scipy.optimize OptomizeResult for better
    compatibility. Small changes in contained information due to different philosophies.

    Attributes
    ----------
    x : vector
        Identified minimum
    success : bool
        Whether the optimization was successful
    fun : float
        cost function value at minimum
    jac : vector
        gradient of the cost function at minimum
    nfev, njev, nhev, nhdev, nhdiagev : int
        Number of evaluations of costfunction, costfunction gradient, costfunction
        hessian, costfunction hessian dot product, and cost function approximate hessian
        inverse methods
    nit : int
        number of used iterations
    conv : dict
        Dictionary indicating which stopping criteria terminated the optimization.
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


class Minimizer(object):
    """
    Class for computing non-linear minima of cost functions.

    Parameters
    ----------
    stepper : type

    Returns
    -------
    OptimizeResult
        One of the Stepper classes provided in this module.
    """
    def __init__(self, stepper):
        self._stepper = stepper
        self._conv = {
            "min_costfunction_gradient": 0,
            "discrepancy_principle_tau": 0,
            "min_costfunction_reduction": 0,
            "min_normalized_stepsize": 0,
            "max_iteration": 10,
        }
        self._log = logging.getLogger(__name__ + ".Minimizer")

    def update_tolerances(self, tol):
        """
        Allows to set convergence criteria.

        Parameters
        ----------
        tol : dict
            min_costfunction_gradient: Terminate if the gradient of the cost function falls
                                       below this limit.
            discrepancy_principle_tau: Terminate if the norm of the discrepancy of the
                                       measurements falls below this limit (in effect chi^2_m < tau^2) is cheched.
            min_costfunction_reduction: Terminate if the relative reduction of the cost
                                        function in one iteration falls below this limit (given in percent).
            min_normalized_stepsize: Terminate if the scalar product between step in state
                                     space and the current gradient of the cost function divided by the number of
                                     state space elements falls below this number.
            max_iteration: Terminate if the number of iterations surpasses this limit.

        """
        assert all([key in self._conv for key in tol]), tol
        self._conv.update(tol)

    def __call__(self, J, x_0, callback=None):
        """
        Executes the minimization.

        Parameters
        ----------
        J : AbstractCostFunction
            CostFunction
        x_0 : vector
            initial guess

        Returns
        -------
        x : vector
            Minimum
        """

        J = CountingCostFunction(J)
        x_i = x_0.copy()

        if hasattr(J, "init"):
            J.init(x_i)
        if hasattr(self._stepper, "init"):
            self._stepper.init()

        disq = 0.0
        it = 0
        chisq = J.chisq
        converged = {}
        while True:
            if hasattr(J, "update_jacobian") and it > 0:
                J.update_jacobian(x_i)

            b = -J.jac(x_i)

            if la.norm(b) <= self._conv["min_costfunction_gradient"]:
                converged["min_costfunction_gradient"] = True
                self._log.info("Convergence criteria reached. " + str([x for x in converged if converged[x]]))
                break

            _print_info(self._log, it, J, disq, la.norm(b))

            chisq_old = J.chisq
            x_step = self._stepper(J, b, x_i)
            if np.any(np.isnan(x_step)):
                raise RuntimeError("Retrieval failed (x_step is NaN)! " + repr(x_step))

            x_i += x_step
            it += 1

            chisq = J.chisq
            assert chisq <= chisq_old, (chisq, chisq_old, chisq_old - chisq)

            # normalize step size in state space
            disq = np.dot(x_step, b) / J.n

            if hasattr(J, "chisq_m") and J.chisq_m is not None:
                converged["discrepancy_principle_tau"] = (
                    J.chisq_m < self._conv["discrepancy_principle_tau"] ** 2)
            converged.update({
                "min_costfunction_reduction":
                    100. * abs(1. - chisq / chisq_old) <= self._conv["min_costfunction_reduction"],
                "min_normalized_stepsize":
                    disq <= self._conv["min_normalized_stepsize"],
                "max_iteration": it >= self._conv["max_iteration"]})

            if any(converged.values()):
                self._log.info("Convergence criteria reached. " + str([x for x in converged if converged[x]]))
                break

            if callback:
                callback(x_i)

        if hasattr(J, "update_jacobian"):
            J.update_jacobian(x_i)
        b = -J.jac(x_i)
        _print_info(self._log, it, J, disq, la.norm(b))

        self._log.debug("J: call={} jac={} hess_dot={} hess={} hess_diag={}".format(
            J.cnt_call, J.cnt_jac, J.cnt_hess_dot, J.cnt_hess, J.cnt_hess_diag))

        result = OptimizeResult({
            "x": x_i,
            "success": True,
            "fun": chisq,
            "jac": -b,
            "nit": it,
            "nfev": J.cnt_call,
            "njev": J.cnt_jac,
            "nhev": J.cnt_hess,
            "nhdev": J.cnt_hess_dot,
            "nhdiagev": J.cnt_hess_diag,
            "conv": converged,
        })
        return result


class LevenbergMarquardtAbstractBase(object):
    def __init__(self, lmpar=1, factor=10,
                 cg_max_iter=-1, cg_tol_rel=1e-20, cg_tol_abs=1e-20):
        self._lmpar_init = lmpar
        self._lmpar = self._lmpar_init
        self._lmpar_factor = factor
        self._cg_max_iter = cg_max_iter
        self._cg_tol_rel = cg_tol_rel
        self._cg_tol_abs = cg_tol_abs
        self._log = logging.getLogger(__name__ + ".LevenbergMarquardt")

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
                    raise RuntimeError(
                        "Retrieval failed (levenberg marquardt parameter too large)! i" +
                        repr(self._lmpar))
                self._log.info("Increasing lmpar to {} ({} > {})".format(self._lmpar, chisq, chisq_old))
            else:
                self._lmpar /= self._lmpar_factor
                self._log.info("Decreasing lmpar to {} ({} < {})".format(self._lmpar, chisq, chisq_old))

                return x_step


class LevenbergMarquardtPredictorStepper(LevenbergMarquardtAbstractBase):
    def __call__(self, J, b, x_i):
        chisq_old = J.chisq
        while True:
            x_step = cg.conj_grad_solve(
                CostFunctionOperator(J, x_i, lmpar=self._lmpar), b,
                max_iter=self._cg_max_iter, abs_tol=self._cg_tol_abs, rel_tol=self._cg_tol_rel)
            x_new = x_i + x_step

            chisq_pred = chisq_old - np.dot(b, x_step) + 0.5 * np.dot(x_step, J.hess_dot(x_i, x_step))
            chisq = J(x_new)

            chisq_factor = chisq_pred / chisq

            if chisq_factor < 0.25:
                self._lmpar *= self._lmpar_factor
                if self._lmpar > 1e30:
                    raise RuntimeError(
                        "Retrieval failed (levenberg marquardt parameter too large)! i" +
                        repr(self._lmpar))
                self._log.info("Increasing lmpar to {} ({} < 0.25)".format(self._lmpar, chisq_factor))
            elif chisq_factor > 0.5:
                self._lmpar /= self._lmpar_factor
                self._log.info("Decreasing lmpar to {} ({} > 0.50)".format(self._lmpar, chisq_factor))
            if chisq <= chisq_old:
                return x_step


class SteepestDescentStepper(object):
    def __init__(self, preconditioner=lambda _, x: x):
        self._preconditioner = preconditioner

    def __call__(self, J, b, x_i):
        chisq_old = J.chisq
        _, _, x_new = lnsrch(x_i, chisq_old, -b, self._preconditioner(x_i, b), J)
        return x_new - x_i


class CauchyPointSteepestDescentStepper(object):
    def __init__(self, preconditioner=lambda _, x: x):
        self._preconditioner = preconditioner

    def __call__(self, J, b, x_i):
        chisq_old = J.chisq
        # This computes the optimal distance along the steepest descent
        # direction
        p = self._preconditioner(x_i, b)
        direc = (np.dot(b, p) / np.dot(J.hess_dot(x_i, p), p)) * p
        _, _, x_new = lnsrch(x_i, chisq_old, -b, direc, J)

        return x_new - x_i


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

        return x_new - x_i


class TruncatedCGGaussNewtonStepper(object):
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

        return x_new - x_i


TruncatedCGQuasiNewtonStepper = TruncatedCGGaussNewtonStepper


class TruncatedCGTrustRegionStepper(object):
    """
    todo newton requires dampening (p - 1) for l_p normed cost functions?
    """
    def __init__(self, conv_rel=1e-4, factor=10, cg_max_iter=-1, verbose=False):
        self._conv_rel_init = conv_rel
        self._conv_rel = self._conv_rel_init
        self._factor = factor
        self._cg_max_iter = cg_max_iter
        self._verbose = verbose
        self._log = logging.getLogger(__name__ + ".TruncatedCGTrustRegionStepper")

    def _get_err_rels(self):
        result = [self._conv_rel]
        if self._conv_rel == 0:
            raise ValueError("relative convergence set to 0! Choose a positive value.")
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
            P=CostFunctionPreconditioner(J, x_i),
            max_iter=self._cg_max_iter, rel_tol=err_rels, abs_tol=0, verbose=self._verbose)

        delta_chisq_preds = [- np.dot(b, x_step) + 0.5 * np.dot(x_step, J.hess_dot(x_i, x_step)) for x_step in x_steps]
        for i, (x_step, delta_chisq_pred) in enumerate(zip(x_steps, delta_chisq_preds)):
            x_new = x_i + x_step
            chisq = J(x_new)
            min_reduction = delta_chisq_pred * 0.1  # require 10% of predicted reduction

            self._log.info("  try {} with err_rel= {} , new chisq= {} , old chisq= {}, pred. chisq= {}".format(
                i, err_rels[i], chisq, chisq_old, chisq_old + delta_chisq_pred))
            if chisq > chisq_old + min_reduction and i + 1 < len(x_steps):
                continue
            if chisq > chisq_old + min_reduction and i + 1 == len(x_steps):
                self._log.info("  CG steps exhausted. Employing line search.")
                _, chisq, x_new = lnsrch(x_i, chisq_old, -b, x_step, J)
                x_step = x_new - x_i
                self._conv_rel = 1. / self._factor
            else:
                self._conv_rel = err_rels[i] / self._factor
            break
        self._log.info("  Setting reltol to {} ({}<{})".format(self._conv_rel, chisq, chisq_old))
        return x_step


TrustRegionTruncatedCGQuasiNewtonStepper = TruncatedCGTrustRegionStepper


class TruncatedCGTrustRegionStepper2(object):
    """
    todo newton requires dampening (p - 1) for l_p normed cost functions?
    """
    def __init__(self, conv_rel=1e-4, factor=10, cg_max_iter=-1, verbose=False):
        self._conv_rel_init = conv_rel
        self._conv_rel = self._conv_rel_init
        self._factor = factor
        self._cg_max_iter = cg_max_iter
        self._verbose = verbose
        self._log = logging.getLogger(__name__ + ".TruncatedCGTrustRegion2Stepper")

    def _get_err_rels(self):
        result = [self._conv_rel]
        if self._conv_rel == 0:
            raise ValueError("relative convergence set to 0! Choose a positive value.")
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
            P=CostFunctionPreconditioner(J, x_i),
            max_iter=self._cg_max_iter, rel_tol=err_rels, abs_tol=0, verbose=self._verbose)

        delta_chisq_preds = [- np.dot(b, x_step) + 0.5 * np.dot(x_step, J.hess_dot(x_i, x_step)) for x_step in x_steps]
        last_step = None
        for i, (x_step, delta_chisq_pred) in enumerate(zip(x_steps, delta_chisq_preds)):
            if last_step is not None and np.all(last_step == x_step):
                continue
            self._conv_rel = err_rels[i]

            x_new = x_i + x_step
            chisq_pred = chisq_old + delta_chisq_pred
            chisq = J(x_new)
            chisq_factor = chisq_pred / chisq

            self._log.info(
                "  try {} with err_rel= {} , new chisq= {} , old chisq= {}, chisq_pred= {}, "
                "chisq_factor= {}".format(
                    i, err_rels[i], chisq, chisq_old, chisq_pred, chisq_factor))

            if chisq < chisq_old + 0.2 * delta_chisq_pred:
                self._conv_rel = err_rels[i]
                if chisq_factor < 0.25:
                    self._conv_rel *= self._factor
                if chisq_factor > 0.5:
                    self._conv_rel /= self._factor
                self._log.info("  Setting reltol to {} ({}<{})".format(self._conv_rel, chisq, chisq_old))
                return x_step

            last_step = x_step

        self._log.info("  CG steps exhausted. Employing line search.")
        _, chisq, x_new = lnsrch(x_i, chisq_old, -b, x_steps[-1], J)
        x_step = x_new - x_i
        self._conv_rel = 1. / self._factor

        self._log.info("  Setting reltol to {} ({}<{})".format(self._conv_rel, chisq, chisq_old))
        return x_step


def minimize(J, x0, method="TruncatedCGTrustRegion", options={}, tol={}, callback=None):
    """
    Front-end for JUTIL non-linear minimization.

    Supported methods area

    * SteepestDescent
    * CauchyPointSteepestDescent
    * LevenbergMarquardtReduction
    * LevenbergMarquardtPredictor
    * GaussNewton
    * TruncatedCGQuasiNewton
    * TrustRegionTruncatedCGQuasiNewton(default)

    Parameters
    ----------
    J : type
        CostFunction
    x0 : vector
        inital guess
    method : str
        String determining method
    options : dict
        Additional parameters for the chosen method
    tol : dict
        convergence options for the outer loop
    """

    try:
        meth = globals()[method + "Stepper"]
    except KeyError:
        raise ValueError("Method {} unknown.".format(method))
    mini = Minimizer(meth(**options))
    mini.update_tolerances(tol)
    return mini(J, x0, callback=callback)


def scipy_minimize(J, x0, method=None, options=None, tol=None):
    """
    Wrapper around scipy.optimize. Tested are "BFGS", "Newton-CG", "CG", and "trust-ncg".

    Parameters
    ----------
    J : AbstractCostFunction
        CostFunction
    x0 : vector
        inital guess
    method : str
        String determining method
    options : dict
        Additional parameters for the chosen method
    tol : dict
        convergence options for the outer loop
    """
    log = logging.getLogger(__name__)

    def print_info(x_i):
        normb = la.norm(J.jac(x_i))
        _print_info(log, print_info.it, J, None, normb)
        print_info.it += 1
    print_info.it = 0

    if x0 is None:
        x0 = np.zeros(J.n)
    import scipy.optimize as sopt
    J.init(x0)
    return sopt.minimize(J.__call__, x0, jac=J.jac, hessp=J.hess_dot,
                         method=method, tol=tol, options=options,
                         callback=print_info)


def scipy_custom_method(
        fun, x0, args, jac, hess, hessp, bounds, constraints, callback, **options):
    """
    This function may be passed as custom method to scipy.optimize minimize routine to
    minimize an arbitrary function with jutil algorithms. Only efficient if jac and hessp
    are provided.
    """
    assert bounds is None, "bounds are not supported"
    J = WrapperCostFunction(
        fun, len(x0), 1, jac=jac, hess=hess, hess_dot=hessp)
    return minimize(J, x0, options["method"], options, callback=callback)
