import numpy as np
import numpy.linalg as la
import cg


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


class QuasiNewton(object):

    def __init__(stepper)
        self._stepper = stepper


    def printInfo(self, it, J, disq, normb):
        print "it= {it} / chi^2/m= {chisq} (meas= {chisqm} / apr= {chisqa} ) / d_i^2/n= {disq} / |J'|= {normb} / Q= {prob}".format(
                it=it, chisq=J.getLastChisq(), chisqm=J.getLastChisqM(),
                chisqa=getLastChisqA(), disq=disq, normb=normb,
                prob=getChiSquareProbability(J.getLastChisq() * J.m, J.m))


    def __call__(J, x_0):
        x_i = np.array(x_0, copy=True)

        J.init(x_i)
        J.updateDiag(x_i)

        disq  = 0.0
        kernel_recomp = False
        last_kernel_recomp = 0
        it = 0
        while True:
            if ((ret__jacobian_matrix_recompute_ > 0) and
                    ((it % ret__jacobian_matrix_recompute_) == 0) and
                    (it > 0)):
                kernel_recomp = True

            if kernel_recomp:
                J.updateKernel(x_i)
                J.updateDiag(x_i)
                kernel_recomp = False
                last_kernel_recomp = it
            else:
                print "Skipping Kernel calculation in this iteration."


            b = -J.dJ(x_i)

            if la.norm(b) / J.m < ret__conv_min_costfunction_gradient_:
                print "Convergence criteria reached (dfmin)")
                if last_kernel_recomp == it:
                    break
                else:
                    # make sure that the Kernel was just recalculated, so the
                    # convergence and the following analysis is correct.
                    print "Performing one more iteration with accurate Kernel"
                    kernel_recomp = True
                    continue

            self.printInfo(it, J, disq, la.norm(b))

            chisq_old = J.getLastChisq()
            x_i += self._stepper(J, b, x_i):
            chisq = J.getLastChisq()

            it += 1

            # normalize step size in state space
            disq = np.dot(x_step, b) / J.n

            # Discrepancy principle
            dp_reached = (np.sqrt(J.getLastChisqM() * J.m) <
                          ret__conv_discrepancy_principle_tau_ *
                          np.sqrt(J.m))

            # Convergence test based on reduction of cost function...
            fmin_reached = 100. * abs(1. - chisq / chisq_old) < ret__conv_min_costfunction_reduction_
            # Convergence test on normalized step size
            dmin_reached = disq < ret__conv_min_normalized_stepsize_
            maxit_reached = it >= ret__conv_max_iteration_

            if dmin_reached or fmin_reached or dp_reached or maxit_reached:
                print "Convergence criteria reached. {dp}{fmin}{dmin}{maxit}".format(
                        dp="(dp)" if dp_reached else "",
                        fmin="(fmin)" if fmin_reached else "",
                        dmin="(dmin)" if dmin_reached else "",
                        maxit="(maxit)" if maxit_reached else "")
            if last_kernel_recomp + 1 == it or dp_reached:
                break
            else:
                # make sure that the Kernel was just recalculated, so the
                # convergence and the following analysis is correct.
                print "Performing one more iteration with accurate Kernel"
                kernel_recomp = True


    b = -J.dJ(x_i)
    self.printInfo(it, J, disq, la.norm(b))

    return x_i

  # ****************************************************************************

class LevenbergMarquardStepMethod(object):
    def __init__(self, solver, lmpar, factor):
        self._lmpar = lmpar
        self._factor = factor
        self._solver = solver

    def __call__(solver, J, b, x_i):
        chisq_old = J.getLastChisq()
        while True:
            # Solve J''(x_i) x_step = J'(x_i)
            x_step = self._solver(J, self._lmpar, x_i, b);

            if np.any(np.isnan(x_step)):
                raise RuntimeError("Retrieval failed (x_step is NaN)! " + repr(x_step))

            x_new = x_i + x_step
            chisq = J(x_new)

            # Modify Levenberg-Marquardt parameter
            if chisq > chisq_old:
                self._lmpar *= self._factor
                if self.lmpar_ > 1e30:
                    raise RuntimeError("Retrieval failed (levenberg marquardt parameter too large)! i" + repr(self._lmpar))
                print "Increasing lmpar to {}: {} > {}".format(self._lmpar, chisq, chisq_old)
            else:
                self._lmpar /= self._factor
                print "Decreasing lmpar to {}: {} < {}".format(self._lmpar, chisq, chisq_old)

                return x_step
  }

  # ****************************************************************************

class GaussNewtonStepMethod(object):
    def __init__(self, solver):
        self._solver = solver
        pass

    def __call__(J, b, x_i):
        # Solve J''(x_i) x_step = J'(x_i)
        x_step = self._solver(0, x_i, b, x_step)
        grad = np.array(-x_step, copy=True)
        x_old = np.array(x_i, copy=True)
        lnsrch(x_old, chisq_old, grad, x_step, x_i,
               chisq, 100.0, check, J);

        return x_step

  # ****************************************************************************

class HybridStepMethod(object):
    def __init__(self):
        passe

    def __call__(J, b, x_i):

    err_rels = [self.ret__cg_conv_rel_]
    while err_rels[-1] < 1:
        err_rels.append(min(err_rels[-1] * self.factor_, 1.0))


    x_steps = cg.conj_grad2_solve(costfunction.jacobiadaptor(J), b, -1, 1e-20, err_rels)

    for i, x_step in enumerate(x_steps):

        x_new = x_i + x_step
        chisq = J(x_new)

        if chisq > chisq_old and i + 1 < len(x_steps):
            MESSAGE(INFO, "Increasing reltol to " << err_rels[i + 1] <<  " " << chisq << " > " << chisq_old);
            continue;

        if chisq > chisq_old and i + 1 == len(x_steps):
            MESSAGE(INFO, "Employing line search.");
            DenseVector grad(-x_step), x_old(x_i);
            bool check;
            la::lnsrch(x_old, chisq_old, grad, x_step, x_i, chisq, 100.0, check, J)
            this->ctl_->ret__cg_conv_rel_ = 1. / self._factor
        else:
            this->ctl_->ret__cg_conv_rel_ = err_rels[i] / self->_factor
       break
    print "Decreasing reltol to", this->ctl_->ret__cg_conv_rel_

    return x_step
  }

  // ****************************************************************************

  bool
  HybridStepMethod2::computeStepSize(double &chisq, const double &chisq_old,
                                     DenseVector& b, DenseVector &x_i, DenseVector &x_step)
  {
    std::unique_ptr<Solver> solver(SolverFactory::create(this->ctl_, this->J_)->createRetrievalSolver());
    // Solve J''(x_i) x_step = J'(x_i)
    (*solver)(this->lmpar_, x_i, b, x_step);

    DenseVector x_new = x_i + x_step;
    chisq = (*this->J_)(x_new);
    if (chisq > chisq_old && this->ctl_->ret__cg_max_iterations_ > 1) {
      this->ctl_->ret__cg_max_iterations_ =
        max(static_cast<int>(this->ctl_->ret__cg_max_iterations_ / this->lmpar_factor_), 1);
      MESSAGE(INFO, "Decreasing maxit to " << this->ctl_->ret__cg_max_iterations_ <<  " " <<
              chisq << " > " << chisq_old);
      return false;
    }
    if (chisq > chisq_old && this->ctl_->ret__cg_max_iterations_ <= 1) {
      MESSAGE(INFO, "Employing line search. (" << chisq << " > " << chisq_old << ")");
      DenseVector grad(-x_step), x_old(x_i);
      bool check;
      la::lnsrch(x_old, chisq_old, grad, x_step, x_i,
                 chisq, 100.0, check, *this->J_);
      this->ctl_->ret__cg_max_iterations_ = 1;
    } else {
      noalias(x_i) = x_new;
    }

    this->ctl_->ret__cg_max_iterations_ = static_cast<int>(
        this->ctl_->ret__cg_max_iterations_ * this->lmpar_factor_);
    MESSAGE(INFO, "Increasing maxit to " << this->ctl_->ret__cg_max_iterations_);
    return true;
  }

  // ****************************************************************************

  bool
  HybridStepMethod3::computeStepSize(double &chisq, const double &chisq_old,
                                     DenseVector& b, DenseVector &x_i, DenseVector &x_step)
  {
    std::unique_ptr<Solver> solver(SolverFactory::create(this->ctl_, this->J_)->createRetrievalSolver());
    // Solve J''(x_i) x_step = J'(x_i)
    (*solver)(this->lmpar_, x_i, b, x_step);

    DenseVector x_new = x_i + x_step;
    chisq = (*this->J_)(x_new);
    if (chisq > chisq_old) {
      MESSAGE(INFO, "Employing line search. (" << chisq << " > " << chisq_old << ")");
      DenseVector grad(-x_step), x_old(x_i);
      bool check;
      la::lnsrch(x_old, chisq_old, grad, x_step, x_i,
                 chisq, 100.0, check, *this->J_);
    } else {
      noalias(x_i) = x_new;
    }

    this->ctl_->ret__cg_max_iterations_ = static_cast<int>(
        this->ctl_->ret__cg_max_iterations_ * this->lmpar_factor_);

    MESSAGE(INFO, "Increasing maxit to " << this->ctl_->ret__cg_max_iterations_);
    return true;
  }

  // ****************************************************************************

  bool
  HybridStepMethod4::computeStepSize(double &chisq, const double &chisq_old,
                                     DenseVector& b, DenseVector &x_i, DenseVector &x_step)
  {
    std::unique_ptr<Solver> solver(SolverFactory::create(this->ctl_, this->J_)->createRetrievalSolver());
    // Solve J''(x_i) x_step = J'(x_i)
    (*solver)(this->lmpar_, x_i, b, x_step);

    DenseVector x_new = x_i + x_step;
    chisq = (*this->J_)(x_new);
    if (chisq > chisq_old) {
      MESSAGE(INFO, "Employing line search. (" << chisq << " > " << chisq_old << ")");
      DenseVector grad(-x_step), x_old(x_i);
      bool check;
      la::lnsrch(x_old, chisq_old, grad, x_step, x_i,
                 chisq, 100.0, check, *this->J_);
    } else {
      noalias(x_i) = x_new;
    }

    this->ctl_->ret__cg_conv_rel_ /= this->lmpar_factor_;
    MESSAGE(INFO, "Decreasing cgrel to " << this->ctl_->ret__cg_conv_rel_);
    return true;
  }

}
