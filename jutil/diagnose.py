import numpy as np
import jutil


def get_row_of_gain(J, fwd, s_e_inv, x_f, row_idx):
    A = jutil.operator.CostFunctionOperator(J, x_f)
    P = jutil.preconditioner.CostFunctionPreconditioner(J, x_f)
    e_i = np.zeros(J.n)
    e_i[row_idx] = 1

    row_S = conj_grad_solve(A, e_i, P=P)

    std = sqrt(row_S(row_idx));
    gain_row = s_e_inv.dot(fwd.jac_dot(x_f, std))
    avk_row = fwd.jac_T_dot(x_f,  gain_row)
    return std, gain_row, avk_row


def get_measurement_contribution(J, fwd, s_e_inv, x_f):
    A = jutil.operator.CostFunctionOperator(J, x_f)
    P = jutil.preconditioner.CostFunctionPreconditioner(J, x_f)
    ones = np.ones(J.n)

    temp1 = fwd.jac_dot(x_f, ones)
    temp2 = s_e_inv.dot(temp1)
    temp3 = fwd.jac_T_dottemp2)

    mc = conj_grad_solve(A, temp3, P=P)
    return mc
