import numpy as np

import jutil

jutil.misc.setup_logging()


Boltzmann_Constant = 1.3806488e-23
First_Radiation_Constant = 1.191042869e-8
Second_Radiation_Constant = 1.4387770


Atmosphere = {
    "altitude": np.arange(121),
    "temperature": np.asarray([300.93, 294.35, 288.49, 282.9, 277.88, 272.1, 266.63, 260.16, 254.04, 246.39, 239.28, 230.84, 222.77, 214.42, 206.79, 200.62, 197.28, 197.9, 200.18, 203.64, 206.89, 209.77, 212.33, 214.81, 217.09, 219.26, 221.28, 223.18, 225.22, 227.24, 229.72, 232.5, 235.5, 238.74, 241.73, 244.82, 247.6, 250.37, 252.92, 255.39, 257.7, 259.87, 261.84, 263.62, 265.07, 266.27, 267.13, 267.78, 268.0, 268.0, 267.46, 266.67, 265.36, 263.87, 261.86, 259.76, 257.08, 254.42, 251.32, 248.41, 244.95, 241.75, 238.14, 234.72, 231.13, 227.62, 224.32, 221.1, 218.21, 215.51, 213.21, 211.24, 209.9, 208.83, 208.02, 207.44, 207.04, 206.85, 206.9, 206.99, 206.92, 206.48, 205.82, 204.54, 202.75, 200.62, 197.97, 195.52, 193.26, 191.17, 189.57, 188.36, 187.51, 187.14, 187.0, 187.0, 186.93, 186.97, 187.11, 187.32, 187.72, 188.28, 189.08, 190.45, 192.17, 194.71, 198.31, 202.23, 207.88, 215.04, 223.38, 235.47, 247.21, 262.86, 277.31, 293.96, 309.86, 325.23, 341.44, 356.28, 370.68]),
    "pressure": np.asarray([1017.0, 907.019, 806.988, 716.336, 634.46, 560.626, 494.126, 434.287, 380.507, 332.189, 288.826, 249.975, 215.226, 184.277, 156.861, 132.803, 111.995, 94.3416, 79.5743, 67.2848, 57.0525, 48.4962, 41.3117, 35.2599, 30.1487, 25.8209, 22.148, 19.0242, 16.3636, 14.0948, 12.159, 10.5072, 9.0966, 7.89067, 6.85755, 5.97041, 5.20685, 4.54807, 3.97849, 3.48499, 3.0566, 2.68402, 2.35939, 2.07601, 1.82817, 1.611, 1.42038, 1.25281, 1.10528, 0.975208, 0.860369, 0.758846, 0.668998, 0.589418, 0.518888, 0.456357, 0.400901, 0.351723, 0.308131, 0.269523, 0.235356, 0.205148, 0.178477, 0.154959, 0.13426, 0.116073, 0.100135, 0.0862029, 0.0740581, 0.0635031, 0.0543574, 0.0464585, 0.0396603, 0.0338278, 0.0288338, 0.0245651, 0.0209215, 0.0178152, 0.01517, 0.0129189, 0.0110025, 0.009369, 0.00797499, 0.00678357, 0.0057634, 0.00488915, 0.00413956, 0.00349753, 0.0029492, 0.00248218, 0.00208573, 0.00175044, 0.00146771, 0.00123002, 0.00103062, 0.000863542, 0.000723564, 0.000606301, 0.000508113, 0.00042592, 0.000357145, 0.000299626, 0.000251543, 0.0002114, 0.000177922, 0.000150038, 0.000126868, 0.00010762, 9.1648e-05, 7.84309e-05, 6.74937e-05, 5.8474e-05, 5.10218e-05, 4.48483e-05, 3.97076e-05, 3.53912e-05, 3.17414e-05, 2.86218e-05, 2.59365e-05, 2.3607e-05, 2.15688e-05]),
    "CO2": np.asarray([0.0003685, 0.0003685, 0.0003685, 0.0003685, 0.0003685, 0.0003685, 0.0003685, 0.0003685, 0.0003685, 0.0003685, 0.0003685, 0.0003685, 0.0003685, 0.0003685, 0.0003684, 0.000368, 0.0003673, 0.0003663, 0.0003652, 0.0003642, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.000363, 0.0003625, 0.0003617, 0.0003604, 0.0003586, 0.0003564, 0.0003541, 0.0003519, 0.0003495, 0.0003471, 0.0003447, 0.0003421, 0.0003394, 0.0003364, 0.0003332, 0.0003295, 0.0003252, 0.00032, 0.0003141, 0.0003074, 0.0002996, 0.000291, 0.0002817, 0.0002714, 0.0002602, 0.0002481, 0.0002353, 0.0002223, 0.0002091, 0.0001955, 0.0001821, 0.000169, 0.0001563, 0.0001443, 0.0001331, 0.0001238, 0.0001154, 0.0001078, 9.947e-05, 9.319e-05, 8.886e-05, 8.465e-05, 8.051e-05, 7.654e-05, 7.268e-05, 6.894e-05, 6.525e-05, 6.159e-05, 5.798e-05, 5.442e-05, 5.089e-05, 4.737e-05, 4.385e-05]),
}
for key in Atmosphere:
    Atmosphere[key] = Atmosphere[key][::-1]


def planck(temperature, wavenumber):
    fnord1 = Second_Radiation_Constant * wavenumber / temperature
    fnord2 = np.expm1(fnord1)
    result = First_Radiation_Constant * (wavenumber ** 3) / fnord2
    return result, (result / fnord2) * (fnord2 + 1) * fnord1 / temperature


def convert_vmr_to_numberdensity(temperature, pressure, vmr):
    result = vmr * pressure * 10. / (Boltzmann_Constant * temperature)
    return result, -result / temperature


def convert_vmr_to_emissivity(temperature, pressure, vmr, xsc):
    numberdensity, numberdensity_adj = convert_vmr_to_numberdensity(temperature, pressure, vmr)
    emissivity = -np.expm1(-xsc * numberdensity)
    emissivity_adj = xsc * np.exp(-xsc * numberdensity) * numberdensity_adj
    return emissivity, emissivity_adj


def model(wavenumber, pressures, xsc, temperatures, vmrs):
    emissivity, emissivity_adj = convert_vmr_to_emissivity(temperatures, pressures, vmrs, xsc)
    source, source_adj = planck(temperatures, wavenumber)
    transmissivity = 1
    radiance = 0
    radiance_adj = np.zeros_like(temperatures)
    transmissivity_adj = np.zeros_like(temperatures)
    for i, (s, s_adj, e, e_adj) in enumerate(zip(source, source_adj, emissivity, emissivity_adj)):
        radiance += s * transmissivity * e

        radiance_adj[i] = s * transmissivity * e_adj + s_adj * transmissivity * e
        radiance_adj[:i] += transmissivity_adj[:i] * s * e

        transmissivity_adj[:i] *= (1. - e)
        transmissivity_adj[i] -= transmissivity * e_adj

        transmissivity = transmissivity * (1. - e)
    return radiance, radiance_adj


class ForwardModel(object):
    """
    This class provides a very basic approximation to an actual
    radiative transport model to do nadir sounding temperature
    measurements. It simulates that the atmosphere emits and absorbs
    with different strength at different altitudes, which allows
    the reconstruction of a height-resolved temperature profile.

    This class offers a very generic, but efficient interface to a
    model for the purposes of inverse modelling. It allows to be called
    with a parameter vector (here: atmospheric temperatures) and returns
    a set of modelled output parameters (here: radiances at different channels).

    In addition, it has a routine to compute and return the Jacobian matrix
    and a routine to compute the product of a vector with the Jacobian. and the
    product of its transpose. please note that the inversion only needs these
    two routines and not the Jacobian itself (i.e. a matrix free implementation
    is feasible.)
    """

    def __init__(self, xscs):
        self._pressures = Atmosphere["pressure"]
        self._vmrs = Atmosphere["CO2"]
        self._xscs = xscs
        self._temperatures = None
        self.n, self.m = len(self._vmrs), len(self._xscs)

    def _update(self, x):
        if self._temperatures is None or np.any(self._temperatures != x):
            x[x <= 0] = 1e-12
            self._temperatures = x
            results = [model(792, self._pressures, xsc, self._temperatures, self._vmrs)
                       for xsc in self._xscs]
            self._radiances, self._jacobian = [np.asarray([_x[i] for _x in results]) for i in [0, 1]]

    def __call__(self, x):
        self._update(x)
        return self._radiances

    def jac(self, x):
        self._update(x)
        return self._jacobian

    def jac_dot(self, x, vec):
        return self.jac(x).dot(vec)

    def jac_T_dot(self, x, vec):
        return self.jac(x).T.dot(vec)


class CostFunction(jutil.costfunction.LeastSquaresCostFunction):
    """
    This cost function wraps the ForwardModel together with a
    basic regularization based on finite differences.

    Please note that this implementation is matrix free aside the
    sparse finite difference operator.
    """

    def __init__(self, lam):
        import scipy.sparse
        self._F = ForwardModel(
            [1e-14, 1e-15, 1e-16, 1.2e-16, 1e-17, 1.1e-17,
             1e-18, 1e-19, 9.9e-19, 1e-20, 1.1e-20, 1e-21, 1e-22])
        self._x_t = Atmosphere["temperature"]
        self._lambda = lam
        self._rad_t = self._F(self._x_t)
        self._rad = self._rad_t * (
            100 + 5 * np.random.randn(len(self._rad_t))) / 100.

        m = len(self._rad)
        n = len(self._x_t)
        self._D = scipy.sparse.lil_matrix((n, n))
        for i in range(0, n - 1):
            self._D[i, i] = 1
            self._D[i, i + 1] = -1
        self._D = self._D.tocsr()

        def func(x):
            """
            The actual costfunction. This implements
            (|| F(x) - y ||_2^2 + lambda ||d/dz x||_2^2) / m

            The division by m allows an interpretation in a statistical
            chi^2 style.
            """
            sqrt_m = m ** 0.5
            dy = (self._F(x) - self._rad) / sqrt_m
            dx = ((self._lambda / self.m) ** 0.5) * self._D.dot(x)
            value = np.concatenate([dy, dx])
            model_operator = jutil.operator.Function(
                shape=(m, n),
                F=lambda vec: self._F.jac_dot(x, vec),
                FT=lambda vec: self._F.jac_T_dot(x, vec)
            )
            derivative = jutil.operator.Scale(
                jutil.operator.VStack([
                    model_operator,
                    jutil.operator.Scale(
                        self._D, self._lambda)]), 1 / sqrt_m)
            return value, derivative

        super().__init__(
            func, n, m,
            func_returns_both=True)

    def hess_diag(self, x):
        return np.ones_like(x)


def main():

    import jutil.minimizer as mini
    import matplotlib.pyplot as plt
    # The cost function. lam determines regularization strength
    # Vary by (small) orders of magnitude to see the impact
    J = CostFunction(lam=1e-8)
    # Initiate a basic Levenberg-Marquardt minimization
    # with at most 20 iterations
    optimize = mini.Minimizer(
        mini.LevenbergMarquardtReductionStepper(1, 10.),
        tol={"max_iteration": 20})
    result = optimize(J, 220 * np.ones_like(J._x_t))
    x_t = J._x_t

    plt.subplot(2, 1, 1)
    plt.plot(x_t, Atmosphere["altitude"], label="true atmospheric temperature")
    plt.plot(result["x"], Atmosphere["altitude"], label="retrieved atmospheric temperature")
    plt.ylabel("altitude (km)")
    plt.xlabel("temperature (K)")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.title("sensitivities of different channels")
    plt.plot(J._F.jac(x_t).T, Atmosphere["altitude"])
    plt.ylabel("altitude (km)")
    plt.show()


main()
