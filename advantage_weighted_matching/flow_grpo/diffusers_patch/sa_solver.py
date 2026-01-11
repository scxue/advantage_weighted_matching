import torch
import torch.nn.functional as F
import math
from tqdm import tqdm

class NoiseScheduleFlowMatching:
    def __init__(
            self,
            dtype=torch.float32,
    ):
        """Create a wrapper class for the forward SDE (Flow Matching type).
        ***
        The forward SDE ensures that the condition distribution q_{t|0}(x_t | x_0) = N ( alpha_t * x_0, sigma_t^2 * I ).
        For flow matching, alpha_t = 1-t and sigma_t = t.
        We further define lambda_t = log(alpha_t) - log(sigma_t), which is the half-logSNR (described in the DPM-Solver paper).
        Therefore, we implement the functions for computing alpha_t, sigma_t and lambda_t. For t in [0, T], we have:
            log_alpha_t = self.marginal_log_mean_coeff(t)
            sigma_t = self.marginal_std(t)
            lambda_t = self.marginal_lambda(t)
        Moreover, as lambda(t) is an invertible function, we also support its inverse function:
            t = self.inverse_lambda(lambda_t)
        Returns:
            A wrapper object of the forward SDE (VP type).

        ===============================================================
        Example:
        >>> ns = NoiseScheduleFlowMatching()
        """
        self.T = 1

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        return torch.log(1. - t)

    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        return 1. - t

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return t

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = torch.log(self.marginal_alpha(t))
        log_std = torch.log(self.marginal_std(t))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        t = 1. / (1. + torch.exp(lamb))
        return t

    def edm_sigma(self, t):
        return self.marginal_std(t) / self.marginal_alpha(t)

    def edm_inverse_sigma(self, edmsigma):
        t = edmsigma / (1. + edmsigma)
        return t


class SASolver:
    def __init__(
            self,
            model_fn,
            noise_schedule,
            algorithm_type="data_prediction",
    ):
        """
        Construct a SA-Solver
        The default value for algorithm_type is "data_prediction" and we recommend not to change it to
        "noise_prediction". For details, please see Appendix A.2.4 in SA-Solver paper https://arxiv.org/pdf/2309.05019.pdf
        """

        self.model = lambda x, t: model_fn(x, t)
        self.noise_schedule = noise_schedule
        assert algorithm_type in ["data_prediction", "noise_prediction"]

        self.predict_x0 = algorithm_type == "data_prediction"

    def noise_prediction_fn(self, x, t):
        """
        Return the noise prediction model.
        """
        return x + (1-t) * self.model(x,t)

    def data_prediction_fn(self, x, t):
        """
        Return the data prediction model
        """
        return x - t * self.model(x,t)

    def model_fn(self, x, t):
        """
        Convert the model to the noise prediction model or the data prediction model.
        """
        if self.predict_x0:
            return self.data_prediction_fn(x, t)
        else:
            return self.noise_prediction_fn(x, t)


    def get_coefficients_exponential_negative(self, order, interval_start, interval_end):
        """
        Calculate the integral of exp(-x) * x^order dx from interval_start to interval_end
        For calculating the coefficient of gradient terms after the lagrange interpolation,
        see Eq.(15) and Eq.(18) in SA-Solver paper https://arxiv.org/pdf/2309.05019.pdf
        For noise_prediction formula.
        """
        assert order in [0, 1, 2, 3], "order is only supported for 0, 1, 2 and 3"

        if order == 0:
            return torch.exp(-interval_end) * (torch.exp(interval_end - interval_start) - 1)
        elif order == 1:
            return torch.exp(-interval_end) * (
                        (interval_start + 1) * torch.exp(interval_end - interval_start) - (interval_end + 1))
        elif order == 2:
            return torch.exp(-interval_end) * (
                        (interval_start ** 2 + 2 * interval_start + 2) * torch.exp(interval_end - interval_start) - (
                            interval_end ** 2 + 2 * interval_end + 2))
        elif order == 3:
            return torch.exp(-interval_end) * (
                        (interval_start ** 3 + 3 * interval_start ** 2 + 6 * interval_start + 6) * torch.exp(
                    interval_end - interval_start) - (interval_end ** 3 + 3 * interval_end ** 2 + 6 * interval_end + 6))

    def get_coefficients_exponential_positive(self, order, interval_start, interval_end, tau):
        """
        Calculate the integral of exp(x(1+tau^2)) * x^order dx from interval_start to interval_end
        For calculating the coefficient of gradient terms after the lagrange interpolation,
        see Eq.(15) and Eq.(18) in SA-Solver paper https://arxiv.org/pdf/2309.05019.pdf
        For data_prediction formula.
        """
        assert order in [0, 1, 2, 3], "order is only supported for 0, 1, 2 and 3"

        # after change of variable(cov)
        interval_end_cov = (1 + tau ** 2) * interval_end
        interval_start_cov = (1 + tau ** 2) * interval_start

        if order == 0:
            return torch.exp(interval_end_cov) * (1 - torch.exp(-(interval_end_cov - interval_start_cov))) / (
            (1 + tau ** 2))
        elif order == 1:
            return torch.exp(interval_end_cov) * ((interval_end_cov - 1) - (interval_start_cov - 1) * torch.exp(
                -(interval_end_cov - interval_start_cov))) / ((1 + tau ** 2) ** 2)
        elif order == 2:
            return torch.exp(interval_end_cov) * ((interval_end_cov ** 2 - 2 * interval_end_cov + 2) - (
                        interval_start_cov ** 2 - 2 * interval_start_cov + 2) * torch.exp(
                -(interval_end_cov - interval_start_cov))) / ((1 + tau ** 2) ** 3)
        elif order == 3:
            return torch.exp(interval_end_cov) * (
                        (interval_end_cov ** 3 - 3 * interval_end_cov ** 2 + 6 * interval_end_cov - 6) - (
                            interval_start_cov ** 3 - 3 * interval_start_cov ** 2 + 6 * interval_start_cov - 6) * torch.exp(
                    -(interval_end_cov - interval_start_cov))) / ((1 + tau ** 2) ** 4)

    def lagrange_polynomial_coefficient(self, order, lambda_list):
        """
        Calculate the coefficient of lagrange polynomial
        For lagrange interpolation
        """
        assert order in [0, 1, 2, 3]
        assert order == len(lambda_list) - 1
        if order == 0:
            return [[1]]
        elif order == 1:
            return [[1 / (lambda_list[0] - lambda_list[1]), -lambda_list[1] / (lambda_list[0] - lambda_list[1])],
                    [1 / (lambda_list[1] - lambda_list[0]), -lambda_list[0] / (lambda_list[1] - lambda_list[0])]]
        elif order == 2:
            denominator1 = (lambda_list[0] - lambda_list[1]) * (lambda_list[0] - lambda_list[2])
            denominator2 = (lambda_list[1] - lambda_list[0]) * (lambda_list[1] - lambda_list[2])
            denominator3 = (lambda_list[2] - lambda_list[0]) * (lambda_list[2] - lambda_list[1])
            return [[1 / denominator1,
                     (-lambda_list[1] - lambda_list[2]) / denominator1,
                     lambda_list[1] * lambda_list[2] / denominator1],

                    [1 / denominator2,
                     (-lambda_list[0] - lambda_list[2]) / denominator2,
                     lambda_list[0] * lambda_list[2] / denominator2],

                    [1 / denominator3,
                     (-lambda_list[0] - lambda_list[1]) / denominator3,
                     lambda_list[0] * lambda_list[1] / denominator3]
                    ]
        elif order == 3:
            denominator1 = (lambda_list[0] - lambda_list[1]) * (lambda_list[0] - lambda_list[2]) * (
                        lambda_list[0] - lambda_list[3])
            denominator2 = (lambda_list[1] - lambda_list[0]) * (lambda_list[1] - lambda_list[2]) * (
                        lambda_list[1] - lambda_list[3])
            denominator3 = (lambda_list[2] - lambda_list[0]) * (lambda_list[2] - lambda_list[1]) * (
                        lambda_list[2] - lambda_list[3])
            denominator4 = (lambda_list[3] - lambda_list[0]) * (lambda_list[3] - lambda_list[1]) * (
                        lambda_list[3] - lambda_list[2])
            return [[1 / denominator1,
                     (-lambda_list[1] - lambda_list[2] - lambda_list[3]) / denominator1,
                     (lambda_list[1] * lambda_list[2] + lambda_list[1] * lambda_list[3] + lambda_list[2] * lambda_list[
                         3]) / denominator1,
                     (-lambda_list[1] * lambda_list[2] * lambda_list[3]) / denominator1],

                    [1 / denominator2,
                     (-lambda_list[0] - lambda_list[2] - lambda_list[3]) / denominator2,
                     (lambda_list[0] * lambda_list[2] + lambda_list[0] * lambda_list[3] + lambda_list[2] * lambda_list[
                         3]) / denominator2,
                     (-lambda_list[0] * lambda_list[2] * lambda_list[3]) / denominator2],

                    [1 / denominator3,
                     (-lambda_list[0] - lambda_list[1] - lambda_list[3]) / denominator3,
                     (lambda_list[0] * lambda_list[1] + lambda_list[0] * lambda_list[3] + lambda_list[1] * lambda_list[
                         3]) / denominator3,
                     (-lambda_list[0] * lambda_list[1] * lambda_list[3]) / denominator3],

                    [1 / denominator4,
                     (-lambda_list[0] - lambda_list[1] - lambda_list[2]) / denominator4,
                     (lambda_list[0] * lambda_list[1] + lambda_list[0] * lambda_list[2] + lambda_list[1] * lambda_list[
                         2]) / denominator4,
                     (-lambda_list[0] * lambda_list[1] * lambda_list[2]) / denominator4]

                    ]

    def get_coefficients_fn(self, order, interval_start, interval_end, lambda_list, tau):
        """
        Calculate the coefficient of gradients.
        """
        assert order in [1, 2, 3, 4]
        assert order == len(lambda_list), 'the length of lambda list must be equal to the order'
        coefficients = []
        lagrange_coefficient = self.lagrange_polynomial_coefficient(order - 1, lambda_list)
        for i in range(order):
            coefficient = sum(
                lagrange_coefficient[i][j]
                * self.get_coefficients_exponential_positive(
                    order - 1 - j, interval_start, interval_end, tau
                )
                if self.predict_x0
                else lagrange_coefficient[i][j]
                * self.get_coefficients_exponential_negative(
                    order - 1 - j, interval_start, interval_end
                )
                for j in range(order)
            )
            coefficients.append(coefficient)
        assert len(coefficients) == order, 'the length of coefficients does not match the order'
        return coefficients

    def adams_bashforth_update(self, order, x, tau, model_prev_list, t_prev_list, noise, t):
        """
        SA-Predictor, without the "rescaling" trick in Appendix D in SA-Solver paper https://arxiv.org/pdf/2309.05019.pdf
        """
        assert order in [1, 2, 3, 4], "order of stochastic adams bashforth method is only supported for 1, 2, 3 and 4"

        # get noise schedule
        ns = self.noise_schedule
        alpha_t = ns.marginal_alpha(t)
        sigma_t = ns.marginal_std(t)
        lambda_t = ns.marginal_lambda(t)
        alpha_prev = ns.marginal_alpha(t_prev_list[-1])
        sigma_prev = ns.marginal_std(t_prev_list[-1])
        gradient_part = torch.zeros_like(x)
        h = lambda_t - ns.marginal_lambda(t_prev_list[-1])
        lambda_list = [ns.marginal_lambda(t_prev_list[-(i + 1)]) for i in range(order)]
        gradient_coefficients = self.get_coefficients_fn(order, ns.marginal_lambda(t_prev_list[-1]), lambda_t,
                                                         lambda_list, tau)

        for i in range(order):
            if self.predict_x0:
                gradient_part += (1 + tau ** 2) * sigma_t * torch.exp(- tau ** 2 * lambda_t) * gradient_coefficients[
                    i] * model_prev_list[-(i + 1)]
            else:
                gradient_part += -(1 + tau ** 2) * alpha_t * gradient_coefficients[i] * model_prev_list[-(i + 1)]

        if self.predict_x0:
            noise_part = sigma_t * torch.sqrt(1 - torch.exp(-2 * tau ** 2 * h)) * noise
        else:
            noise_part = tau * sigma_t * torch.sqrt(torch.exp(2 * h) - 1) * noise

        if self.predict_x0:
            x_t = torch.exp(-tau ** 2 * h) * (sigma_t / sigma_prev) * x + gradient_part + noise_part
        else:
            x_t = (alpha_t / alpha_prev) * x + gradient_part + noise_part

        return x_t

    def adams_moulton_update(self, order, x, tau, model_prev_list, t_prev_list, noise, t):
        """
        SA-Corrector, without the "rescaling" trick in Appendix D in SA-Solver paper https://arxiv.org/pdf/2309.05019.pdf
        """

        assert order in [1, 2, 3, 4], "order of stochastic adams bashforth method is only supported for 1, 2, 3 and 4"

        # get noise schedule
        ns = self.noise_schedule
        alpha_t = ns.marginal_alpha(t)
        sigma_t = ns.marginal_std(t)
        lambda_t = ns.marginal_lambda(t)
        alpha_prev = ns.marginal_alpha(t_prev_list[-1])
        sigma_prev = ns.marginal_std(t_prev_list[-1])
        gradient_part = torch.zeros_like(x)
        h = lambda_t - ns.marginal_lambda(t_prev_list[-1])
        t_list = t_prev_list + [t]
        lambda_list = [ns.marginal_lambda(t_list[-(i + 1)]) for i in range(order)]
        gradient_coefficients = self.get_coefficients_fn(order, ns.marginal_lambda(t_prev_list[-1]), lambda_t,
                                                         lambda_list, tau)

        for i in range(order):
            if self.predict_x0:
                gradient_part += (1 + tau ** 2) * sigma_t * torch.exp(- tau ** 2 * lambda_t) * gradient_coefficients[
                    i] * model_prev_list[-(i + 1)]
            else:
                gradient_part += -(1 + tau ** 2) * alpha_t * gradient_coefficients[i] * model_prev_list[-(i + 1)]

        if self.predict_x0:
            noise_part = sigma_t * torch.sqrt(1 - torch.exp(-2 * tau ** 2 * h)) * noise
        else:
            noise_part = tau * sigma_t * torch.sqrt(torch.exp(2 * h) - 1) * noise

        if self.predict_x0:
            x_t = torch.exp(-tau ** 2 * h) * (sigma_t / sigma_prev) * x + gradient_part + noise_part
        else:
            x_t = (alpha_t / alpha_prev) * x + gradient_part + noise_part

        return x_t

    def adams_bashforth_update_few_steps(self, order, x, tau, model_prev_list, t_prev_list, noise, t):
        """
        SA-Predictor, with the "rescaling" trick in Appendix D in SA-Solver paper https://arxiv.org/pdf/2309.05019.pdf
        """

        assert order in [1, 2, 3, 4], "order of stochastic adams bashforth method is only supported for 1, 2, 3 and 4"

        # get noise schedule
        ns = self.noise_schedule
        alpha_t = ns.marginal_alpha(t)
        sigma_t = ns.marginal_std(t)
        lambda_t = ns.marginal_lambda(t)
        alpha_prev = ns.marginal_alpha(t_prev_list[-1])
        sigma_prev = ns.marginal_std(t_prev_list[-1])
        gradient_part = torch.zeros_like(x)
        h = lambda_t - ns.marginal_lambda(t_prev_list[-1])
        lambda_list = [ns.marginal_lambda(t_prev_list[-(i + 1)]) for i in range(order)]
        gradient_coefficients = self.get_coefficients_fn(order, ns.marginal_lambda(t_prev_list[-1]), lambda_t,
                                                         lambda_list, tau)

        if self.predict_x0:
            if order == 2:  ## if order = 2 we do a modification that does not influence the convergence order similar to unipc. Note: This is used only for few steps sampling.
                # The added term is O(h^3). Empirically we find it will slightly improve the image quality.
                # ODE case
                # gradient_coefficients[0] += 1.0 * torch.exp(lambda_t) * (h ** 2 / 2 - (h - 1 + torch.exp(-h))) / (ns.marginal_lambda(t_prev_list[-1]) - ns.marginal_lambda(t_prev_list[-2]))
                # gradient_coefficients[1] -= 1.0 * torch.exp(lambda_t) * (h ** 2 / 2 - (h - 1 + torch.exp(-h))) / (ns.marginal_lambda(t_prev_list[-1]) - ns.marginal_lambda(t_prev_list[-2]))
                gradient_coefficients[0] += 1.0 * torch.exp((1 + tau ** 2) * lambda_t) * (
                            h ** 2 / 2 - (h * (1 + tau ** 2) - 1 + torch.exp((1 + tau ** 2) * (-h))) / (
                                (1 + tau ** 2) ** 2)) / (ns.marginal_lambda(t_prev_list[-1]) - ns.marginal_lambda(
                    t_prev_list[-2]))
                gradient_coefficients[1] -= 1.0 * torch.exp((1 + tau ** 2) * lambda_t) * (
                            h ** 2 / 2 - (h * (1 + tau ** 2) - 1 + torch.exp((1 + tau ** 2) * (-h))) / (
                                (1 + tau ** 2) ** 2)) / (ns.marginal_lambda(t_prev_list[-1]) - ns.marginal_lambda(
                    t_prev_list[-2]))

        for i in range(order):
            if self.predict_x0:
                gradient_part += (1 + tau ** 2) * sigma_t * torch.exp(- tau ** 2 * lambda_t) * gradient_coefficients[
                    i] * model_prev_list[-(i + 1)]
            else:
                gradient_part += -(1 + tau ** 2) * alpha_t * gradient_coefficients[i] * model_prev_list[-(i + 1)]

        if self.predict_x0:
            noise_part = sigma_t * torch.sqrt(1 - torch.exp(-2 * tau ** 2 * h)) * noise
        else:
            noise_part = tau * sigma_t * torch.sqrt(torch.exp(2 * h) - 1) * noise

        if self.predict_x0:
            x_t = torch.exp(-tau ** 2 * h) * (sigma_t / sigma_prev) * x + gradient_part + noise_part
        else:
            x_t = (alpha_t / alpha_prev) * x + gradient_part + noise_part

        return x_t

    def adams_moulton_update_few_steps(self, order, x, tau, model_prev_list, t_prev_list, noise, t):
        """
        SA-Corrector, without the "rescaling" trick in Appendix D in SA-Solver paper https://arxiv.org/pdf/2309.05019.pdf
        """

        assert order in [1, 2, 3, 4], "order of stochastic adams bashforth method is only supported for 1, 2, 3 and 4"

        # get noise schedule
        ns = self.noise_schedule
        alpha_t = ns.marginal_alpha(t)
        sigma_t = ns.marginal_std(t)
        lambda_t = ns.marginal_lambda(t)
        alpha_prev = ns.marginal_alpha(t_prev_list[-1])
        sigma_prev = ns.marginal_std(t_prev_list[-1])
        gradient_part = torch.zeros_like(x)
        h = lambda_t - ns.marginal_lambda(t_prev_list[-1])
        t_list = t_prev_list + [t]
        lambda_list = [ns.marginal_lambda(t_list[-(i + 1)]) for i in range(order)]
        gradient_coefficients = self.get_coefficients_fn(order, ns.marginal_lambda(t_prev_list[-1]), lambda_t,
                                                         lambda_list, tau)

        if self.predict_x0:
            if order == 2:  ## if order = 2 we do a modification that does not influence the convergence order similar to UniPC. Note: This is used only for few steps sampling.
                # The added term is O(h^3). Empirically we find it will slightly improve the image quality.
                # ODE case
                # gradient_coefficients[0] += 1.0 * torch.exp(lambda_t) * (h / 2 - (h - 1 + torch.exp(-h)) / h)
                # gradient_coefficients[1] -= 1.0 * torch.exp(lambda_t) * (h / 2 - (h - 1 + torch.exp(-h)) / h)
                gradient_coefficients[0] += 1.0 * torch.exp((1 + tau ** 2) * lambda_t) * (
                            h / 2 - (h * (1 + tau ** 2) - 1 + torch.exp((1 + tau ** 2) * (-h))) / (
                                (1 + tau ** 2) ** 2 * h))
                gradient_coefficients[1] -= 1.0 * torch.exp((1 + tau ** 2) * lambda_t) * (
                            h / 2 - (h * (1 + tau ** 2) - 1 + torch.exp((1 + tau ** 2) * (-h))) / (
                                (1 + tau ** 2) ** 2 * h))

        for i in range(order):
            if self.predict_x0:
                gradient_part += (1 + tau ** 2) * sigma_t * torch.exp(- tau ** 2 * lambda_t) * gradient_coefficients[
                    i] * model_prev_list[-(i + 1)]
            else:
                gradient_part += -(1 + tau ** 2) * alpha_t * gradient_coefficients[i] * model_prev_list[-(i + 1)]

        if self.predict_x0:
            noise_part = sigma_t * torch.sqrt(1 - torch.exp(-2 * tau ** 2 * h)) * noise
        else:
            noise_part = tau * sigma_t * torch.sqrt(torch.exp(2 * h) - 1) * noise

        if self.predict_x0:
            x_t = torch.exp(-tau ** 2 * h) * (sigma_t / sigma_prev) * x + gradient_part + noise_part
        else:
            x_t = (alpha_t / alpha_prev) * x + gradient_part + noise_part

        return x_t
    
    def denoise_to_zero(self, order, x, tau, model_prev_list, t_prev_list, noise, t):
        x_0 = model_prev_list[-1]
        return x_0

    def sample(self, x, tau, total_timesteps, predictor_order=3, corrector_order=4, pc_mode='PEC', return_intermediate=False):
        """
        For the PC-mode, please refer to the wiki page
        https://en.wikipedia.org/wiki/Predictor%E2%80%93corrector_method#PEC_mode_and_PECE_mode
        'PEC' needs one model evaluation per step while 'PECE' needs two model evaluations
        We recommend use pc_mode='PEC' for NFEs is limited. 'PECE' mode is only for test with sufficient NFEs.
        """
        disable_first_corrector = True
        skip_final_step = True
        lower_order_final = True

        assert pc_mode in ['PEC', 'PECE'], 'Predictor-corrector mode only supports PEC and PECE'
        device = x.device
        intermediates = []
        with torch.no_grad():
            steps = total_timesteps.shape[0] - 1
            assert steps >= max(predictor_order, corrector_order - 1)
            timesteps = total_timesteps
            assert timesteps.shape[0] - 1 == steps
            # Init the initial values.
            step = 0
            t = timesteps[step]
            noise = torch.randn_like(x)
            t_prev_list = [t]

            model_prev_list = [self.model_fn(x, t)]

            if return_intermediate:
                intermediates.append(x)

            # determine the first several values
            for step in range(1, max(predictor_order, corrector_order - 1)):

                t = timesteps[step]
                predictor_order_used = min(predictor_order, step)
                corrector_order_used = min(corrector_order, step + 1)
                noise = torch.randn_like(x)
                # predictor step
                x_p = self.adams_bashforth_update_few_steps(order=predictor_order_used, x=x, tau=tau(t),
                                                            model_prev_list=model_prev_list, t_prev_list=t_prev_list,
                                                            noise=noise, t=t)
                # evaluation step
                model_x = self.model_fn(x_p, t)

                # update model_list
                model_prev_list.append(model_x)
                # corrector step
                if corrector_order > 0 and not (disable_first_corrector and step == 1):
                    x = self.adams_moulton_update_few_steps(order=corrector_order_used, x=x, tau=tau(t),
                                                            model_prev_list=model_prev_list, t_prev_list=t_prev_list,
                                                            noise=noise, t=t)
                else:
                    x = x_p

                # evaluation step if correction and mode = pece
                if corrector_order > 0 and pc_mode == 'PECE' and not (disable_first_corrector and step == 1):
                    model_x = self.model_fn(x, t)
                    del model_prev_list[-1]
                    model_prev_list.append(model_x)

                if return_intermediate:
                    intermediates.append(x)

                t_prev_list.append(t)

            for step in range(max(predictor_order, corrector_order - 1), steps + 1):
                if lower_order_final:
                    predictor_order_used = min(predictor_order, steps - step + 1)
                    corrector_order_used = min(corrector_order, steps - step + 2)

                else:
                    predictor_order_used = predictor_order
                    corrector_order_used = corrector_order
                t = timesteps[step]
                noise = torch.randn_like(x)

                # predictor step
                if skip_final_step and step == steps:
                    x_p = self.denoise_to_zero(order=predictor_order_used, x=x, tau=0,
                                                                model_prev_list=model_prev_list,
                                                                t_prev_list=t_prev_list, noise=noise, t=t)
                else:
                    x_p = self.adams_bashforth_update_few_steps(order=predictor_order_used, x=x, tau=tau(t),
                                                                model_prev_list=model_prev_list,
                                                                t_prev_list=t_prev_list, noise=noise, t=t)

                # evaluation step
                # do not evaluate if skip_final_step and step = steps
                if not skip_final_step or step < steps:
                    model_x = self.model_fn(x_p, t)

                # update model_list
                # do not update if skip_final_step and step = steps
                if not skip_final_step or step < steps:
                    model_prev_list.append(model_x)

                # corrector step
                # do not correct if skip_final_step and step = steps
                if corrector_order > 0 and (not skip_final_step or step < steps):
                    x = self.adams_moulton_update_few_steps(order=corrector_order_used, x=x, tau=tau(t),
                                                            model_prev_list=model_prev_list,
                                                            t_prev_list=t_prev_list, noise=noise, t=t)
                else:
                    x = x_p

                # evaluation step if mode = pece and step != steps
                if corrector_order > 0 and (pc_mode == 'PECE' and step < steps):
                    model_x = self.model_fn(x, t)
                    del model_prev_list[-1]
                    model_prev_list.append(model_x)

                if return_intermediate:
                    intermediates.append(x)

                t_prev_list.append(t)
                del model_prev_list[0]

        return (x, intermediates) if return_intermediate else x
