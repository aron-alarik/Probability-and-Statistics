
import numpy as np
import pandas as pd
import ray
import tensorflow as tf
import tensorflow_probability as tfp
from icecream import ic
import warnings
import bezier

from sklearn.linear_model import BayesianRidge

from statsmodels.tsa.stattools import adfuller

import matplotlib.pyplot as plt
import matplotx
plt.style.use(matplotx.styles.dracula)

import tensorflow as tf
from tensorflow_probability import distributions as tfd
import pywt
import hurst
from fracdiff.sklearn import Fracdiff, FracdiffStat
from arch import arch_model
from fracdiff import fdiff
from scipy.special import binom
from scipy.special import gamma, gammaln
from scipy.special import comb

warnings.filterwarnings("ignore")

from collections import namedtuple

DiscreteJumpFnResults = namedtuple("DiscreteJumpFnResults", ["log_accept_ratio", "accepted_results"])

class DiscreteJumpFn_(tfp.mcmc.TransitionKernel):

    def __init__(self, target_log_prob_fn, step_size, num_macro_regimes, num_micro_regimes):
        self._target_log_prob_fn = target_log_prob_fn
        self._step_size = step_size
        self._num_macro_regimes = num_macro_regimes
        self._num_micro_regimes = num_micro_regimes

    def one_step(self, current_state, previous_kernel_results):
        # generate a proposed state within the range of possible states
        proposed_macro_state = tf.random.uniform(shape=current_state[0].shape, minval=0, maxval=self._num_macro_regimes, dtype=tf.int32)
        proposed_micro_state = tf.random.uniform(shape=current_state[1].shape, minval=0, maxval=self._num_micro_regimes, dtype=tf.int32)
        proposed_state = (proposed_macro_state, proposed_micro_state)

        log_accept_ratio = (self._target_log_prob_fn(*proposed_state) -
                            self._target_log_prob_fn(*current_state))

        do_accept = tf.less_equal(
            tf.random.uniform(shape=log_accept_ratio.shape, minval=0., maxval=1., dtype=tf.float32),
            tf.exp(log_accept_ratio))

        next_state = tf.nest.map_structure(lambda cs, ps, da: tf.where(da, ps, cs),
                                           current_state, proposed_state, do_accept)

        return next_state, DiscreteJumpFnResults(log_accept_ratio=log_accept_ratio, accepted_results=do_accept)

    def bootstrap_results(self, current_state):
        return DiscreteJumpFnResults(log_accept_ratio=tf.zeros_like(current_state), accepted_results=tf.zeros_like(current_state))

    def is_calibrated(self):
        return True

class DiscreteJumpFn(tfp.mcmc.TransitionKernel):

    def __init__(self, target_log_prob_fn, step_size, num_states):
        self._target_log_prob_fn = target_log_prob_fn
        self._step_size = step_size
        self._num_states = num_states

    def one_step(self, current_state, previous_kernel_results, seed=None):

        current_state = tf.convert_to_tensor(current_state, dtype=tf.int32)
        proposed_state = tf.random.uniform(shape=current_state.shape, minval=0, maxval=self._num_states, dtype=tf.int32)
        print("Proposed state: ", proposed_state)

        log_accept_ratio = tf.cast(self._target_log_prob_fn(proposed_state) - self._target_log_prob_fn(current_state), tf.float32)
        log_u = tf.math.log(tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32))
        do_accept = tf.less_equal(log_u, log_accept_ratio)
        next_state = tf.where(do_accept, proposed_state, current_state)

        print("Next state: ", next_state)

        # Update the previous_kernel_results
        new_kernel_results = previous_kernel_results._replace(
            log_accept_ratio=tf.reshape(log_accept_ratio, (1, -1)),
            accepted_results=tf.reshape(tf.cast(do_accept, tf.float32), (1, -1))
        )

        print("Previous kernel results: ", new_kernel_results)

        return next_state, new_kernel_results




    def bootstrap_results(self, current_state):
        return DiscreteJumpFnResults(log_accept_ratio=tf.zeros_like(current_state), accepted_results=tf.zeros_like(current_state))

    def is_calibrated(self):
        return True

    
class HestonModel:

    def __init__(self, mu, kappa, theta, sigma, rho, S0, v0, dt, n_steps):
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.S0 = S0
        self.v0 = v0
        self.dt = dt
        self.n_steps = n_steps

    def simulate(self):

        # Initialize S and v
        S = tf.TensorArray(tf.float32, size=self.n_steps + 1)
        v = tf.TensorArray(tf.float32, size=self.n_steps + 1)
        S = S.write(0, self.S0)
        v = v.write(0, self.v0)

        # Generate standard normal random variables
        Z1 = tf.random.normal(self.n_steps)
        Z2 = tf.random.normal(self.n_steps)

        # Generate correlated Wiener processes
        W1 = tf.math.sqrt(self.dt) * Z1
        W2 = tf.math.sqrt(self.dt) * (self.rho * Z1 + tf.math.sqrt(1 - self.rho**2) * Z2)

        for t in range(self.n_steps):
            # Update S and v using the Heston model equations
            S[t+1] = S[t] + self.mu * S[t] * self.dt + tf.math.sqrt(v[t]) * S[t] * W1[t]
            v[t+1] = v[t] + self.kappa * (self.theta - v[t]) * self.dt + self.sigma * tf.math.sqrt(v[t]) * W2[t]

        return S, v

class AsymmetricLaplace(tfp.distributions.Distribution):
    """Asymmetric Laplace distribution.

    Args:
        loc: The location parameter.
        diversity_left: The diversity parameter on the left side of the location parameter.
        diversity_right: The diversity parameter on the right side of the location parameter.
        name: The name of the distribution.

    """

    def __init__(self, loc, diversity_left, diversity_right, name='AsymmetricLaplace'):
        parameters = dict(locals())
        super().__init__(dtype=tf.float32, reparameterization_type=tfp.distributions.FULLY_REPARAMETERIZED, 
                         validate_args=False, allow_nan_stats=True, parameters=parameters, name=name)

        self.loc = loc
        self.diversity_left = diversity_left
        self.diversity_right = diversity_right

    def _log_prob(self, x):
        """Log probability density function.

        Args:
            x: The value to evaluate the log probability density function at.

        Returns:
            The log probability density function evaluated at x.

        """
        return tf.where(x < self.loc,
                        -tf.abs(x - self.loc) * self.diversity_left - tf.math.log(1 / self.diversity_left),
                        -tf.abs(x - self.loc) * self.diversity_right - tf.math.log(1 / self.diversity_right))

    def _sample_n(self, n, seed=None):

        """Sample n values from the distribution."""

        scale=tf.math.abs(1.0/self.diversity_left - 1.0/self.diversity_right)
        ic (scale.numpy())

        standard_laplace_samples = tfp.distributions.Laplace(loc=0, scale=scale.numpy()).sample(n, seed=seed)

        samples = tf.where(standard_laplace_samples < 0,
                           self.loc + (1 / self.diversity_left) * standard_laplace_samples,
                           self.loc + (1 / self.diversity_right) * standard_laplace_samples)
        return samples

    def _batch_shape_tensor(self):
        return tf.broadcast_dynamic_shape(tf.shape(self.loc), tf.shape(self.diversity_left))

    def _batch_shape(self):
        return tf.broadcast_static_shape(self.loc.shape, self.diversity_left.shape)

    def _event_shape_tensor(self):
        return tf.constant([], dtype=tf.int32)

    def _event_shape(self):
        return tf.TensorShape([])

class Lucy:

    def __init__(self, n_macro_states, n_micro_states, n_binary_components, 
                 df, window_size, list_covariates, returns= 'barReturn', 
                 price='typical_price', ma_short='kama_ma_short', series_size= 5000):
        """
        Initialize Lucy with the number of macro states, micro states, and binary components.
        """
        self.n_macro_states = n_macro_states
        self.n_micro_states = n_micro_states
        self.n_binary_components = n_binary_components
        self.df = df
        self.window_size = window_size
        self.optimizer = tf.keras.optimizers.Adam()
        self.model = None
        self.series_size = series_size
        self.theta = tf.Variable(initial_value=tf.ones([self.n_binary_components]), name='theta')
        self.scaling_stddev = tf.Variable(1.0, dtype=tf.float32)
        self.list_covariates = list_covariates
        self.df = self.df[:self.series_size]
        self.series = self.df['kama_ma_short'].values
        self.price = self.df['typical_price'].values
        self.returns = self.df['barReturn'].values
        self.n_time_steps = len(self.df)
        self.counter = 0
        self.init_log_prob = -865776200000.0
        self.percent_change = []
        
    def garchVolModel_ext(self, series):

        #  NaN or inf values to 0
        series = np.nan_to_num(series)
        
        am = arch_model(series, x=None,  #power=1.0,
                                mean='HARX', lags= 1, 
                                vol='GARCH', p=1, o=0, q=1, 
                                dist='studentst', hold_back=None, rescale=True)

        volatility_model = am.fit(update_freq=5, disp=False)

        #print ('ext_parameters Garch model: **********\n', volatility_model.params, '\n**********')
        garch_vol = volatility_model.conditional_volatility

        # set nan values to 0
        garch_vol = np.nan_to_num(garch_vol)

        return volatility_model, garch_vol

    def get_covariates(self):
        
        covariates = []

        for covariate in self.list_covariates:
            series = self.df[covariate].values
            covariates.append(series)

        price_ma_diff = self.price - self.series
        covariates.append(price_ma_diff)

        return covariates

    def heston_vol_model(self, series):
        # calculate heston volatility process
        series_size = self.series_size

        returns = self.returns[:series_size]
        price = self.price[:series_size]

        volatility_model_price, garch_vol_price = self.garchVolModel_ext(price)

        # get correlation between returns and garch_vol
        rho = np.around(np.corrcoef(returns, garch_vol_price)[0, 1], 4)
        mu = np.around(np.mean(returns), 4)
        theta = np.around(np.mean(garch_vol_price), 4)
        omega = np.around(volatility_model_price.params['omega'], 4)
        alpha = np.around(volatility_model_price.params['alpha[1]'], 4)
        beta = np.around(volatility_model_price.params['beta[1]'], 4)
        kappa = 2.0
        dt = 0.01
        n_steps = 1000

        S0 = series[-1]
        v0 = omega
        sigma = np.around(np.sqrt(alpha + beta * omega), 4)

        ic (rho, mu, theta, omega, alpha, beta, sigma)

    def estimate_data_coeff(self, time_series, covariates: list ):

        '''time_series_tf = tf.convert_to_tensor(time_series, dtype=tf.float32)

        nsamples = 50

        trend_model = tfp.sts.LocalLinearTrend( observed_time_series=time_series_tf)

        seasonal_model = tfp.sts.Seasonal( num_seasons=4, observed_time_series=time_series_tf)

        model = tfp.sts.Sum([trend_model, seasonal_model], observed_time_series=time_series_tf)

        variational_posteriors = tfp.sts.build_factored_surrogate_posterior(model=model)

        # Allow external control of optimization to reduce test runtimes.
        num_variational_steps = 200 
        num_variational_steps = int(num_variational_steps)

        # Optimize the model
        optimizer = tf.optimizers.Adam(learning_rate=0.1)

        # Using negative log-likelihood as loss function
        # Using negative log-likelihood as loss function
        def nll():

            samples = variational_posteriors.sample()
            return -variational_posteriors.log_prob(samples)

        
        # Optimize
        losses = tfp.math.minimize(nll, num_steps=num_variational_steps, optimizer=optimizer, )

        # Sample from the variational posteriors
        samples = variational_posteriors.sample(50)

        # Compute the mean of the samples
        mean_samples = {k: np.mean(v, axis=0) for k, v in samples.items()}

        # Print the mean values of the parameters
        for param in mean_samples.keys():
            print(f'{param} mean: {mean_samples[param]}')'''

        reshape_covariates = []
        # process the covariates
        for covariate in covariates:
            
            # reshape the covariates
            #covariate = tf.convert_to_tensor(covariate, dtype=tf.float32)
            reshape_covariates.append(covariate.reshape(-1, 1))
            
        covariates = np.hstack(reshape_covariates)

        # Create and fit the model
        model = BayesianRidge()
        
        model.fit(covariates, time_series)

        # round off each value in list(model.coef_) to 4 decimal places
        covariates_list = [np.around(x, 4) for x in list(model.coef_)]

        return covariates_list

    def estimate_hurst_exponent_wav(self, time_series):

        hurst_wav_list = []
        count = 100
        typPrice = time_series

        for wavelet_sub in pywt.wavelist(family=None, kind='discrete'):

            wavelet = wavelet_sub 
            mode = 'cpd'
            level = 3
            DWT_coeffs = pywt.wavedec(time_series, wavelet, mode, level)

            if len(DWT_coeffs) < 2:
                raise ValueError("Insufficient number of wavelet coefficients for Hurst exponent estimation.")

            DWT_std = np.array([np.std(coeff) for coeff in DWT_coeffs])
            dH = np.diff(np.log10(DWT_std))

            if len(dH) < 1:
                raise ValueError("Insufficient number of logarithmic differences for Hurst exponent estimation.")

            Hurst_wav = -dH[0] / np.log10(2.0)
            Hurst_wav = np.around(Hurst_wav/10, 4)

            hurst_wav_list.append(Hurst_wav)
        
        hurst_wav = np.around(np.mean(hurst_wav_list), 4)
        
        # estimate Hurst exponent using polyfit
        maxLag = int(count - (count*0.2))
        lags = range(2, maxLag)
        
        tau = [np.std(np.subtract(typPrice[lag:], typPrice[:-lag])) for lag in lags]
        tau = np.around(tau, 3)
        
        hurstExponent = np.polyfit(np.log(lags), np.log(tau), 1)[0]
        hurstExponent = np.around(hurstExponent, 4)

        # estimate Hurst exponent using hurst package
        hurst_lib_exp = []
        kind_rv= ['random_walk', 'change', 'price']

        for kind in kind_rv:

            try:

                H, c, data = hurst.compute_Hc(time_series, kind=kind, simplified=True)
                hurst_lib_exp.append(np.around(H, 4))

            except:
                pass

        # get mean of all estimates
        hurst_mean = np.around(np.mean([hurst_wav, hurstExponent] + hurst_lib_exp), 4)

        return [hurst_mean, hurst_wav, hurstExponent] + hurst_lib_exp
    
    def stochastic_volatility_model(self, series):

        # estimate the stochastic volatility using GARCH model 
        try:
            volatility_model, garch_vol = self.garchVolModel_ext(series)
        
        except:
            volatility_model, garch_vol = self.garchVolModel_ext(series[10:])
            # pad the beginning of series with 10 values
            garch_vol = np.pad(garch_vol, (10, 0), 'edge')

        finally:
            volatility_model, garch_vol = self.garchVolModel_ext(series[:-10])
            # pad the end of series with 10 values
            garch_vol = np.pad(garch_vol, (0, 10), 'edge')
        
        # simulate the garch volatility process

        T = len(series)

        # The volatility process
        volatility_process = np.zeros(T)

        omega = np.around(volatility_model.params['omega'], 4)
        alpha = np.around(volatility_model.params['alpha[1]'], 4)
        beta = np.around(volatility_model.params['beta[1]'], 4)

        # Calculate the volatility process
        for t in range(1, T):
            volatility_process[t] = np.sqrt(omega + alpha * series[t-1]**2 + beta * volatility_process[t-1]**2)

        volatility_process = volatility_process[10:]

        # pad the beginning of series with 10 edge values
        volatility_process = np.pad(volatility_process, (10, 0), 'edge')
        price = self.df['typical_price'].values[:5000]

        return volatility_process

    def fractional_differencing(self, series, hurst_exp):
        """
        Implements fractional differencing on a time series using the Hurst exponent.

        Args:
            series: Time series data
            hurst: The Hurst exponent

        Returns:
            The fractionally differenced series
        """
        
        # The differencing parameter, calculated based on the Hurst exponent
        d = np.around(2 * hurst_exp - 1, 3)

        # The length of the series
        T = len(series)

        # The weights for the fractionally differenced series
        weights = np.array([(-1)**i * binom(d, i) for i in range(T)])
        # The weights for the fractionally differenced series
        #weights = np.array([(-1)**i * comb(d, i) for i in range(T)])
        # The weights for the fractionally differenced series
        #weights = np.array([(-1)**i * binomial(d, i) for i in range(T)])
        # The weights for the fractionally differenced series
        # The weights for the fractionally differenced series

        # Reverse the weights
        weights = weights[::-1]

        # The fractionally differenced series
        fd_series_1 = np.zeros(T)
        
        
        # Calculate the fractionally differenced series
        for t in range(T):
            fd_series_1[t] = np.dot(weights[:t+1][::-1], series[:t+1])

        # differencing with d
        fd = Fracdiff(d=d)
        fd_series_2 = fd.fit(series.reshape(1, -1))
        fd_series_2 = fd_series_2.transform(series.reshape(1, -1))

        # differencing with auto calculated d
        f = FracdiffStat()
        fd_series_3 = f.fit_transform(series.reshape(-1, 1))
        #ic(hurst_exp, d, f.d_)

        # differencing using fdiff function with d
        fd_series_4  = fdiff(series, d)
        
        fd_series_2 = fd_series_2.reshape(-1)
        fd_series_3 = fd_series_3.reshape(-1)
        fd_series_4 = fd_series_4.reshape(-1)
        
        # clean the data
        fd_series_1 = fd_series_1[:-10]
        fd_series_3 = fd_series_3[10:]
        fd_series_4 = fd_series_4[10:]
        
        # pad the ending of series with 10 edge values
        fd_series_1 = np.pad(fd_series_1, (0, 10), 'edge')
        fd_series_3 = np.pad(fd_series_3, (10, 0), 'edge')
        fd_series_4 = np.pad(fd_series_4, (10, 0), 'edge')

        # get mean series  of fd_series_1, fd_series_2, fd_series_3, fd_series_4
        fd_series_mean = np.mean([fd_series_1, fd_series_2, fd_series_3, fd_series_4], axis=0)

        return fd_series_mean
    
    def adjust_for_covariates(self, base_scaling_process, series, covariates):
        """
        Adjusts the base scaling process based on the RSI and MACD covariates.

        Args:
            base_scaling_process: The base scaling process
            rsi: The Relative Strength Index
            macd: The Moving Average Convergence Divergence

        Returns:
            The adjusted base scaling process
        """
        # Define the coefficients for RSI and MACD
        # In reality, you would learn these coefficients from your data.

        covariates_est = self.estimate_data_coeff(series, covariates)

        sum_covariates = 0

        for indicator, coeff in zip(covariates, covariates_est):

            sum_covariates += coeff * indicator
        
        # The adjusted base scaling process
        adjusted_base_scaling_process = base_scaling_process + sum_covariates

        return adjusted_base_scaling_process

    def sliding_windows(self, data, window_size = None):

        if window_size is None:
            window_size = self.window_size
        
        # Calculate the number of sliding windows
        num_windows = data.shape[0] - window_size + 1
        # Create a view into the data array with the shape of the sliding windows
        shape = (num_windows, window_size) + data.shape[1:]
        strides = (data.strides[0],) + data.strides
        windows = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

        return windows

    def define_transition_matrices(self):

        """
        Define the transition probability matrices for macro and micro states.
        Initially, all transition probabilities are equal (uniform distribution).
        """
        # Define the transition probability matrix for macro states
        macro_transition_matrix = np.ones((self.n_macro_states, self.n_macro_states)) / self.n_macro_states

        # Define the transition probability matrix for micro states
        # Here, we assume that each macro state has the same number of micro states
        micro_transition_matrix = np.ones((self.n_macro_states, self.n_micro_states, self.n_micro_states)) / self.n_micro_states

        return macro_transition_matrix, micro_transition_matrix

    def define_duration_distributions(self, use_dist="weibull"):

        """
        Define the duration distributions for each macro and micro state.
        In this example, we use Weibull distributions with shape and scale parameters set to 1.
        """

        if use_dist == "weibull":
            
            # Define the duration distributions for macro states
            macro_state_duration_distribution = [
                tfp.distributions.Weibull(concentration=1.0, scale=1.0) for _ in range(self.n_macro_states)
            ]

            # Define the duration distributions for micro states
            # Here, we assume that each macro state has the same number of micro states
            micro_state_duration_distribution = [
                [tfp.distributions.Weibull(concentration=1.0, scale=1.0) for _ in range(self.n_micro_states)]
                for _ in range(self.n_macro_states)
            ]

            return macro_state_duration_distribution, micro_state_duration_distribution

        elif use_dist == "negative_binomial":
        
            """
            Define the duration distributions for each macro and micro state.
            In this example, we use Negative Binomial distributions.
            """
            # Define the duration distributions for macro states
            macro_state_duration_distribution = [
                tfp.distributions.NegativeBinomial(total_count=5.0, logits=1.0) for _ in range(self.n_macro_states)
            ]

            # Define the duration distributions for micro states
            # Here, we assume that each macro state has the same number of micro states
            micro_state_duration_distribution = [
                [tfp.distributions.NegativeBinomial(total_count=5.0, logits=1.0) for _ in range(self.n_micro_states)]
                for _ in range(self.n_macro_states)
            ]
        
            return macro_state_duration_distribution, micro_state_duration_distribution
        
        else:
            raise ValueError("use_dist must be either weibull or negative_binomial")

    def define_observation_distributions(self, diversity_parameters):
        """
        This method defines the observation distributions for each micro state.
        These distributions capture the conditional distribution of the time series given the current micro state.
        The diversity_parameters should be a 2D tensor with shape (number of micro states, 2).
        Each row of the diversity_parameters tensor corresponds to the left and right diversity parameters in the corresponding micro state.
        """
        self.observation_distributions = []
        for i in range(self.n_micro_states):
            diversity_param_left, diversity_param_right = diversity_parameters[i]
            # Define an asymmetric distribution for the i-th micro state
            dist = AsymmetricLaplace(loc=0., diversity_left=diversity_param_left, diversity_right=diversity_param_right)
            self.observation_distributions.append(dist)

    def define_initial_state_distribution(self):
        
        """
        Define the initial state distribution for macro and micro states.
        Initially, all states are equally likely.
        """
        # Define the initial state distribution for macro states
        self.initial_macro_state_distribution = np.ones(self.n_macro_states) / self.n_macro_states

        # Define the initial state distribution for micro states
        # Here, we assume that each macro state has the same number of micro states
        self.initial_micro_state_distribution = np.ones((self.n_macro_states, self.n_micro_states)) / self.n_micro_states

        return self.initial_macro_state_distribution, self.initial_micro_state_distribution

    def msm_scaling_process(self, binary_components, theta, hurst_exp, covariates=None):

        # supress convergence warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        # Calculate the base scaling process using the binary components and theta
        base_scaling_process = tf.reduce_sum(binary_components * theta, axis=-1)
        base_scaling_process = base_scaling_process.numpy()
        T, n_micro_states= base_scaling_process.shape
        
        # get values of each column in the base scaling process
        scaling_process_list = []

        for i in range(n_micro_states):
            scaling_process_list.append(list(base_scaling_process[:, i].reshape(-1)))
            
        scaling_process_list_fdiff = []
        
        # for each colum in the base scaling process, apply fractional differencing
        # If the Hurst exponent has been provided, apply fractional differencing
        if hurst_exp is not None:
            for scaling_process in scaling_process_list:

                scaling_process_t = np.array(scaling_process)
                scaling_process_t = self.fractional_differencing(scaling_process_t, hurst_exp)
                scaling_process_list_fdiff.append(scaling_process_t)

        # If covariates are provided, adjust the scaling process based on the covariates
        scaling_process_cov = []
        time_series = self.price
        for scaling_process in scaling_process_list_fdiff:

            if covariates is not None:

                scaling_process_q = self.adjust_for_covariates(scaling_process, time_series, covariates)
                scaling_process_cov.append(scaling_process_q)
                
        # Now implement a stochastic volatility model. 
        # This could be done in a variety of ways, depending on the specific model you want to use.
        # Here's a simple example where the volatility is a stochastic process that depends on the base scaling process:
        volatility_processes = []
        for scaling_process in scaling_process_cov:

            volatility_process = self.stochastic_volatility_model(scaling_process)
            volatility_processes.append(volatility_process)            
        

        return volatility_processes

    def compute_state_durations(self, state_sequence):

        durations = []
        current_state = state_sequence[0]
        current_duration = 1

        for state in state_sequence[1:]:
            if state == current_state:
                # If the state remains the same, increment the duration
                current_duration += 1
            else:
                # If the state changes, record the duration of the previous state
                durations.append((current_state, current_duration))
                # And reset the current state and duration
                current_state = state
                current_duration = 1

        # Don't forget to add the last state and its duration
        durations.append((current_state, current_duration))

        return durations

    def msm_binary_components(self, transition_matrix, initial_state_distribution, 
                              num_steps):
        """
        Generate the binary components for the MSM.
        """
        markov_chain = tfp.distributions.MarkovChain(
            initial_distribution=initial_state_distribution,
            transition_distribution=transition_matrix,
            num_steps=num_steps
        )
        return markov_chain.sample()

    def log_prob_scaling(self, params):

        """
        Compute the log probability of the scaling process given the binary components and states.
        """

        binary_components = params["binary_components"]
        theta = params["theta"]
        covariates= params["covariates"]
        hurst_param= params["hurst_exp"]
        core_series = params["core_series"]

        log_prob_list = []
        
        # Get the MSM scaling process from the binary components and states, 
        # considering the stochastic volatility, covariates, and the Hurst exponent
        scaling_processes = self.msm_scaling_process(binary_components, theta, hurst_param, 
                                                     covariates=covariates)

        # Define the distribution for the scaling process
        # Assuming the scaling process follows a Normal distribution
        for scaling_process in scaling_processes:

            scaling_process = scaling_process.astype(np.float32)
            
            scaling_distribution = tfp.distributions.Normal(loc=scaling_process, scale=self.scaling_stddev)

            # The log_prob method of a TensorFlow probability distribution gives the log probability of a sample under the distribution
            log_prob = scaling_distribution.log_prob(core_series)
            log_prob_list.append(np.sum(log_prob.numpy()))

        return np.sum(log_prob_list)

    def log_prob_states_del(self, macro_state_sequence, micro_state_sequence):

        """
        Compute the log probability of the state sequence.
        """
        # Initialize log_prob with the log probability of the initial state
        log_prob_macro = np.log(self.initial_macro_state_distribution[macro_state_sequence[0]])
        log_prob_micro = np.log(self.initial_micro_state_distribution[macro_state_sequence[0], micro_state_sequence[0]])

        # Loop over the state sequence
        for t in range(1, len(macro_state_sequence)):
            # Get the current and previous states
            prev_macro_state = macro_state_sequence[t-1]
            current_macro_state = macro_state_sequence[t]

            prev_micro_state = micro_state_sequence[t-1]
            current_micro_state = micro_state_sequence[t]

            # Add the log probability of the transition from prev_state to current_state
            log_prob_macro += np.log(self.macro_transition_matrix[prev_macro_state, current_macro_state])
            log_prob_micro += np.log(self.micro_transition_matrix[prev_macro_state, prev_micro_state, current_micro_state])

        return log_prob_macro + log_prob_micro

    def log_prob_states(self, params):

        """
        Compute the log probability of the state sequence.
        """

        macro_state_sequences = params['macro_state_sequence']
        micro_state_sequences = params['micro_state_sequence']

        micro_state_distribution = params['micro_state_distribution'] 
        macro_state_distribution = params['macro_state_distribution']

        micro_transition_matrix = params['micro_transition_matrix'] 
        macro_transition_matrix = params['macro_transition_matrix']

        # Initialize log_prob with the log probability of the initial states

        log_prob_macro = np.log(macro_state_distribution[macro_state_sequences[0]])

        log_prob_micro = 0
        for i, micro_state_sequence in enumerate(micro_state_sequences):
            log_prob_micro += np.log(micro_state_distribution[i, micro_state_sequence[0]])

        # Loop over the state sequence
        for t in range(1, len(macro_state_sequences)):
            # Get the current and previous states
            prev_macro_state = macro_state_sequences[t-1]
            current_macro_state = macro_state_sequences[t]

            log_prob_macro += np.log(macro_transition_matrix[prev_macro_state, current_macro_state])

            # Add the log probabilities of the transitions for each micro state sequence
            for i, micro_state_sequence in enumerate(micro_state_sequences):
                prev_micro_state = micro_state_sequence[t-1]
                current_micro_state = micro_state_sequence[t]

                log_prob_micro += np.log(micro_transition_matrix[prev_macro_state, prev_micro_state, current_micro_state])

        return log_prob_macro + log_prob_micro

    def log_prob_data_given_states_del(self, data, state_sequence):

        """
        Compute the log probability of the data given the state sequence.
        """

        # Initialize log_prob with 0.0
        log_prob = 0.0

        # Loop over the time series
        for t in range(len(data)):

            # Get the current state
            state = state_sequence[t]

            # Get the observed data at time t
            observed_data_t = data[t]

            # Calculate the log probability of the observed data under the distribution associated with the current state
            log_prob_state = self.observation_distributions[state].log_prob(observed_data_t)

            # Add to the total log_prob
            log_prob += log_prob_state

            # If we are using duration distributions, we also need to consider the duration dependence
            if self.duration_distributions is not None:
                # Get the current duration
                duration = ... # this should be calculated based on the state sequence

                # Add the log probability of the duration under the distribution associated with the current state
                log_prob_duration = self.duration_distributions[state].log_prob(duration)
                log_prob += log_prob_duration

            # If we are using a scaling process (e.g., MSM), we also need to consider it
            if self.msm_scaling_process is not None:
                # Get the binary components and theta associated with the current state
                binary_components = ... # this should be defined based on your implementation
                theta = ... # this should be defined based on your implementation

                # Compute the MSM scaling process
                scaling_process = self.msm_scaling_process(binary_components, theta, hurst_exp, covariates)

                # Add the log probability of the observed data under the scaling process
                log_prob_scaling = ... # this should be calculated based on the scaling process and the observed data
                log_prob += log_prob_scaling

        return log_prob

    def log_prob_data_given_states(self, params):
        
        """
        Compute the log probability of the data given the states and durations.
        """

        #log_prob_states = self.log_prob_states(macro_state_sequence, micro_state_sequence)
        log_prob_states = self.log_prob_states(params)

        # get macro and micro state durations
        micro_state_sequence = params['micro_state_sequence']
        macro_state_sequence = params['macro_state_sequence']

        macro_duration_distributions = params['macro_duration_distributions']
        micro_duration_distributions = params['micro_duration_distributions']

        initial_macro_durations = self.compute_state_durations(macro_state_sequence)

        initial_micro_state_durations = []

        for sequence in micro_state_sequence:
            durations = self.compute_state_durations(sequence)
            initial_micro_state_durations.append(durations)
        
        log_prob_durations_macro = tf.reduce_sum([
                                    tf.reduce_sum(macro_duration_distributions[i].log_prob(initial_macro_durations[i]))
                                    for i in range(self.n_macro_states)
                                        ])
        
        log_prob_durations_micro = tf.reduce_sum([
                                    tf.reduce_sum([
                                        micro_duration_distributions[i][j].log_prob(initial_micro_state_durations[i][j])
                                        for j in range(self.n_micro_states)
                                    ])
                                    for i in range(self.n_macro_states)
                                ])

        #log_prob_scaling = self.log_prob_scaling(states, data)

        log_prob_scaling = self.log_prob_scaling(params)

        percent_change = np.around(((log_prob_scaling - self.init_log_prob)/self.init_log_prob)*100, 2)
        self.percent_change.append(percent_change)
        ic (percent_change)
        
        log_prob = log_prob_states + log_prob_durations_macro.numpy() + log_prob_durations_micro.numpy() + log_prob_scaling

        return log_prob

    def joint_log_prob(self, macro_state_sequence):

        """
        Define the joint log probability function of the observed data, the states, 
        the duration distributions, and the MSM scaling process.
        """
        self.counter += 1

        macro_state_sequence = tf.squeeze(macro_state_sequence)
        macro_state_sequence = tf.cast(macro_state_sequence, tf.int32)

        params = {

                    "macro_state_sequence": macro_state_sequence,
                    "micro_state_sequence": self.micro_state_sequence,
                    
                    "macro_duration_distributions": self.macro_duration_distributions,
                    "micro_duration_distributions" : self.micro_duration_distributions,
                    
                    "macro_state_distribution" : self.macro_state_distribution,
                    "micro_state_distribution" : self.micro_state_distribution,
                    
                    "macro_transition_matrix" : self.macro_transition_matrix,
                    "micro_transition_matrix" : self.micro_transition_matrix,
                    
                    "binary_components" : self.binary_components,
                    "theta" : self.theta,
                    "covariates" : self.covariates_var,
                    "hurst_exp" : self.hurst_param,
                    "core_series" : self.price

                }
        
        if self.hurst_param  < 0.001 or self.hurst_param  > 1:
            
            log_prob_est = -np.inf #10.0e20
            # cast as tf variable
            log_prob_est = tf.Variable(log_prob_est, dtype=tf.float64)
            #ic (hurst_param)
        
        else:
            
            log_prob_est = self.log_prob_data_given_states(params)
        
        ic(self.counter)
        ic(log_prob_est)

        return log_prob_est

    def initialize_parameters(self):

        """
        Initialize the model parameters.
        """

        # Initialize macro and micro states
        initial_macro_state = np.random.choice(self.n_macro_states)
        initial_micro_state = np.random.choice(self.n_micro_states)
        
        # Initialize transition matrices with uniform probabilities
        initial_macro_transitions = np.ones((self.n_macro_states, self.n_macro_states)) / self.n_macro_states
        initial_micro_transitions = np.ones((self.n_macro_states, self.n_micro_states, self.n_micro_states)) / self.n_micro_states
        
        # Initialize durations
        initial_macro_durations = np.ones(self.n_macro_states)
        initial_micro_durations = np.ones((self.n_macro_states, self.n_micro_states))

        # Initialize the state sequences
        initial_macro_state_sequence = np.random.choice(self.n_macro_states, self.n_time_steps)
        initial_micro_state_sequence = np.random.choice(self.n_micro_states, (self.n_macro_states, self.n_time_steps))

        # Initialize binary components and theta
        initial_theta = tf.Variable(initial_value=tf.ones([self.n_binary_components]), name='theta')
        
        initial_binary_components = np.random.randint(0, 2, (self.n_time_steps, self.n_micro_states, self.n_binary_components))

        # Initialize Hurst exponent and covariates
        initial_hurst = self.estimate_hurst_exponent_wav(self.price)
        initial_hurst = np.around(np.mean(initial_hurst), 2)
        initial_covariates = self.get_covariates()
        
        macro_state_distribution, micro_state_distribution = self.define_initial_state_distribution()
        macro_duration_distributions, micro_duration_distributions = self.define_duration_distributions()
        macro_transition_matrix, micro_transition_matrix = self.define_transition_matrices()

        initial_macro_state_sequence = tf.convert_to_tensor(initial_macro_state_sequence, dtype=tf.float32)

        init_params = {

                    'macro_state_sequence': initial_macro_state_sequence, 
                    'micro_state_sequence': initial_micro_state_sequence,
                    
                    'macro_duration_distributions':macro_duration_distributions, 
                    'micro_duration_distributions':micro_duration_distributions, 
                    
                    'macro_state_distributions':macro_state_distribution,
                    'micro_state_distributions':micro_state_distribution,
                    
                    'macro_transition_matrix':macro_transition_matrix, 
                    'micro_transition_matrix':micro_transition_matrix, 
                    
                    "binary_components": initial_binary_components, 
                    "theta": initial_theta, 
                    "covariates":initial_covariates,
                    "hurst_exp":initial_hurst,
                    "core_series":self.price,

                }

        old_params =    {  
                            "macro_transitions": initial_macro_transitions,
                            "micro_transitions": initial_micro_transitions,
                            "macro_durations": initial_macro_durations,
                            "micro_durations": initial_micro_durations,
                            "initial_macro_state": initial_macro_state,
                            "initial_micro_state": initial_micro_state,
                            "macro_state_sequence": initial_macro_state_sequence,
                            "micro_state_sequence": initial_micro_state_sequence,
                            "binary_components": initial_binary_components,
                            "theta": initial_theta,
                            "hurst_exp": initial_hurst,
                            "covariates": initial_covariates,
                            "macro_duration_distributions": macro_duration_distributions,
                            "micro_duration_distributions": micro_duration_distributions,
                            #"state_distribution": initial_state_distribution,
                            "macro_transition_matrix": macro_transition_matrix,
                            "micro_transition_matrix": micro_transition_matrix,
                            "core_series": self.price,
                        }

        return init_params

    def run_mcmc_chain(self, observed_data, num_results=1000, num_burnin_steps=500):

        """
        Run the MCMC chain to estimate the model parameters.
        """

        init_params = self.initialize_parameters()

        # unpack the initial parameters

        macro_state_sequence = init_params["macro_state_sequence"]
        micro_state_sequence = init_params["micro_state_sequence"]
        micro_state_distribution = init_params["micro_duration_distributions"]
        macro_state_distribution = init_params["macro_duration_distributions"]
        micro_transition_matrix = init_params["micro_transition_matrix"]
        macro_transition_matrix = init_params["macro_transition_matrix"]
        binary_components = init_params["binary_components"]
        theta = init_params["theta"]
        covariates = init_params["covariates"]
        hurst_param = init_params["hurst_exp"]
        core_series = init_params["core_series"]

        prob_est = self.joint_log_prob(macro_state_sequence, micro_state_sequence, 
                                        micro_state_distribution, macro_state_distribution, 
                                        micro_transition_matrix, macro_transition_matrix, 
                                        binary_components, theta, covariates, 
                                        hurst_param, core_series)
        
        ic (prob_est)

        # Set the initial state of the chain
        initial_macro_state, initial_micro_state, macro_transitions, \
        micro_transitions, macro_durations, micro_durations, binary_components, \
        theta, hurst, covariates = self.initialize_parameters()

        # Wrap the joint_log_prob function in a lambda so we can use it in tfp.mcmc.sample_chain
        unnormalized_posterior_log_prob = lambda *args: self.joint_log_prob(observed_data, *args)

        # Define the transition kernel
        kernel = tfp.mcmc.RandomWalkMetropolis(
            target_log_prob_fn=unnormalized_posterior_log_prob
        )

        # Define the transition kernel
        kernel_2 = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=unnormalized_posterior_log_prob,
            num_leapfrog_steps=2,
            step_size=0.01
        )

        # Run the MCMC chain
        [initial_macro_state_samples, initial_micro_state_samples, macro_transitions_samples, \
        micro_transitions_samples, macro_durations_samples, micro_durations_samples, \
        binary_components_samples, theta_samples, hurst_samples, covariates_samples], \
        kernel_results = tfp.mcmc.sample_chain(
                                                num_results=num_results,
                                                num_burnin_steps=num_burnin_steps,
                                                current_state=[
                                                    initial_macro_state, 
                                                    initial_micro_state, 
                                                    macro_transitions, 
                                                    micro_transitions, 
                                                    macro_durations, 
                                                    micro_durations, 
                                                    binary_components, 
                                                    theta, 
                                                    hurst, 
                                                    covariates
                                                ],
                                                kernel=kernel
                                            )

        print(f"Acceptance rate: {tf.reduce_mean(tf.cast(kernel_results.is_accepted, dtype=tf.float32))}")

        # Now `chains` has a shape of [1, num_steps, parameter_dimension]
        #chains = tf.expand_dims(traces, axis=0)

        # Compute the potential scale reduction
        #r_hat = tfp.mcmc.potential_scale_reduction(chains)

        
        return {"initial_macro_state": initial_macro_state_samples,
                "initial_micro_state": initial_micro_state_samples,
                "macro_transitions": macro_transitions_samples,
                "micro_transitions": micro_transitions_samples,
                "macro_durations": macro_durations_samples,
                "micro_durations": micro_durations_samples,
                "binary_components": binary_components_samples,
                "theta": theta_samples,
                "hurst": hurst_samples,
                "covariates": covariates_samples, }

    def loss(self, y_true, y_pred):
        # Define your loss function here
        return tf.reduce_mean((y_true - y_pred) ** 2)

    def update(self, x, y):
        # This function performs one step of gradient descent on a single batch of data

        with tf.GradientTape() as tape:
            # Make a prediction on the input and calculate the loss
            y_pred = self.model(x)  # model() should be replaced with your actual model prediction function
            loss_value = self.loss(y, y_pred)
        
        # Calculate the gradients
        gradients = tape.gradient(loss_value, self.model.trainable_variables)

        # Update the weights of the model
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    
    def run_old(self):

        # intialize parameters
        init_params = self.initialize_parameters()

        macro_transition_matrix, micro_transition_matrix = self.define_transition_matrices()
        macro_duration_distributions, micro_duration_distributions = self.define_duration_distributions()
        initial_state_distribution = self.define_initial_state_distribution()

        # compute durations
        
        initial_macro_state_sequence = init_params["macro_state_sequence"]
        initial_micro_state_sequence = init_params["micro_state_sequence"]

        initial_macro_durations = self.compute_state_durations(initial_macro_state_sequence)

        initial_micro_state_durations = []

        for sequence in initial_micro_state_sequence:
            durations = self.compute_state_durations(sequence)
            initial_micro_state_durations.append(durations)

        
        # Compute log probabilities

        log_prob_states = self.log_prob_states(initial_macro_state_sequence, initial_micro_state_sequence)

        log_prob_durations_macro = tf.reduce_sum([
                                    tf.reduce_sum(self.macro_duration_distributions[i].log_prob(initial_macro_durations[i]))
                                    for i in range(self.n_macro_states)
                                        ])
        
        log_prob_durations_micro = tf.reduce_sum([
                                    tf.reduce_sum([
                                        self.micro_duration_distributions[i][j].log_prob(initial_micro_state_durations[i][j])
                                        for j in range(self.n_micro_states)
                                    ])
                                    for i in range(self.n_macro_states)
                                ])

        binary_components = init_params["binary_components"]
        theta = init_params["theta"]

        ic (binary_components.shape)

        states = (initial_macro_state_sequence, initial_micro_state_sequence)
        covariates = init_params["covariates"]
        hurst_param = init_params["hurst_exp"]

        scaling_process = self.msm_scaling_process(binary_components, theta, hurst_param, covariates=covariates)
        log_prob_scaling = self.log_prob_scaling(binary_components, theta, covariates=covariates, hurst_param=hurst_param)

        ic (log_prob_scaling)


        #log_prob_data_given_states = self.log_prob_data_given_states(data, states, durations)
        
        
        
        #joint_log_prob = joint_log_prob(observed_data, initial_macro_state, initial_micro_state, macro_transitions, micro_transitions, macro_durations, micro_durations, binary_components, theta, hurst, covariates)

        # Run MCMC chain to estimate parameters
        #samples, is_accepted = run_mcmc_chain(joint_log_prob)

        ic (macro_transition_matrix, micro_transition_matrix)
        ic (macro_duration_distributions, micro_duration_distributions)
        ic (initial_state_distribution)
        ic (initial_macro_state_sequence, initial_micro_state_sequence)
        #ic (initial_micro_state_durations)
        ic (log_prob_states)
        ic (log_prob_durations_macro.numpy())
        ic (log_prob_durations_micro.numpy())

        
        for key, value in init_params.items():
            ic (key)
        
        # estimate hurst exponent

        series_size = self.series_size

        series = self.series[:series_size]
        price = self.price[:series_size]
        
        returns = self.returns[:series_size] 

        series_fdiff = self.df['fracDiff'].values[:series_size]
        series_hurst = self.df['hurstExponent'].values[:series_size]
        series_rsi = self.df['rsi_ver_three'].values[:series_size]
        series_atr = self.df['atr_ver_three'].values[:series_size]

        # Generate a Gaussian random walk
        random_changes = np.random.normal(loc=0, scale=0.01, size=series_size)
        random_walk = np.cumsum(random_changes)
        covariates = [series_atr, series_rsi]

        # Plot the random walk

        hurst_est_rw = self.estimate_hurst_exponent_wav(random_walk)
        hurst_est_rc = self.estimate_hurst_exponent_wav(random_changes)
        hurst_est_series = self.estimate_hurst_exponent_wav(series)
        hurst_est_series_fdiff = self.estimate_hurst_exponent_wav(series_fdiff)

        fd_series_1, fd_series_2, fd_series_3, fd_series_4 = self.fractional_differencing(series, np.mean(hurst_est_series))
     
        print("Estimated Hurst Exponent for random walk:", hurst_est_rw, np.mean(hurst_est_rw))
        print("Estimated Hurst Exponent for random changes:", hurst_est_rc)
        print("Estimated Hurst Exponent for series:", hurst_est_series, np.mean(hurst_est_series))
        print("Estimated Hurst Exponent for series fracDiff:", hurst_est_series_fdiff)

        # adjust_for_covariates
        adjusted_base_scaling_process_1 = self.adjust_for_covariates(fd_series_1, series, covariates)
        adjusted_base_scaling_process_2 = self.adjust_for_covariates(fd_series_2, series, covariates)
        adjusted_base_scaling_process_3 = self.adjust_for_covariates(fd_series_3, series, covariates)
        adjusted_base_scaling_process_4 = self.adjust_for_covariates(fd_series_4, series, covariates)
        
        plt.plot(adjusted_base_scaling_process_1, label='adjusted_base_scaling_process_1')
        plt.plot(adjusted_base_scaling_process_2, label='adjusted_base_scaling_process_2')
        plt.plot(adjusted_base_scaling_process_3, label='adjusted_base_scaling_process_3')
        plt.plot(adjusted_base_scaling_process_4, label='adjusted_base_scaling_process_4')
        plt.plot(series, label = 'series')
        plt.legend()
        plt.show()

        # get correlation between series and fd_series
        corr_1 = np.corrcoef(series, adjusted_base_scaling_process_1)[0, 1]
        corr_2 = np.corrcoef(series, adjusted_base_scaling_process_2)[0, 1]
        corr_3 = np.corrcoef(series, adjusted_base_scaling_process_3)[0, 1]
        corr_4 = np.corrcoef(series, adjusted_base_scaling_process_4)[0, 1]

        ic (corr_1, corr_2, corr_3, corr_4)

        vol_process = self.stochastic_volatility_model(adjusted_base_scaling_process_4)

        plt.plot(vol_process, label='GARCH Volatility Process')
        plt.plot(price, label='price')
        plt.legend()
        plt.show()

        # Instantiate the distribution
        
        loc = tf.constant(0.0)
        diversity_left = tf.constant(2.0)
        diversity_right = tf.constant(0.5)
        
        asym_laplace = AsymmetricLaplace(loc, diversity_left, diversity_right)

        # Sample from the distribution
        sample = asym_laplace.sample(10_000)  # draws 10 samples

        plt.hist(sample.numpy(), bins=100)
        
        # draw the pdf
        x = np.linspace(-10, 10, 1000)
        plt.plot(x, asym_laplace.prob(x))
        
        print(f"Sample: {sample.numpy()}")  # use the .numpy() method to convert the tensor to a numpy array

        # Compute the log-probability of a sample
        log_prob = asym_laplace.log_prob(sample)
        print(f"Log-probability: {log_prob.numpy()}")

        macro_dist, micro_dist = self.define_duration_distributions(use_dist="weibull")

        for dist in macro_dist:
            sample = dist.sample(10_000)
            plt.hist(sample.numpy(), bins=100)
        
        for dist in micro_dist[:2]:
            for d in dist:
                sample = d.sample(10_000)
                plt.hist(sample.numpy(), bins=100)

    def mcmc_experiment(self):

        # Define your target log-probability function
        def target_log_prob_fn(x):
            return -x**2

        # Define your initial state
        initial_state = 1.0

        # Define your kernel
        kernel = tfp.mcmc.RandomWalkMetropolis(target_log_prob_fn)

        # Sample from your distribution
        samples, kernel_results = tfp.mcmc.sample_chain(
            num_results=1000,
            current_state=initial_state,
            kernel=kernel,
            num_burnin_steps=500)
        
        print(f"Acceptance rate: {tf.reduce_mean(tf.cast(kernel_results.is_accepted, dtype=tf.float32))}")

        plt.plot(samples.numpy())
        plt.show()

    def run(self):

        # run asymmetric laplace

        # Instantiate the distribution
        
        init_params = self.initialize_parameters()

        # unpack the initial parameters

        macro_state_sequence = init_params["macro_state_sequence"]
        self.micro_state_sequence = init_params["micro_state_sequence"]
        
        self.macro_duration_distributions = init_params["macro_duration_distributions"]
        self.micro_duration_distributions = init_params["micro_duration_distributions"]
        
        self.macro_state_distribution = init_params["macro_state_distributions"]
        self.micro_state_distribution = init_params["micro_state_distributions"]

        self.micro_transition_matrix = init_params["micro_transition_matrix"]
        self.macro_transition_matrix = init_params["macro_transition_matrix"]
        
        self.binary_components = init_params["binary_components"]
        self.theta = init_params["theta"]
        self.covariates_var = init_params["covariates"]
        self.hurst_param = init_params["hurst_exp"]
        
        num_results=1000 
        num_burnin_steps=500

        ic (init_params.keys())

        prob_est = self.joint_log_prob(macro_state_sequence)
        
        # Wrap the joint_log_prob function in a lambda so we can use it in tfp.mcmc.sample_chain
        unnormalized_posterior_log_prob = lambda *args: self.joint_log_prob(*args)

        # Define the transition kernel
        kernel_1 = tfp.mcmc.RandomWalkMetropolis(
            target_log_prob_fn=unnormalized_posterior_log_prob
        )

        # Define the transition kernel
        kernel_2 = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=unnormalized_posterior_log_prob,
            num_leapfrog_steps=2,
            step_size=0.01
        )

        rwm_kernel = DiscreteJumpFn(target_log_prob_fn=unnormalized_posterior_log_prob,
                                                        step_size=1,
                                                        num_states = self.n_macro_states)

        # Run the MCMC chain



        #[initial_hurst]
        samples, kernel_results = tfp.mcmc.sample_chain(num_results=num_results,
                                                        num_burnin_steps=num_burnin_steps,
                                                        current_state=[
                                                        macro_state_sequence
                                                        ],
                                                        kernel=rwm_kernel,
                                                        )

        print(f"Acceptance rate: {tf.reduce_mean(tf.cast(kernel_results.is_accepted, dtype=tf.float32))}")

        plt.plot(self.percent_change, label='percent change in log probability over base')
        plt.legend()
        plt.show()

        ic (prob_est)
        ic (init_params.keys())
        ic (kernel_results)
        ic (samples)
        
        mean_samples = tf.reduce_mean(samples, axis=0)
        median_samples = tfp.stats.percentile(samples, 50.0, axis=0)

        ic (mean_samples)
        ic (median_samples)


    


features_bar_returns = ['barReturn', 'cumSumPrice', 'returnOnCumSum',
                        'rocRetCumsum', 'priceGrad', 'rocPriceGrad']

ma_price = ['fracDiff', 'typical_price', 'kama_ma_short', 
            'kama_ma_mid', 'kama_ma_long', 'bbWidth_ver_two', 
            'bbWidth_ver_two', 'rsi_ver_three', 'adx_ver_three',
            'stoch_rsi_ver_two', 'hurstExponent', 'atr_ver_one', 
            'atr_ver_two', 'atr_ver_three', 'rocPriceGrad', 'dynamicRsi', 
            'xbom_zone_1', 'xbom_zone_2', 'xbom_zone_3', 'xbom_zone_4', 
            'xbom_zone_5', 'xbom_zone_6', 'xbom_zone_7', 'xbom_zone_8', 
            'xbom_zone_9', 'xbom_zone_10']

covariates = ['xbom_zone_1', 'xbom_zone_2', 'xbom_zone_3', 'xbom_zone_4', 'xbom_zone_5',
                'xbom_zone_6', 'xbom_zone_7', 'xbom_zone_8', 'xbom_zone_9', 'xbom_zone_10',
                'hurstExponent', 'atr_ver_three', 'rsi_ver_three', 'cumSumPrice']

feature_set = features_bar_returns + ma_price

df_original = pd.read_csv('ts_discrete_upload_non_discrete.csv', index_col=0)

df_original = df_original[feature_set]

lucy_markov = Lucy(3, 3, 3, df_original, 1000, covariates)

matrices =  lucy_markov.run()


nodes2 = np.asfortranarray([[0.0, 0.25,  0.5, 0.75, 1.0],
                            [0.0, 2.0 , -2.0, 2.0 , 0.0],])

curve2 = bezier.Curve(nodes2,  degree=4)
intersections = curve2.intersect(curve2)
curve2.plot(1000)
plt.plot(nodes2[0], nodes2[1], "ro")
plt.plot(intersections[0], intersections[1], "go")
plt.legend()
plt.show()


ic (intersections)