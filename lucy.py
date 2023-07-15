
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

class Lucy:

    def __init__(self, n_macro_states, n_micro_states, n_binary_components, num_quinary_components,
                 df, window_size, list_covariates, returns= 'barReturn', 
                 price='typical_price', ma_short='kama_ma_short', series_size= 5000):
        """
        Initialize Lucy with the number of macro states, micro states, and binary components.
        """
        self.n_macro_states = n_macro_states
        self.n_micro_states = n_micro_states
        self.n_binary_components = n_binary_components
        self.df_o = df
        self.window_size = window_size
        self.optimizer = tf.keras.optimizers.Adam()
        self.model = None
        self.series_size = series_size
        self.theta = tf.Variable(initial_value=tf.ones([self.n_binary_components]), name='theta')
        self.scaling_stddev = tf.Variable(1.0, dtype=tf.float32)
        self.list_covariates = list_covariates
        self.df = self.df_o[:self.series_size]
        self.validation_df = self.df[1500:]
        self.series = self.df['kama_ma_short'].values
        self.price = self.df['typical_price'].values
        self.returns = self.df['barReturn'].values
        self.n_time_steps = len(self.df)
        self.counter = 0
        self.init_log_prob = 3500
        self.percent_change = []
        self.num_quinary_components = num_quinary_components
        self.num_regimes = 4
        self.verbose = False

    def pre_process_data(self, data):

        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(data.reshape(-1, 1)).reshape(-1)

        return data

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

    def get_quinary_components(self, data):
        
        observed_data = np.array(data).reshape(-1, 1)
        num_states = self.num_quinary_components

        # Create and train Gaussian HMM

        model = hmm.GaussianHMM(n_components=num_states, covariance_type="diag")
        model.fit(observed_data)

        # Predict the optimal sequence of internal hidden state
        hidden_states = model.predict(observed_data)

        return hidden_states

    def get_quinary_components_tfp(self, data):

        num_states = self.num_quinary_components
        num_steps = self.series_size

        initial_state_distribution = tf.fill([num_states], 1.0 / num_states)

        stay_prob = 0.6
        move_prob = (1.0 - stay_prob) / (num_states - 1)

        transition_matrix = tf.fill((num_states, num_states), move_prob)
        transition_matrix = tf.linalg.set_diag(transition_matrix, [stay_prob]*num_states)

        def initial_state_prior_fn():
            return tfp.distributions.Categorical(probs=initial_state_distribution)

        def transition_fn(step, previous_state):
            return tfp.distributions.Categorical(probs=tf.gather(transition_matrix, previous_state))

        num_steps = 100

        markov_chain = tfp.distributions.MarkovChain(
            initial_state_prior=initial_state_prior_fn(),
            transition_fn=transition_fn,
            num_steps=num_steps
            )
        
        # fit the hidden markov model

        observed_data = np.array(data[100:], dtype=np.float32)

        # Initialize parameters as tf.Variables
        initial_distribution_probs = tf.Variable(tf.fill([num_states], 1.0 / num_states), trainable=True)
        transition_distribution_probs = tf.Variable(tf.fill([num_states, num_states], 1.0 / num_states), trainable=True)
        observation_distribution_locs = tf.Variable(tf.fill([num_states], 0.), trainable=True)
        observation_distribution_scales = tf.Variable(tf.fill([num_states], 1.), trainable=True)

        # Define the HMM
        model = tfp.distributions.HiddenMarkovModel(

            initial_distribution=tfp.distributions.Categorical(probs=initial_distribution_probs),
            transition_distribution=tfp.distributions.Categorical(probs=transition_distribution_probs),
            observation_distribution=tfp.distributions.Normal(loc=observation_distribution_locs, scale=observation_distribution_scales),
            num_steps=len(observed_data)
            )

        trainable_variables = [initial_distribution_probs, transition_distribution_probs, 
                               observation_distribution_locs, observation_distribution_scales]

        # Define the log prob function
        def log_prob_fn():
            return model.log_prob(observed_data)

        # Optimizer
        #optimizer = tf.keras.optimizers.Adam(learning_rate=0.00075)
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, 
                                                momentum=0.9,
                                                nesterov=True,
                                                weight_decay=0.2,
                                                clipnorm=None,
                                                clipvalue=None,
                                                global_clipnorm=None,
                                                use_ema=True,
                                                ema_momentum=0.99,)

        # Training
        for step in range(75):

            with tf.GradientTape() as tape:

                neg_log_prob = -log_prob_fn()

            grads = tape.gradient(neg_log_prob, trainable_variables)

            ic (step, neg_log_prob.numpy())

            optimizer.apply_gradients(zip(grads, trainable_variables))

        ic (trainable_variables)

        posterior_distributions = tfp.distributions.HiddenMarkovModel.posterior_marginals(
                                                        model,
                                                        
                                                    )

        most_probable_states = posterior_distributions.mode().numpy()

        new_samples = model.sample(sample_shape=(100,)).numpy()

        ic (new_samples, most_probable_states)

        # Find the most likely sequence of states
        most_likely_states = model.posterior_mode(observed_data)

        print(most_likely_states)

        return most_likely_states

    def get_covariates(self, range_window):
        
        covariates = []

        if 'price_ma_diff' in self.list_covariates:
            self.list_covariates.remove('price_ma_diff')

        for covariate in self.list_covariates:
            series = self.df[covariate].values
            series = series[range_window]
            covariates.append(series)

        price_ma_diff = self.price - self.series
        covariates.append(price_ma_diff[range_window])
        self.list_covariates.append('price_ma_diff')

        return covariates

    def adjust_for_covariates(self, base_scaling_process, series, range_window):

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

        covariates = self.get_covariates(range_window)

        # check the length of the base_scaling_process, series and make them match
        if len(base_scaling_process) != len(series):

            diff = len(series) - len(base_scaling_process)

            if diff > 0:
                series = series[diff:]
            elif diff < 0:
                base_scaling_process = base_scaling_process[abs(diff):]
            else:
                pass
        
        # reshape the covariates for the regression
        covariates = np.array(covariates)
        covariates = covariates.reshape(-1, len(covariates))

        # Let's assume X is your covariates and y is your base series
        X_train, X_test, y_train, y_test = train_test_split(covariates, base_scaling_process, 
                                                            test_size=0.2, random_state=42)

        # Create a XGBoost specific DMatrix data format
        dSeries = xgb.DMatrix(covariates, label=series, feature_names=self.list_covariates)
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.list_covariates)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=self.list_covariates)

        # Define parameters

        param = {
            'max_depth': 5,  # maximum depth of each tree
            'eta': 0.3,  # learning rate
            'objective': 'reg:squarederror',  # loss function
            'nthread': 4}  # number of CPU threads for speeding up 
        
        num_round = 20  # number of trees to build

        # Train your XGBoost model with the important features and get the predictions
        
        xgb_model = xgb.train(param, dSeries, num_round)
        predictions = xgb_model.predict(dSeries)

        # Then adjust your base series
        adjusted_base_series = base_scaling_process - predictions

        return adjusted_base_series

    def stochastic_volatility_model(self, series, heston_params = None):

        '''# estimate the stochastic volatility using GARCH model 
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
        volatility_process = np.pad(volatility_process, (10, 0), 'edge')'''

        # simulate the heston process
        if heston_params is None:

            heston_params = [ 0.34204963,  2.2920492 ,  0.3320496 ,  0.59204954, -0.2079504 ,
                                0.3320496 ]
            
        dt= 0.01
        n_steps = len(series)

        heston_model = HestonModel(series)        
        S, heston_vol = heston_model.simulate(heston_params, dt, n_steps)

        return heston_vol

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

        return hurst_mean

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
        weights = np.array([(-1)**i *binom(d, i)  for i in range(T)])
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

    def msm_scaling_process(self, quinary_components, theta, hurst_exp, range_window):

        # supress convergence warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        # Calculate the base scaling process using the binary components and theta
        
        base_scaling_process = tf.pow(quinary_components, theta)

        # get values of each column in the base scaling process
        scaling_process_list = [base_scaling_process]

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

            scaling_process_q = self.adjust_for_covariates(scaling_process, time_series, range_window)
            scaling_process_cov.append(scaling_process_q)

        # Now implement a stochastic volatility model. 
        # This could be done in a variety of ways, depending on the specific model you want to use.
        # Here's a simple example where the volatility is a stochastic process that depends on the base scaling process:
        volatility_processes = []

        for scaling_process in scaling_process_cov:

            volatility_process = self.stochastic_volatility_model(scaling_process)

            # reshape the volatility process
            volatility_process = volatility_process.reshape(-1)

            volatility_process = np.array([i*1e2 for i in volatility_process])

            volatility_processes.append(volatility_process)    

        return volatility_processes[0][1:]

    def get_macro_states(self):

        
        def get_quantiles_states(series, quantiles = [0.97, 0.5, 0.03]):

            # quantiles = [0.97, 0.5, 0.03]

            q_1 = np.around(np.quantile(series,  quantiles[0]), 3)
            q_2 = np.around(np.quantile(series,  quantiles[1]), 3)
            q_3 = np.around(np.quantile(series,  quantiles[2]), 3)

            c_1 = (series >= q_1)
            c_2 = (q_1 > series) & (series >= q_2)
            c_3 = (q_2 > series) & (series >= q_3)
            c_4 = (series < q_3)

            states = np.select([c_1, c_2, c_3, c_4], [1, 2, 3, 4], 5)

            return states

        df = self.df
        # Set NaN values to 0
        price_data = df['typical_price'].values
        atr_ver_3 = df['atr_ver_three'].values
        rsi_data_3 = df['rsi_ver_three'].values
        rsi_data_2 = df['stoch_rsi_ver_two'].values
        rsi_data_1 = df['dynamicRsi'].values

        # perform pca on RSI data

        rsi_data = np.concatenate([rsi_data_3.reshape(-1, 1), 
                                   rsi_data_2.reshape(-1, 1), 
                                   rsi_data_1.reshape(-1, 1)], axis=1)

        pca = PCA(n_components=1)
        pca.fit(rsi_data)
        series_pca = pca.transform(rsi_data)
        var_explained = pca.explained_variance_ratio_
        
        rsi_pca = series_pca

        # macro state definition

        # get quantiles of rsi and atr

        quantiles_rsi = get_quantiles_states(rsi_pca, quantiles = [0.8, 0.5, 0.2])
        quantiles_rsi = quantiles_rsi[:, 0]

        quantiles_atr = get_quantiles_states(atr_ver_3, quantiles = [0.8, 0.5, 0.2])

        c_1 = (quantiles_rsi == 1) & (quantiles_atr == 1) # bearish
        c_2 = (quantiles_rsi == 1) & (quantiles_atr == 4) # bearish
        
        c_3 = (quantiles_rsi == 4) & (quantiles_atr == 4) # bullish
        c_4 = (quantiles_rsi == 4) & (quantiles_atr == 1) # bullish
        
        c_5 = (quantiles_rsi == 1) & (quantiles_atr == 2) # bearish
        c_6 = (quantiles_rsi == 1) & (quantiles_atr == 3) # bearish
        
        c_7 = (quantiles_rsi == 2) & (quantiles_atr == 1) # choppiness, could be bearish
        c_8 = (quantiles_rsi == 2) & (quantiles_atr == 2) # choppiness, could be bearish
        c_9 = (quantiles_rsi == 2) & (quantiles_atr == 3) # choppiness, potential bear trigger
        c_10 = (quantiles_rsi == 2) & (quantiles_atr == 4) # choppiness, potential bear continuity

        c_11 = (quantiles_rsi == 3) & (quantiles_atr == 1) # choppiness
        c_12 = (quantiles_rsi == 3) & (quantiles_atr == 2) # choppiness, bullish conutinuity
        c_13 = (quantiles_rsi == 3) & (quantiles_atr == 3) # choppiness, bullish conutinuity
        c_14 = (quantiles_rsi == 3) & (quantiles_atr == 4) # bullish conutiguous

        c_15 = (quantiles_rsi == 4) & (quantiles_atr == 2) # bullish
        c_16 = (quantiles_rsi == 4) & (quantiles_atr == 3) # bullish
               
        # all bearish
        c_bear = c_1 | c_2  | c_5 | c_6 
        
        # all bullish
        c_bull = c_3 | c_4 | c_15 | c_16
        
        # bearish  continuity
        c_bear_cont =  c_7 | c_8 | c_9 | c_10  

        # bullish continuity
        c_bull_cont = c_12 | c_13 | c_14

        # all choppiness
        c_chop = c_11

        # get the states
        states = np.select([c_bull, c_bull_cont, c_bear, c_bear_cont, c_chop], 
                           ['blue', 'cyan', 'red', 'orange', 'olive'], 'white')
        
        plt.scatter (range(len(price_data)), price_data, c= states, label='multi-conditions', cmap='tab10')
        plt.plot(price_data, label='price')
        plt.colorbar()
        plt.legend()
        #plt.show()

        macro_states = np.select([c_bull, c_bull_cont, c_bear, c_bear_cont, c_chop], 
                           [0, 1, 2, 3, 4], 5)
        
            
        return macro_states

    def garchVolModel_ext(self, series):

        #  NaN or inf values to 0
        series = np.nan_to_num(series)
        
        am = arch_model(series, x=None,  #power=1.0,
                                mean='HARX', lags= 1, 
                                vol='GARCH', p=1, o=0, q=1, power = 2,
                                dist='studentst', hold_back=None, rescale=True)

        volatility_model = am.fit(update_freq=5, disp=False)

        
        garch_vol = volatility_model.conditional_volatility

        # set nan values to 0
        garch_vol = np.nan_to_num(garch_vol)

        return volatility_model, garch_vol

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
        macro_state_sequence = self.get_macro_states()

        initial_macro_state_sequence = macro_state_sequence #np.random.choice(self.n_macro_states, self.n_time_steps)
        initial_micro_state_sequence = np.random.choice(self.n_micro_states, (self.n_macro_states, self.n_time_steps))

        # Initialize binary components and theta
        initial_theta = tf.Variable(initial_value=tf.ones([self.n_binary_components]), name='theta')
        
        initial_binary_components = np.random.randint(0, 2, (self.n_time_steps, self.n_micro_states, self.n_binary_components))

        # Initialize Hurst exponent and covariates
        initial_hurst = 0.53 #self.estimate_hurst_exponent_wav(self.price)
        initial_hurst = np.around(np.mean(initial_hurst), 2)
        initial_covariates = 17.1#self.get_covariates()
        
        macro_state_distribution, micro_state_distribution = self.define_initial_state_distribution()
        macro_duration_distributions, micro_duration_distributions = self.define_duration_distributions()
        macro_transition_matrix, micro_transition_matrix = self.define_transition_matrices()

        initial_macro_state_sequence = tf.convert_to_tensor(initial_macro_state_sequence, dtype=tf.float32)
        initial_macro_state_sequence = tf.cast(initial_macro_state_sequence, dtype=tf.int32)

        lamda_k = 1.0
        k = 1.0 
        total_count=5.0
        logits=1.0

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
                    "lamda_k":lamda_k,
                    "k":k,
                    "total_count":total_count,
                    "logits":logits       

            }

        return init_params

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
    
    def define_duration_distributions(self, use_dist="weibull", lamda_k = 1.0, k = 1.0, 
                                      total_count=5.0, logits=1.0):

        """
        Define the duration distributions for each macro and micro state.
        In this example, we use Weibull distributions with shape and scale parameters set to 1.
        """

        if use_dist == "weibull":
            
            # Define the duration distributions for macro states
            macro_state_duration_distribution = [
                tfp.distributions.Weibull(concentration=lamda_k, scale=k) for _ in range(self.n_macro_states)
            ]

            # Define the duration distributions for micro states
            # Here, we assume that each macro state has the same number of micro states
            micro_state_duration_distribution = [
                [tfp.distributions.Weibull(concentration=lamda_k, scale=k) for _ in range(self.n_micro_states)]
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
                tfp.distributions.NegativeBinomial(total_count=total_count, logits=logits) for _ in range(self.n_macro_states)
            ]

            # Define the duration distributions for micro states
            # Here, we assume that each macro state has the same number of micro states
            micro_state_duration_distribution = [
                [tfp.distributions.NegativeBinomial(total_count=total_count, logits=logits) for _ in range(self.n_micro_states)]
                for _ in range(self.n_macro_states)
            ]
        
            return macro_state_duration_distribution, micro_state_duration_distribution
        
        else:
            raise ValueError("use_dist must be either weibull or negative_binomial")

    def define_initial_state_distribution(self):
        
        """
        Define the initial state distribution for macro and micro states.
        Initially, all states are equally likely.
        """
        # Define the initial state distribution for macro states
        initial_macro_state_distribution = np.ones(self.n_macro_states) / self.n_macro_states

        # Define the initial state distribution for micro states
        # Here, we assume that each macro state has the same number of micro states
        initial_micro_state_distribution = np.ones((self.n_macro_states, self.n_micro_states)) / self.n_micro_states

        return initial_macro_state_distribution, initial_micro_state_distribution

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

    def log_prob_states(self, params):

        """
        Compute the log probability of the state sequence.
        """
        macro_state_sequence = params['macro_state_sequence']
        micro_state_sequences = params['micro_state_sequence']

        micro_state_distribution = params['micro_state_distributions'] 
        macro_state_distribution = params['macro_state_distributions']

        micro_transition_matrix = params['micro_transition_matrix'] 
        macro_transition_matrix = params['macro_transition_matrix']

        #macro_duration_distributions = params['macro_duration_distributions']
        #micro_duration_distributions = params['micro_duration_distributions']

        lambda_k = params['lamda_k']
        k = params['k']

        macro_duration_distributions, micro_duration_distributions = \
                                    self.define_duration_distributions("weibull", lambda_k, k)

        # Initialize log_prob with the log probability of the initial states

        log_prob_macro = tf.math.log(macro_state_distribution[macro_state_sequence[0]])
        log_prob_macro = tf.cast(log_prob_macro, tf.float32)

        log_prob_micro = 0
        
        for i, micro_state_sequence in enumerate(micro_state_sequences):
            log_prob_micro += tf.math.log(micro_state_distribution[i, micro_state_sequence[0]])

        # Loop over the state sequence
        for t in range(1, len(macro_state_sequence)):
            # Get the current and previous states
            prev_macro_state = macro_state_sequence[t-1]
            current_macro_state = macro_state_sequence[t]
            #log_prob_macro += tf.math.log(macro_transition_matrix[prev_macro_state, current_macro_state])
            log_prob_macro += tf.cast(tf.math.log(macro_transition_matrix[prev_macro_state, current_macro_state]), tf.float32)

            # Add the log probabilities of the transitions for each micro state sequence
            for i, micro_state_sequence in enumerate(micro_state_sequences):

                prev_micro_state = micro_state_sequence[t-1]
                current_micro_state = micro_state_sequence[t]

                log_prob_micro += tf.math.log(micro_transition_matrix[prev_macro_state, prev_micro_state, current_micro_state])
                

        # log probability of durations

        initial_macro_durations = self.compute_state_durations(macro_state_sequence)

        initial_micro_state_durations = []

        for sequence in micro_state_sequences:
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

        log_prob_macro = tf.cast(log_prob_macro, tf.float32)
        log_prob_micro = tf.cast(log_prob_micro, tf.float32)
        log_prob_durations_macro = tf.cast(log_prob_durations_macro, tf.float32)
        log_prob_durations_micro = tf.cast(log_prob_durations_micro, tf.float32)

        return log_prob_macro + log_prob_micro + log_prob_durations_macro + log_prob_durations_micro

    def joint_log_prob(self, lambda_k, k):

        """
        Define the joint log probability function of the observed data, the states, 
        the duration distributions, and the MSM scaling process.
        """
        self.counter += 1

        params = {

                    "macro_state_sequence": self.macro_state_sequence,
                    "micro_state_sequence": self.micro_state_sequence,
                    
                    "macro_duration_distributions": self.macro_duration_distributions,
                    "micro_duration_distributions" : self.micro_duration_distributions,
                    
                    "macro_state_distributions" : self.macro_state_distribution,
                    "micro_state_distributions" : self.micro_state_distribution,
                    
                    "macro_transition_matrix" : self.macro_transition_matrix,
                    "micro_transition_matrix" : self.micro_transition_matrix,

                    "lamda_k" : lambda_k,
                    "k" : k,

                }
        
        log_prob_est = self.log_prob_states(params)
        log_prob_est = tf.cast(log_prob_est, tf.float32)

        improvement = (log_prob_est.numpy() - self.init_log_prob)*100/tf.abs(self.init_log_prob)
        improvement = np.around(improvement, 2)

        ic (lambda_k.numpy(), k.numpy())
        ic (self.counter, improvement)

        return log_prob_est

    def run(self):

        # Define the number of results and burn-in steps
        num_results = 1000
        num_burnin_steps = 500

        init_params = self.initialize_parameters()

        lamda_k = init_params["lamda_k"]
        k = init_params["k"]

        lambda_k = tf.cast(lamda_k, tf.float32)
        k = tf.cast(k, tf.float32)

        self.macro_state_sequence = init_params["macro_state_sequence"]
        self.micro_state_sequence = init_params["micro_state_sequence"]
        
        self.macro_duration_distributions = init_params["macro_duration_distributions"]
        self.micro_duration_distributions = init_params["micro_duration_distributions"]
        
        self.macro_state_distribution = init_params["macro_state_distributions"]
        self.macro_state_distribution= tf.cast(self.macro_state_distribution, tf.float32)

        self.micro_state_distribution = init_params["micro_state_distributions"]

        self.micro_transition_matrix = init_params["micro_transition_matrix"]
        # self.macro_transition_matrix 
        self.macro_transition_matrix = init_params["macro_transition_matrix"]
        self.macro_transition_matrix = tf.cast(self.macro_transition_matrix, tf.float32)

        log_prob = self.joint_log_prob(lambda_k, k)

        # Define the transition kernel
        
        kernel = tfp.mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn=self.joint_log_prob,
                    step_size=0.01,
                    num_leapfrog_steps=3
                )
        
        ic (lambda_k, k)

        # Run the MCMC sampler
        samples, kernel_results = tfp.mcmc.sample_chain(
                    num_results=num_results,
                    current_state= [lambda_k, k],
                    kernel=kernel,
                    num_burnin_steps=num_burnin_steps,
                    trace_fn=lambda _, pkr: pkr.is_accepted
                )

        print(f"Acceptance rate: {tf.reduce_mean(tf.cast(kernel_results.is_accepted, dtype=tf.float32))}")

        # calculate the mode of the samples
        macro_transition_matrix_mode = tfp.stats.mode(samples).mode[0]

        # calculate the mean of the samples
        macro_transition_matrix_mean = tfp.stats.mean(samples).mean()

        ic (macro_transition_matrix_mode)
        ic (macro_transition_matrix_mean)

    def run_heston(self):

        data = self.df['typical_price'].values

        data = self.pre_process_data(data)

        '''returns_data = self.df['barReturn'].values

        volatility_model, garch_vol = self.garchVolModel_ext(returns_data)

        f = Fitter(garch_vol)
        f.fit()
        # may take some time since by default, all distributions are tried
        # but you call manually provide a smaller set of distributions
        print(f.summary())'''

        heston_model = HestonModel(data)

        # Estimate parameters using Monte Carlo
        estimated_params = heston_model.MLE(data)
        
        ic(estimated_params)

        #estimated_params = [ 0.34204963,  2.2920492 ,  0.3320496 ,  0.59204954, -0.2079504 ,
        #                      0.3320496 ]
        
        dt= 0.01
        n_steps = len(data)
        
        S, V = heston_model.simulate(estimated_params, dt, n_steps)
 
    def run_scaling_process(self, range_window):

        data = self.df['typical_price'].values
        data = self.pre_process_data(data)
        data = data[range_window]

        garch_model, garch_vol = self.garchVolModel_ext(data)

        quinary_components = self.get_quinary_components(garch_vol)
        hurst_exponent = self.estimate_hurst_exponent_wav(garch_vol)
        ic (hurst_exponent)

        fracdiff_series = self.fractional_differencing(garch_vol, hurst_exponent)

        plt.plot(fracdiff_series, label='fractional_differenced_series')
        plt.plot(garch_vol, label='garch_vol')
        plt.legend()
        plt.show()

        # find covariates

        cov_adjustment = self.adjust_for_covariates(garch_vol, data, range_window)

        plt.plot(cov_adjustment, label='cov_adjustment')
        plt.plot(garch_vol, label='heston_vol')
        plt.plot(fracdiff_series, label='fractional_differenced_series')
        plt.legend()
        plt.show()

        initial_theta = 2.2

        quinary_components = np.array([i+1 for i in quinary_components])
        base_scaling_process = np.power(quinary_components, initial_theta)

        ic (base_scaling_process.shape, base_scaling_process)

        plt.plot(base_scaling_process, label='base_scaling_process')
        plt.legend()
        plt.show()

        vol_scaling_process = self.msm_scaling_process(quinary_components, initial_theta, hurst_exponent, range_window)

        ic (len(vol_scaling_process))
