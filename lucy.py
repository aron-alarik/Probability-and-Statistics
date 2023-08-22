
import numpy as np
import pandas as pd
import os
from pprint import pprint

# other libraries
from icecream import ic
import warnings
from tqdm import tqdm, trange
from colorama import Fore, Back, Style
from colorama import init
import streamlit as st
import plotly.express as px
from tqdm import tqdm, trange
import gc

# other scientific libraries
from prophet import Prophet
import pywt
import hurst
import pmdarima as pm
from BorutaShap import BorutaShap
import xgboost as xgb
import optuna
from hmmlearn import hmm
from arch import arch_model
import nevergrad as ng
from msm import glo_min, loc_min, g_LLb_h, g_LL, _LL, g_pi_t, _t, g_t, s_p, unpack

# matplotlib imports
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotx

# scipy imports 
import scipy 
from scipy.stats import norm, exponnorm
from scipy.optimize import minimize
import scipy.stats as stats
import scipy.signal as signal
from scipy.special import binom
from scipy.special import gamma, gammaln
from scipy.special import comb
from scipy.special import jv 

# sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# statsmodels imports
from statsmodels.tsa.stattools import acf

# pyro and  torch imports
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from torch.distributions import TransformedDistribution, AffineTransform, StudentT 
from pyro.infer import Predictive
from pyro.infer.autoguide import AutoNormal
from pyro import poutine

# tensorflow imports
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

# init parameters for libraries
warnings.filterwarnings("ignore") # ignore warnings
init(autoreset=True) # colorama
plt.style.use(matplotx.styles.dracula) # matplotx style
np.seterr(under='ignore')
np.seterr(all='ignore')

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
    
    # common functions
    
    def pre_process_data(self, data):
        
        if isinstance(data, pd.DataFrame):
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            data = scaler.fit_transform(data)

        else:
            scaler = MinMaxScaler(feature_range=(0, 1))
            data = scaler.fit_transform(data.reshape(-1, 1)).reshape(-1)

        return data
     
    def sliding_windows(self, data, window_size = None):

        if window_size is None:
            window_size = self.window_size
        
        data = np.array(data) if not isinstance(data, np.ndarray) else data
        
        # Calculate the number of sliding windows
        num_windows = data.shape[0] - window_size + 1
        # Create a view into the data array with the shape of the sliding windows
        shape = (num_windows, window_size) + data.shape[1:]
        strides = (data.strides[0],) + data.strides
        windows = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

        return windows

    def feature_selector_boruta_shap_xgb(self, input_df, target, num_trials = 100,
                                        params =  None,
                                        verbose = True, 
                                        target_name = 'priceGrad'):
        # define models
        cols = input_df.columns
        X = input_df #.values
        y = target #.values
        
        # pre-process data
        #X = self.pre_process_data(X)
        #y = self.pre_process_data(y)
        
        # count nan values in dataframe
        nan_count_input_df = input_df.isna().sum().sum()
        # count nan values in target array
        nan_count_target = np.isnan(y).sum()
        
        if nan_count_input_df > 0:
            st.write(f'Number of NaN values in input dataframe: {nan_count_input_df}')
            
        if nan_count_target > 0:
            st.write (f'Number of NaN values in target dataframe: {nan_count_target}')

        params = self.get_best_params(input_df, target)

        model_config = xgb.XGBRegressor(**params)
        output_dict = {}
        
        borutashap = BorutaShap(model = model_config,
                                        importance_measure = 'shap',
                                        classification = True)
        
        borutashap.fit(X = X, y = y, train_or_test = 'test', normalize=True,
                            verbose=verbose, n_trials = num_trials)

        borutashap.TentativeRoughFix()
        features_select = borutashap.accepted
        features_not_select = borutashap.tentative
        #borutashap.TentativeRoughFix()

        # create output dict from model name
        output_dict[target_name] = {'strong_features': features_select,
                                    'tentative_features': features_not_select}    
                
        # print strong features
        print('strong support: ', target_name)
        print(output_dict[target_name]['strong_features'])
        
        #'tentative', 'accepted' or 'rejected'
        #borutashap.plot(which_features='accepted', figsize=(10, 10))
        #borutashap.plot(which_features='tentative', figsize=(10, 10))

        return output_dict

    def get_best_params(self, scaled_df, target_scaled):
        
        # split data into train and test
        X_train, X_test, y_train, y_test = train_test_split(scaled_df, target_scaled, 
                                                            test_size=0.2, shuffle = False, 
                                                            random_state=42)

        #print (f'X_train shape: {X_train.shape}')
        #print (f'X_test shape: {X_test.shape}')
        #print (f'y_train shape: {y_train.shape}')
        #print (f'y_test shape: {y_test.shape}')
        np.seterr(under='ignore')

        def objective(trial):

            """Define the objective function"""
            
            params = {
                        'max_depth': trial.suggest_int('max_depth', 1, 8),
                        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.1, log=False),
                        'n_estimators': trial.suggest_int('n_estimators', 50, 1600),
                        'min_child_weight': trial.suggest_int('min_child_weight', 1, 8),
                        'gamma': trial.suggest_float('gamma', 1e-4, 0.99, log=False),
                        'subsample': trial.suggest_float('subsample', 0.01, 0.99, log=False),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 0.99, log=False),
                        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 0.99, log=False),
                        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 0.99, log=False),
                        'eval_metric': 'mlogloss',
                        'use_label_encoder': False
                    }
            
            # round off parameters to 4 decimal places if params is a float
            #params = {k: round(v, 4) if isinstance(v, float) else v for k, v in params.items()}
            
            
            # Fit the model
            optuna_model = xgb.XGBRegressor(**params)
            optuna_model.fit(X_train, y_train)

            # Make predictions
            y_pred = optuna_model.predict(X_test)

            accuracy = np.around(r2_score(y_test, y_pred), 4)
            print(Fore.GREEN + 'Accuracy:', np.around(accuracy * 100.0, 4), accuracy)
            
            '''trial.report(accuracy, step=trial.number)
            
            if trial.should_prune():
                raise optuna.TrialPruned()'''

            return accuracy
            
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10)
                
        # Report a failure to Optuna
        #trial.report(float('nan'), step=0)
        #optuna.TrialPruned()

        trial = study.best_trial

        print(f'Accuracy: {trial.value}')
        print(f'Best hyperparameters: {trial.params}')

        params = trial.params

        model = xgb.XGBRegressor(**params, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = r2_score(y_test, y_pred)
        print(Fore.GREEN + 'Accuracy: %', np.around(accuracy * 100.0, 4))

        return params

    def get_best_params_nextafter(self, scaled_df, target_scaled):
    
        # split data into train and test
        X_train, X_test, y_train, y_test = train_test_split(scaled_df, target_scaled, 
                                                            test_size=0.2, shuffle=False, 
                                                            random_state=42)

        def objective(max_depth, learning_rate, n_estimators, min_child_weight, gamma, 
                      subsample, colsample_bytree, reg_alpha, reg_lambda):
            
            # Define model parameters
            model_params = {
                'max_depth': int(max_depth),
                'learning_rate': learning_rate,
                'n_estimators': int(n_estimators),
                'min_child_weight': int(min_child_weight),
                'gamma': gamma,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'reg_alpha': reg_alpha,
                'reg_lambda': reg_lambda,
                'eval_metric': 'mlogloss',
                'use_label_encoder': False
            }

            # Fit the model
            model = xgb.XGBRegressor(**model_params)
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Compute accuracy
            accuracy = r2_score(y_test, y_pred)

            # Return negative accuracy because Nevergrad minimizes
            return -accuracy

        # Define parameter space
        param_space = ng.p.Instrumentation(
            ng.p.Scalar(lower=1, upper=8).set_integer_casting(),
            ng.p.Log(lower=0.0001, upper=0.1),
            ng.p.Scalar(lower=50, upper=1600).set_integer_casting(),
            ng.p.Scalar(lower=1, upper=8).set_integer_casting(),
            ng.p.Scalar(lower=1e-8, upper=1.0),
            ng.p.Scalar(lower=0.01, upper=1.0),
            ng.p.Scalar(lower=0.01, upper=1.0),
            ng.p.Scalar(lower=1e-8, upper=1.0),
            ng.p.Scalar(lower=1e-8, upper=1.0)
        )

        # Create optimizer
        optimizer = ng.optimizers.NGOpt(parametrization=param_space, budget=100)

        # Run optimization
        recommendation = optimizer.minimize(objective)

        # Extract best parameters
        best_params = recommendation.kwargs

        # Print results
        print(f'Best parameters: {best_params}')

        # Train final model with best parameters
        model = xgb.XGBRegressor(**best_params, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = r2_score(y_test, y_pred)
        print(Fore.GREEN + 'Accuracy: %', np.around(accuracy * 100.0, 4))

        return best_params

    def create_st_plotly_charts(self, placeholder, df):
        
        with placeholder.container():
            
                st.title(df.shape[0])
                progress_value = np.around(df.shape[0]/self.df.shape[0], 2)
                st.progress(progress_value, f'{progress_value}%')
                # Iterating through the columns of the DataFrame
                
                for col_name in df.columns:
                    # Creating two columns
                    col1, col2 = st.columns(2)
                    
                    # Creating a line chart for the current column
                    line_fig = px.line(df, x=df.index, y=col_name, title=f"{col_name} Line Chart")
                    
                    # Displaying the line chart in the first column
                    col1.plotly_chart(line_fig)
                    
                    # Creating a histogram chart for the current column
                    hist_fig = px.histogram(df, x=col_name, title=f"{col_name} Histogram")
                    
                    # Displaying the histogram chart in the second column
                    col2.plotly_chart(hist_fig)
                    
                st.dataframe(df)

    def extract_features(self, data, model_name_prefix, num_splits=10):
        
        try:
            
            data = data.numpy() if isinstance(data, torch.Tensor) else data
            # Statistical Features
            mean = np.mean(data)
            variance = np.var(data)
            skewness = stats.skew(data)
            kurtosis = stats.kurtosis(data)

            # Spectral Analysis
            freqs, psd = signal.welch(data)

            # get mean psd
            mean_psd = np.mean(psd)

            # Autocorrelation
            autocorr = np.correlate(data, data, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            features_c = ['mean', 'variance', 'skewness', 'kurtosis', 'mean_psd']
            
            features_l = [mean, variance, skewness, kurtosis, mean_psd] 

            # split the autocorr in to 5 parts and take the mean of each part
            autocorr_split = np.array_split(autocorr, num_splits)
            autocorr_split_mean = np.array([np.mean(x) for x in autocorr_split])
            
            autocorr_split_mean_diff = np.diff(autocorr_split_mean)
            
            # add 0 to beginning of autocorr_split_mean_diff
            autocorr_split_mean_diff = np.insert(autocorr_split_mean_diff, 0, 0)
            
            for i in range(1, num_splits):
                features_c.append(f'acf_{i}')
                features_l.append(autocorr_split_mean_diff[i])
            
            data_split = np.array_split(data, num_splits)
            data_mean = np.array([np.mean(x) for x in data_split])
            
            for k in range(num_splits):
                features_c.append(f'data_mean_{k+1}')
                features_l.append(data_mean[k])

            # prefix model name with model_name_prefix
            features_c = [model_name_prefix + '_' + x for x in features_c]
            
            # check for nan values in features_l, set to 0
            features_l = [0 if np.isnan(x) else x for x in features_l]
            
            # round features_l to 4 decimal places
            features_l = [round(x, 4) for x in features_l]
            
            # create dictionary from features_c and features_l
            features_dict = dict(zip(features_c, features_l))
            
            # convert all values to float
            features_dict = {k: float(v) for k, v in features_dict.items()}
            
            # set infinity  values to 0
            features_dict = {k: 0 if v == float('inf') else v for k, v in features_dict.items()}
            
            # round of all values in features_dict to 4 decimal places
            features_dict = {k: np.around(v, 4) for k, v in features_dict.items()}
            
            # check if values are too large or too small
            features_dict = {k: 0 if v > 1e+8 or v < -1e+8 else v for k, v in features_dict.items()}
            
        except Exception as e:
            # set all values to 0
            features_c = ['mean', 'variance', 'skewness', 'kurtosis', 'mean_psd']
            features_l = [0, 0, 0, 0, 0]
            
            for k in range(num_splits):
                features_c.append(f'data_mean_{k+1}')
                features_l.append(0.0)
                
            for i in range(1, num_splits):
                features_c.append(f'acf_{i}')
                features_l.append(0.0)
                
            # prefix model name with model_name_prefix
            features_c = [model_name_prefix + '_' + x for x in features_c]
            
            # create dictionary from features_c and features_l
            features_dict = dict(zip(features_c, features_l))
            
            st.title(e)
            
        return features_dict

    def get_feature_importance(self, filename, target_name):
       
        df_features_mm = pd.read_csv(filename, index_col=False)
        
        # shift df_features_mm['target'] by 1 row
        df_features_mm[target_name] = df_features_mm[target_name].shift(-1)

        # drop the last row

        df_features_mm = df_features_mm.dropna()
        target = df_features_mm.pop(target_name)
        target = target.values
        
        selected_features = lucy_markov.feature_selector_boruta_shap_xgb(
                                                df_features_mm, target, 
                                                num_trials = 100,
                                                params =  None,
                                                verbose = True, 
                                                target_name = 'price_ma_diff')
        
        st.write(selected_features)

    # base ou process
    
    def ou_log_likelihood(self, params, data, dt=0.01):
        # Unpack the parameters
        mu, kappa, sigma = params

        # Compute the drift and diffusion
        drift = kappa * (mu - data[:-1])
        diffusion = sigma * np.sqrt(dt)

        # Compute the log likelihood
        log_likelihood = tf.reduce_sum(tfp.distributions.Normal(loc=drift, scale=diffusion).log_prob(data[1:] - data[:-1]))

        return log_likelihood

    def simulate_basic_ou_process(self, params, dt, num_steps):

        mu, kappa, sigma = params
        # Initialize the process
        ou_process = np.zeros(num_steps)
        
        # Simulate the process
        for t in range(1, num_steps):
            dW = np.random.normal(0, np.sqrt(dt))  # Wiener process
            ou_process[t] = ou_process[t-1] + kappa * (mu - ou_process[t-1]) * dt + sigma * dW

        return ou_process

    def sgd_ou_process_optimize(self, data, params, learning_rate, num_iterations):
        
        params = tf.Variable(params, dtype=tf.float32, name= 'params')
        data = tf.Variable(data, dtype=tf.float32, name= 'data')

        def loss_fn(params, data):

            neg_log_likelihood = self.ou_log_likelihood(params, data) #+ epsilon
            loss = -neg_log_likelihood
            loss = tf.convert_to_tensor(loss)
            tf.summary.scalar('loss', loss, step=0)

            return loss

        # Define the optimizer with learning rate scheduling
        initial_learning_rate = learning_rate
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(  initial_learning_rate,
                                                                        decay_steps=8,
                                                                        decay_rate=0.96,
                                                                        staircase=True)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        #optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.7)
        #optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)

        loss_array = []

        for i in range(num_iterations):

            with tf.GradientTape(persistent=False, watch_accessed_variables=True) as tape:
                
                #tape.watch(params)
                #tape.watch(data)
                loss = loss_fn(params, data)
                
                
            gradients = tape.gradient(loss, params)
            #gradients = tf.where(tf.math.is_nan(gradients), tf.zeros_like(gradients), gradients)
            gradients = [tf.clip_by_value(g, -1.0, 1.0) for g in gradients]  # Gradient clipping

            optimizer.apply_gradients(zip(gradients, [params]))

            loss_array.append(loss.numpy())
            gradients_val = [g.numpy() for g in gradients]

        '''plt.plot(loss_array, label = 'loss')
        plt.legend()
        plt.show()'''

        return params.numpy()

    def lbfgs_optimize_ou_process(self, params, data):

        
        def function_to_minimize(params):

            with tf.GradientTape() as tape:

                tape.watch(params)
                loss = loss_fn(params)
            
            gradients = tape.gradient(loss, params)
            
            return loss, gradients

        # Define the loss function
        def loss_fn(params, data = data):
            neg_log_likelihood = self.ou_log_likelihood(params, data)

            return neg_log_likelihood

        
        # Convert initial parameters to a Tensor
        initial_params = params #tf.convert_to_tensor(params, dtype=tf.float32)

        # Define the optimizer
        optimizer_results = tfp.optimizer.lbfgs_minimize(
            function_to_minimize,
            initial_position=initial_params,
            max_iterations=50,
            parallel_iterations=5,
            tolerance=1e-8
        )

        # Print the optimization results
        '''print(f"Converged: {optimizer_results.converged}")
        print(f"Number of iterations: {optimizer_results.num_iterations}")
        print(f"Final loss: {optimizer_results.objective_value}")'''

        # Get the optimized parameters
        optimized_params = optimizer_results.position

        return optimized_params

    def run_base_ou_process(self, data, model_prefix, learning_rate = 0.01, num_iterations = 1000):

        mu_init = np.mean(data)
        kappa_init = 1.0
        sigma_init = np.std(data)
        dt = 0.01
        
        params = [mu_init, kappa_init, sigma_init]
        basic_ou_process_list = []

        #ll_basic_ou_process = self.ou_log_likelihood(params, data, dt)
        lbfgs_params = self.lbfgs_optimize_ou_process(params, data)
        sgd_params = self.sgd_ou_process_optimize(data, params, learning_rate, num_iterations)
        params = np.mean([sgd_params, lbfgs_params], axis = 0)

        #print (Fore.LIGHTBLUE_EX + 'Log-likelihood of basic OU process: ', ll_basic_ou_process.numpy())
        #print (Fore.LIGHTCYAN_EX + 'SGD-optimized parameters: ', sgd_params)
        #print (Fore.LIGHTCYAN_EX + 'LBFGS-optimized parameters: ', lbfgs_params.numpy())
        #print ('Mean of SGD and LBFGS parameters: ', params)

        for _ in range(1000):

            basic_ou_process = self.simulate_basic_ou_process(params, dt, num_steps = data.shape[0])
            basic_ou_process_list.append(basic_ou_process)

        # take the median value of the ou process
        basic_ou_process_median = np.median(basic_ou_process_list, axis=0)
        extracted_features = self.extract_features(basic_ou_process_median, model_prefix)

        # append basic_ou_process_median to params 
        params = np.append(params, basic_ou_process_median[-1])
 
        return params, extracted_features
                
    # picewise linear ou process
    
    def simulate_piecewise_ou_process(self, params, T, dt, data):
    
        theta, sigma, L, U, k0, k1, k2 = params
        x0 = data[0]
        num_steps = int(T)
        x = np.zeros(num_steps)
        x[0] = x0
        kappa_max = k0*3
        dt = 0.01

        for t in range(1, num_steps):

            # Calculate the mean reversion speed
            if x[t-1] < L:
                kappa = k0 + k1 * (x[t-1] - L)

            elif x[t-1] > U:
                kappa = k0 + k2 * (x[t-1] - U)
            else:
                kappa = k0
            
            # Limit kappa to avoid explosion
            kappa = min(kappa, kappa_max)

            # Euler-Maruyama step
            dW = np.random.normal(0, np.sqrt(dt))
            dx = kappa * (theta - x[t-1]) * dt + sigma * dW
            x[t] = x[t-1] + dx

        return x

    def log_likelihood_ploup(self, params, data):

        theta, sigma, L, U, kappa_0, kappa_1, kappa_2 = params

        x_prev = data[:-1]
        x_curr = data[1:]

        kappa = tf.where(x_curr < L, kappa_0 + kappa_1 * (x_curr - L),
                        tf.where(x_curr > U, kappa_0 + kappa_2 * (x_curr - U), kappa_0))
        
        drift = kappa * (theta - x_prev)

        transition_prob = tfp.distributions.Normal(loc=x_prev + drift, scale=sigma).log_prob(x_curr)
        transition_prob = tf.where(tf.math.is_nan(transition_prob), -100.0, transition_prob)

        log_likelihood = tf.reduce_sum(transition_prob)

        return -log_likelihood
    
    def hmc_optimize_ploup(self, params, data, num_samples = 100, 
                                     num_burnin_steps = 50, num_leapfrog_steps = 2, 
                                     step_size = 0.01):

        
        #Convert data to a Tensor
        #data = tf.convert_to_tensor(data, dtype=tf.float32)
        params = [tf.Variable(i, dtype=tf.float32) for i in params]
        initial_params = params
        self.counter = 0
        data = data

        # Define the log posterior function
        def log_posterior(params):

            neg_log_likelihood = -self.log_likelihood_ploup(params, data)

            return neg_log_likelihood

        # Convert initial parameters to a Tensor
        
        #tf.convert_to_tensor(initial_params, dtype=tf.float32)

        # Define the joint log probability function
        def joint_log_prob(*params):
            params = list(params)
            self.counter += 1
            #print ('iteration:', self.counter)
            return log_posterior(params)

        # Define the target log probability function for the HMC sampler
        target_log_prob_fn = lambda *args: joint_log_prob(*args)

        # Define the HMC transition kernel
        num_adaptation_steps = int(0.8 * num_burnin_steps)  # Adapt the step size during the first 80% of burn-in steps

        hmc = tfp.mcmc.HamiltonianMonteCarlo(target_log_prob_fn=target_log_prob_fn,
                                            step_size=step_size,
                                            num_leapfrog_steps=num_leapfrog_steps)

        # Define the adaptive step size adjustment mechanism
        adaptive_step_size = tfp.mcmc.SimpleStepSizeAdaptation(inner_kernel=hmc,
                                                            num_adaptation_steps=num_adaptation_steps,
                                                            target_accept_prob=0.75)

        # Initialize the state of the HMC sampler
        current_state = initial_params

        # Run the HMC sampler
        samples, _ = tfp.mcmc.sample_chain(num_results=num_samples,
                                            num_burnin_steps=num_burnin_steps,
                                            current_state=current_state,
                                            kernel=adaptive_step_size)

        # Get the optimized parameters from the last sample
        optimized_results = []
        for sample in samples:
            
            # return the median of the optimized_params
            optimized_params_median = np.median(sample, axis=0)

            optimized_results.append(optimized_params_median)

        return optimized_results

    def lbfgs_optimize_ploup(self, data, params):

        def function_to_minimize(params):

            with tf.GradientTape() as tape:

                tape.watch(params)
                loss = loss_fn(params)
            
            gradients = tape.gradient(loss, params)
            
            return loss, gradients

        # Define the loss function
        def loss_fn(params, data = data):
            neg_log_likelihood = self.log_likelihood_ploup(params, data)

            return neg_log_likelihood

        # Convert initial parameters to a Tensor
        initial_params = params #tf.convert_to_tensor(params, dtype=tf.float32)

        # Define the optimizer
        optimizer_results = tfp.optimizer.lbfgs_minimize(
            function_to_minimize,
            initial_position=initial_params,
            max_iterations=50,
            parallel_iterations=5,
            tolerance=1e-8
        )

        # Print the optimization results
        '''print(f"Converged: {optimizer_results.converged}")
        print(f"Number of iterations: {optimizer_results.num_iterations}")
        print(f"Final loss: {optimizer_results.objective_value}")'''

        # Get the optimized parameters
        optimized_params = optimizer_results.position

        return optimized_params

    def run_plou_process(self, data_window, model_name_prefix):

        data = data_window

        theta_init = np.mean(data)
        sigma_init = np.std(data)
        L_init = np.percentile(data, 25)
        U_init = np.percentile(data, 75)
        kappa_0_init = 0.01
        kappa_1_init = 0.01
        kappa_2_init = 0.01

        initial_params = [theta_init, sigma_init, L_init, U_init, 
                            kappa_0_init, kappa_1_init, kappa_2_init]
        
        params_hmc  = self.hmc_optimize_ploup(initial_params, data, num_samples=1000, 
                                              num_burnin_steps=500, step_size=0.01, 
                                              num_leapfrog_steps=3)
        
        
        
        params_lbfgs = self.lbfgs_optimize_ploup(data, initial_params)
        params_mean = params_lbfgs.numpy()

        #params_sgd = self.sgd_optimize_ploup(data, initial_params, 
        #                                        learning_rate, num_iterations)
        
        #ic (params_sgd)
        
        # take mean of params
        #params_mean = np.mean([params_lbfgs, params_sgd], axis=0)
        #ic (params_mean)

        theta, sigma, L, U, kappa_0, kappa_1, kappa_2 = params_mean
        
        # simulate the OU process
        T = len(data) if type(data) == list else data.shape[0]
        dt = 1
        
        plou_process_list = []
        
        for _ in range(100):
            plou_process = self.simulate_piecewise_ou_process(params_mean, T, dt, data)
            plou_process_list.append(plou_process)
        
        # get the median of the simulated processes
        #plou_process_n = np.mean(plou_process_list, axis=0)
        plou_process_m = np.median(plou_process_list, axis=0)
        
        #plou_process = np.where(np.isinf(plou_process_m), 0.1, plou_process_m)

        extracted_features = self.extract_features(plou_process_m, model_name_prefix)
        
        # Plot the mean reversion speed as a function of the process value
        process_values = np.linspace(data[0], data[-1], T)

        mean_reversion_speed = np.mean([kappa_0 + kappa_1 * (x - L) if x < L else \
                                (kappa_0 + kappa_2 * (x - U) if x > U else kappa_0) \
                                for x in process_values])
        
        # combine the features into a list
        feature_list = np.array([theta, sigma, L, U, kappa_0, kappa_1, kappa_2, 
                        mean_reversion_speed, plou_process_m[-1]])
        
        return feature_list, extracted_features
   

    # time varying ou process
        
    def log_likelihood_tv_ou_process(self, params, data, dt=None, size=None):

        """
        Computes the log-likelihood of the data given the model parameters.
        
        :param params: Parameters of the model
        :param data: Observed data points
        :param dt: Time step size
        :return: Log likelihood value
        """

        if size is None:
            size = len(data) if type(data) == list else data.shape[0]
            #data.numpy().shape[0]

        if dt is None:
            dt = 0.01

        '''# if kappa is not tf constant then convert to tf constant
        if not isinstance(kappa, tf.Tensor):
            kappa = tf.constant(kappa)

        # if scale is not tf constant then convert to tf constant
        if not isinstance(scale, tf.Tensor):
            scale = tf.constant(scale)'''

        
        # Unpack parameters
        x0 = data[0]  # Initial state
        b1, b2, s1, s2, s3, s_diff, k_diff, sigma, loc, kappa, scale = params
        
        # Compute the piecewise linear drift based on the breakpoints and slopes
        def drift(x):
            return tf.where(
                x <= b1, s1 * (x - b1),
                tf.where(x <= b2, s2 * (x - b2), s3 * (x - b2))
            )
        
        # Compute the time-varying diffusion function
        def diffusion(x):
            return s_diff * tf.exp(-k_diff * x)
        
        # Compute the transition probability using Gauss-Hermite quadrature
        def transition_prob(x, y, dt):
            mu = x + drift(x) * dt
            sigma_value = diffusion(x)
            scaled_diff = (y - mu) / tf.sqrt(2 * sigma_value**2 * dt)
            return tf.exp(-scaled_diff**2) / tf.sqrt(2 * np.pi * sigma_value**2 * dt)
        
        '''n_quad = 10
        nodes, weights = self.gauss_hermite_nodes_weights(n_quad)
        nodes = tf.constant(nodes, dtype=tf.float32)
        weights = tf.constant(weights, dtype=tf.float32)'''

        # Compute the log-likelihood
        log_likelihood_value = 0.0
        log_likelihood_value_old = 0.0

        #for t in range(1, size):
        #    log_likelihood_value_old += tf.math.log(transition_prob(data[t - 1], data[t], dt)) # type: ignore

        '''for t in range(1, size):
            mu = data[t-1] + drift(data[t-1]) * dt
            scale = diffusion(data[t-1]) * tf.sqrt(dt)
            
            # Normal distribution for the increment
            normal_dist = tfd.Normal(loc=mu, scale=scale)
            
            # Increment the log likelihood
            log_likelihood_value += normal_dist.log_prob(data[t])

        ic (log_likelihood_value)'''

        # Vectorized drift and diffusion
        mus = data[:-1] + drift(data[:-1]) * dt
        scales = diffusion(data[:-1]) * tf.sqrt(dt)

        # Logistic distribution for the increments
        logistic_dist = tfd.Logistic(loc=mus, scale=scales)

        # Normal distribution for the increments
        normal_dists = tfd.Normal(loc=mus, scale=scales)

        # Asymmetric_laplace_dists Vectorized drift and diffusion
        mus_as = data[:-1] + drift(data[:-1]) * dt
        scales_left = diffusion(data[:-1]) * tf.sqrt(dt) # You need to define diffusion_left
        scales_right = diffusion(data[:-1]) * tf.sqrt(dt) # You need to define diffusion_right

        '''# AsymmetricLaplace distribution for the increments
        asymmetric_laplace_dists = AsymmetricLaplace(loc=mus_as, 
                                                     diversity_left=scales_left, 
                                                     diversity_right=scales_right)
        
        # asymmetric_laplace_dists with kappa
        asymmetric_laplace_dists_kappa = AsymmetricLaplaceKappa(loc=mus_as, kappa=kappa, scale=scale)

        # HyperbolicSecant
        #hs = HyperbolicSecant(loc=-0.0437, scale=7.84)
        hs = HyperbolicSecant(loc=mus, scale=scales)

        # Double Gamma
        alpha = 1.17
        double_gamma_dists = DoubleGamma(alpha=alpha, beta=scales, loc=mus)
        #double_gamma_dists = tfd.Gamma(loc=mus, scale=scales)'''

        # students t
        df = 3.6
        students_t_dists = tfd.StudentT(df=df, loc=mus, scale=scales)


        # range wise cyclotomy
        '''try:
            range_cyclotomy = self.range_wise_cyclotomy(logistic_dist, num_segments=10)
        except Exception as e:
            print(e)'''


        # Vectorized log likelihood Logistic
        '''log_likelihood_values_logistic = logistic_dist.log_prob(data[1:])
        total_log_likelihood_logistic = tf.reduce_sum(log_likelihood_values_logistic)
        
        # Vectorized log likelihood
        log_likelihood_values_normal = normal_dists.log_prob(data[1:])
        total_log_likelihood_normal = tf.reduce_sum(log_likelihood_values_normal)

        # Vectorized log likelihood
        log_likelihood_values_hypersecant = hs.log_prob(data[1:])
        total_log_likelihood_hypersecant = tf.reduce_sum(log_likelihood_values_hypersecant)

        log_likelihood_a_laplace = asymmetric_laplace_dists.log_prob(data[1:])
        total_log_likelihood_a_laplace = tf.reduce_sum(log_likelihood_a_laplace)

        log_likelihood_a_laplace_kappa = asymmetric_laplace_dists_kappa.log_prob(data[1:])
        total_log_likelihood_a_laplace_kappa = tf.reduce_sum(log_likelihood_a_laplace_kappa)

        log_likelihood_double_gamma = double_gamma_dists.log_prob(data[1:])
        total_log_likelihood_double_gamma = tf.reduce_sum(log_likelihood_double_gamma)'''

        log_likelihood_students_t = students_t_dists.log_prob(data[1:])
        total_log_likelihood_students_t = tf.reduce_sum(log_likelihood_students_t)
        
        #log_probs = double_gamma_dists.log_prob(data[1:])
        #print(log_probs.numpy())

        '''ic (total_log_likelihood_logistic.numpy())
        ic(total_log_likelihood_normal.numpy())
        ic(total_log_likelihood_hypersecant.numpy())
        ic(total_log_likelihood_a_laplace.numpy())
        ic(total_log_likelihood_double_gamma.numpy())
        ic(total_log_likelihood_a_laplace_kappa.numpy())
        ic(total_log_likelihood_students_t.numpy())'''

        # total_log_likelihood_logistic, total_log_likelihood_normal, 
        # total_log_likelihood_hypersecant, total_log_likelihood_a_laplace, 
        # total_log_likelihood_double_gamma

        # Return the log-likelihood
        #ic (total_log_likelihood_a_laplace.numpy())

        #ic(total_log_likelihood_a_laplace_kappa.numpy())
        #ic(total_log_likelihood_a_laplace.numpy())

        return_ll_value = total_log_likelihood_students_t

        # changes in log likelihood
        current_val = self.init_log_prob
        change_val = return_ll_value.numpy() - current_val
        percent_change = (change_val / current_val) * 100
        self.percent_change.append(percent_change)

        return return_ll_value #total_log_likelihood_a_laplace

    def simulate_tv_ou_process(self, params, data):

        # Your drift function
        def drift(x, b1, s1, b2, s2, s3):
            return tf.where(
                x <= b1, s1 * (x - b1),
                tf.where(x <= b2, s2 * (x - b2), s3 * (x - b2))
            )

        # Your diffusion function
        def diffusion(x, s_diff, k_diff):
            return s_diff * tf.exp(-k_diff * x)
        
        # Unpack parameters
        b1, b2, s1, s2, s3, s_diff, k_diff, sigma, loc, kappa, scale = params
        
        # Initialize the process
        T = len(data) if type(data) == list else data.shape[0]
        dt = 0.01
        num_steps = T
        x = np.zeros(num_steps)
        x0 = data[0]
        x[0] = x0

        # Simulate the process
        for t in range(1, num_steps):
            # Euler-Maruyama step
            dW = np.random.normal(0, np.sqrt(dt))
            dx = drift(x[t-1], b1, s1, b2, s2, s3) * dt + diffusion(x[t-1], s_diff, k_diff) * dW
            x[t] = x[t-1] + dx

        return x

    def sgd_optimize_tv_ou_process(self, data, initial_params, learning_rate, num_iterations, 
                                   validation_data):

        learning_rate = tf.constant(learning_rate, dtype=tf.float32)
        params = tf.Variable(initial_params, dtype=tf.float32, name="params")
        data = tf.Variable(data, dtype=tf.float32, name="data")
        validation_data = tf.Variable(validation_data, dtype=tf.float32, name="validation_data")
        epsilon = 1e-10

        # Initialize variables for early stopping
        best_params = None
        best_loss = float('inf')
        num_iterations_without_improvement = 0
        patience=8

        # log_likelihood_tv_ou_process

        def loss_fn(params, data):
            neg_log_likelihood = -self.log_likelihood_tv_ou_process(params, data) + epsilon
            loss = neg_log_likelihood
            loss = tf.convert_to_tensor(loss)
            tf.summary.scalar('loss', loss, step=0)
            return loss

        # Define the optimizer with learning rate scheduling
        initial_learning_rate = learning_rate
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(  initial_learning_rate,
                                                                        decay_steps=8,
                                                                        decay_rate=0.96,
                                                                        staircase=True)
        
        #optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.7)
        #optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)

        loss_array = []

        for i in range(num_iterations):

            with tf.GradientTape(persistent=False, watch_accessed_variables=True) as tape:
                loss = loss_fn(params, data)

            gradients = tape.gradient(loss, params)
            #gradients = tf.where(tf.math.is_nan(gradients), tf.zeros_like(gradients), gradients)
            gradients = [tf.clip_by_value(g, -1.0, 1.0) for g in gradients]  # Gradient clipping

            optimizer.apply_gradients(zip(gradients, [params]))

            #print(f"Loss at iteration {i}: {loss}")
            loss_array.append(loss.numpy())
            gradients_val = [g.numpy() for g in gradients]
            #print(f"Gradients at iteration {i}: {gradients_val}")

            # Check the loss on the validation set for early stopping
            if i % patience == 0:
                validation_loss = loss_fn(params, validation_data)
                if validation_loss <= best_loss:
                    num_iterations_without_improvement += 1
                else:
                    best_params = params.numpy()
                    best_loss = validation_loss
                    num_iterations_without_improvement = 0

                # Check the stopping criterion
                
                if num_iterations_without_improvement >= patience:
                    break

        # Use the best parameters found during early stopping
        if best_params is not None:
            params = tf.Variable(best_params)

        '''plt.plot(loss_array, label="loss")
        plt.legend()
        plt.show()'''

        return params.numpy()
    
    def lbfgs_optimize_tv_ou(self, data, initial_params):

        def function_to_minimize(params):
            with tf.GradientTape() as tape:
                tape.watch(params)
                loss = loss_fn(params)
            gradients = tape.gradient(loss, params)
            return loss, gradients

        # Define the loss function
        def loss_fn(params):
            neg_log_likelihood = self.log_likelihood_tv_ou_process(params, data)
            return neg_log_likelihood

        # Define the function to minimize
        #def function_to_minimize(params):
        #    return loss_fn(params), tf.gradients(loss_fn(params), params)[0]

        # Convert initial parameters to a Tensor
        initial_params = tf.convert_to_tensor(initial_params, dtype=tf.float32)

        # Define the optimizer
        optimizer_results = tfp.optimizer.lbfgs_minimize(
            function_to_minimize,
            initial_position=initial_params,
            tolerance=1e-6,
        )

        # Print the optimization results
        '''print(f"Converged: {optimizer_results.converged}")
        print(f"Number of iterations: {optimizer_results.num_iterations}")
        print(f"Final loss: {optimizer_results.objective_value}")'''

        # Get the optimized parameters
        optimized_params = optimizer_results.position

        return optimized_params

    def run_ou_tv_process(self, data, valid_data, learning_rate = 0.001, num_iterations = 500):

        b1_init = np.percentile(data, 33)
        b2_init = np.percentile(data, 66)
        s1_init = -0.1
        s2_init = 0
        s3_init = 0.1
        s_diff_init = np.std(np.diff(data)) / np.sqrt(252)
        k_diff_init = 0.01
        sigma_init = np.std(np.diff(data)) / np.sqrt(252)
        dt = 0.01  # Time step size
        
        loc = tf.constant(3.0)
        kappa = tf.constant(1.5)
        scale = tf.constant(2.0)

        learning_rate = learning_rate
        num_iterations = num_iterations
        num_samples = 1500
        num_burnin_steps = 500
        num_leapfrog_steps = 3
        step_size = 0.01
        
        initial_params = [b1_init, b2_init, s1_init, s2_init, s3_init, 
                                        s_diff_init, k_diff_init, sigma_init,
                                        loc, kappa, scale]
        
        params_lbfgs = self.lbfgs_optimize_tv_ou(data, initial_params)

        params_sgd = self.sgd_optimize_tv_ou_process(data, initial_params, learning_rate, 
                                                    num_iterations, valid_data)

        # take mean of params_lbfgs and params_sgd
        params = np.mean([params_lbfgs, params_sgd], axis=0)
        
        tv_ou_process_list = []

        for _ in range(50):
    
            tv_ou_process_a = self.simulate_tv_ou_process(params, data)
            tv_ou_process_list.append(tv_ou_process_a)

        tv_ou_process = np.mean(tv_ou_process_list, axis=0)

        extracted_features = self.extract_features(tv_ou_process, model_name_prefix='tvoup')

        # extract parameters from tv_ou_process
        #ic (params_lbfgs, params_sgd, params, extracted_features)
        
        #params_hmc = self.hmc_optimize_tv_ou(data, initial_params, num_samples, 
        #                num_burnin_steps, num_leapfrog_steps, step_size)
        
        #ic (params_lbfgs, params_hmc, params_sgd)        

        # take the mean value of params_lbfgs, params_hmc, params
        #params = np.mean([params_lbfgs, params_hmc, params_sgd], axis=0)

        params = np.append(params, tv_ou_process[-1])
        #feature_list = np.concatenate((params, extracted_features))

        return params, extracted_features
    
    # levy process using Pyro
    
    def get_jump_durations(self, data, delta = 300):

        list_prev_val = []

        for i in range(1, len(data)):
            if data[i] != 0:
                pass

            elif data[i] == 0:
                list_prev_val.append(data[i-1])

        # remove all values that are 0
        list_prev_val = [int(i + delta) for i in list_prev_val] # if i != 0]        

        return list_prev_val

    def levy_jump_diffusion_model(self, data):

        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)

        # Define priors over the parameters
        mu = pyro.sample('mu', dist.Normal(0., 1.))
        sigma = pyro.sample('sigma', dist.HalfNormal(1.))
        lambda_ = pyro.sample('lambda', dist.Exponential(1.))
        alpha = pyro.sample('alpha', dist.Normal(0., 1.))
        beta = pyro.sample('beta', dist.HalfNormal(1.))

        # Define the jump term
        jump_term = pyro.sample('jump_term', dist.Poisson(lambda_))

        len_data = len(data) if isinstance(data, list) else data.shape[0]

        # Define the diffusion process
        for i in range(len_data):
            drift_term = mu * data[i]
            diffusion_term = sigma * torch.randn(1)
            jump_size = pyro.sample(f'jump_size_{i}', dist.Normal(alpha, beta))

            # Update the process
            if i < len(data) - 1:
                data[i + 1] = data[i] + drift_term + diffusion_term + jump_term * jump_size

    def guide_func_levy(self, data):

        # Define variational parameters
        mu_loc = pyro.param('mu_loc', torch.tensor(0.))
        mu_scale = pyro.param('mu_scale', torch.tensor(1.), constraint=dist.constraints.positive)
        sigma_scale = pyro.param('sigma_scale', torch.tensor(1.), constraint=dist.constraints.positive)
        lambda_rate = pyro.param('lambda_rate', torch.tensor(1.), constraint=dist.constraints.positive)
        alpha_loc = pyro.param('alpha_loc', torch.tensor(0.))
        alpha_scale = pyro.param('alpha_scale', torch.tensor(1.), constraint=dist.constraints.positive)
        beta_scale = pyro.param('beta_scale', torch.tensor(1.), constraint=dist.constraints.positive)

        # Sample latent variables
        mu = pyro.sample('mu', dist.Normal(mu_loc, mu_scale))
        sigma = pyro.sample('sigma', dist.HalfNormal(sigma_scale))
        lambda_ = pyro.sample('lambda', dist.Exponential(lambda_rate))
        alpha = pyro.sample('alpha', dist.Normal(alpha_loc, alpha_scale))
        beta = pyro.sample('beta', dist.HalfNormal(beta_scale))

        # Sample the jump term
        jump_term = pyro.sample('jump_term', dist.Poisson(lambda_))

         # Sample the jump sizes
        for i in range(len(data)):
            jump_size = pyro.sample(f'jump_size_{i}', dist.Normal(alpha, beta))

    def stoch_var_inference_levy(self, data, num_steps = 25):

        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)


        optimizer_sgd = pyro.optim.SGD({'lr': 0.001, 'momentum':0.90})

        # Perform stochastic variational inference
        svi = pyro.infer.SVI(model=self.levy_jump_diffusion_model,
                            guide=self.guide_func_levy,
                            #optim=pyro.optim.Adam({'lr': 0.001}),
                            #optim=pyro.optim.RMSprop({"lr": 0.001}),
                            optim=optimizer_sgd,
                            #loss=pyro.infer.Trace_ELBO(), 
                            loss=pyro.infer.TraceGraph_ELBO(), 
                            #loss=pyro.infer.TraceMeanField_ELBO()
                            )

        num_steps = num_steps  # Number of optimization steps
        loss_array = []

        for step in trange(num_steps):
            loss = svi.step(data)
            loss_array.append(loss)
            #print(f'Step {step}, Loss = {loss}')

        '''plt.plot(loss_array, label = 'loss')
        plt.legend()
        plt.show()'''

        #for name, value in pyro.get_param_store().items():
        #    print(name, pyro.param(name))

        mu_optimized = pyro.param('mu_loc').item()
        sigma_optimized = pyro.param('sigma_scale').item()
        lambda_optimized = pyro.param('lambda_rate').item()
        alpha_optimized = pyro.param('alpha_loc').item()
        beta_optimized = pyro.param('beta_scale').item()

        optimum_params = (mu_optimized, sigma_optimized, 
                          lambda_optimized, alpha_optimized, beta_optimized)
        
        return optimum_params

    def detect_jumps(self, data, threshold):

        jumps = torch.abs(data[1:] - data[:-1]) > threshold
        jump_directions = torch.sign(data[1:] - data[:-1])

        return jumps, jump_directions

    def predict_jump(self, lambda_, alpha, beta):

        time_to_next_jump = dist.Exponential(lambda_).sample()
        jump_size = dist.Normal(alpha, beta).sample()

        return time_to_next_jump, jump_size

    def simulate_levy_process(self, mu, sigma, lambda_, alpha, beta, num_steps):

        data = torch.zeros(num_steps)
        for i in range(num_steps - 1):
            drift_term = mu * data[i]
            diffusion_term = sigma * torch.randn(1)
            jump_term = dist.Poisson(lambda_).sample()
            jump_size = dist.Normal(alpha, beta).sample() if jump_term > 0 else 0
            data[i + 1] = data[i] + drift_term + diffusion_term + jump_term * jump_size

        return data


    # regime switching ou process

    def simulate_ou_process(self, params, data, num_sim=1000):

        mu, theta, sigma, state_sequence = params
        dt = 0.01
        n_steps = len(data) if isinstance(data, list) else data.shape[0]
        last_state = state_sequence[-1]

        x0 = data[-1]
        mu= mu[last_state]
        theta = theta[last_state]
        sigma = sigma[last_state]

        x = np.zeros(n_steps)
        x[0] = x0

        ou_simulations = []
        for _ in range(num_sim):
        
            for t in range(1, n_steps):
                dx = theta * (mu - x[t-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
                x[t] = x[t-1] + dx
            ou_simulations.append(x)

        # take the mean of ou simulations
        ou_simulations = np.mean(ou_simulations, axis = 0)
        '''plt.plot(ou_simulations, label = 'simulated')
        plt.legend()
        plt.show()'''
        # extract features from ou simulations
        ou_sim_features = self.extract_features(ou_simulations, model_name_prefix='rsoup', num_splits=10)

        return ou_sim_features

    def rsoup_model(self, data):
        # Number of states
        n_states = self.num_regimes

        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)

        # Prior distributions for the OU parameters
        mu_prior = dist.Normal(0, 1).expand([n_states]).to_event(1)
        theta_prior = dist.LogNormal(0, 1).expand([n_states]).to_event(1)
        sigma_prior = dist.LogNormal(0, 1).expand([n_states]).to_event(1)

        # Draw the OU parameters
        mu = pyro.sample("mu", mu_prior)
        theta = pyro.sample("theta", theta_prior)
        sigma = pyro.sample("sigma", sigma_prior)

        # Prior distribution for the transition matrix
        transition_matrix_prior = dist.Dirichlet(torch.ones(n_states)).expand([n_states]).to_event(1)

        # Draw the transition matrix
        transition_matrix = pyro.sample("transition_matrix", transition_matrix_prior)

        # Prior distribution for the initial state
        initial_state_prior = dist.Categorical(torch.ones(n_states))

        # Draw the initial state
        state = pyro.sample("initial_state", initial_state_prior)

        # Loop over the observed data
        for t in range(len(data)):
            # Prior distribution for the current state
            current_state_prior = dist.Categorical(transition_matrix[state])

            # Draw the current state
            state = pyro.sample(f"state_{t}", current_state_prior)

            # OU process for the current state
            ou_process = dist.Normal(mu[state], sigma[state])

            # Draw the current observation
            pyro.sample(f"obs_{t}", ou_process, obs=data[t])

    def rsoup_guide(self, data):

        # Number of states
        n_states = self.num_regimes

        # Guide distributions for the OU parameters
        mu_loc = pyro.param("mu_loc", torch.zeros(n_states))
        mu_scale = pyro.param("mu_scale", torch.ones(n_states), constraint=dist.constraints.positive)
        theta_loc = pyro.param("theta_loc", torch.ones(n_states))  # Use ones for initialization
        theta_scale = pyro.param("theta_scale", torch.ones(n_states), constraint=dist.constraints.positive)
        sigma_loc = pyro.param("sigma_loc", torch.ones(n_states))  # Use ones for initialization
        sigma_scale = pyro.param("sigma_scale", torch.ones(n_states), constraint=dist.constraints.positive)

        # Draw the OU parameters
        mu = pyro.sample("mu", dist.Normal(mu_loc, mu_scale).to_event(1))  # Add to_event(1)
        theta = pyro.sample("theta", dist.LogNormal(theta_loc, theta_scale).to_event(1))  # Use LogNormal distribution
        sigma = pyro.sample("sigma", dist.LogNormal(sigma_loc, sigma_scale).to_event(1))  # Use LogNormal distribution

        # Guide distribution for the transition matrix
        transition_matrix_concentration = pyro.param("transition_matrix_concentration", torch.ones(n_states, n_states), constraint=dist.constraints.positive)

        # Draw the transition matrix
        transition_matrix = pyro.sample("transition_matrix", dist.Dirichlet(transition_matrix_concentration).to_event(1))  # Add to_event(1)

        # Guide distribution for the initial state
        initial_state_probs = pyro.param("initial_state_probs", torch.ones(n_states), constraint=dist.constraints.simplex)

        # Draw the initial state
        state = pyro.sample("initial_state", dist.Categorical(initial_state_probs))

        # Loop over the observed data
        for t in range(len(data)):
            # Guide distribution for the current state
            current_state_probs = pyro.param(f"current_state_probs_{t}", torch.ones(n_states), constraint=dist.constraints.simplex)

            # Draw the current state
            state = pyro.sample(f"state_{t}", dist.Categorical(current_state_probs))

    def rsoup_inference(self, data, num_steps=2500):
        # Set up the optimizer
        optimizer = pyro.optim.Adam({"lr": 0.001})

        # Set up the inference algorithm
        svi = SVI(self.rsoup_model, self.rsoup_guide, optimizer, loss=Trace_ELBO())

        # Run the inference algorithm
        n_steps = num_steps

        loss_list = []

        for step in trange(n_steps):
            loss = svi.step(data)
            loss_list.append(loss)
            #print(f"Step {step}, loss = {loss}")

        # Plot the loss
        '''plt.plot(loss_list)
        plt.title("ELBO")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.show()'''

        # Extract the state sequence
        state_sequence = [pyro.param(f"current_state_probs_{t}").argmax().item() for t in range(len(data))]

        # Extract the OU parameters
        mu = pyro.param("mu_loc").detach().numpy()
        theta = pyro.param("theta_loc").detach().numpy()
        sigma = pyro.param("sigma_loc").detach().numpy()

        '''print("State sequence:", state_sequence[-self.num_regimes:])
        print("OU parameters:")
        print("mu:", mu)
        print("theta:", theta)
        print("sigma:", sigma)'''

        params = [mu, theta, sigma, state_sequence[-self.num_regimes:]]

        return params
    
    # multifractal model
        
    def multi_fractal_model(self, data):

        # Prior distributions for the model parameters
        n = self.num_regimes
        p = pyro.sample('p', dist.Beta(1, 1))
        b = pyro.sample('b', dist.Beta(1, 1))
        m = pyro.sample('m', dist.Categorical(torch.ones(n)))
        sigma = pyro.sample('sigma', dist.HalfNormal(1))

        v_list = [sigma.item()]

        for t in range(1, len(data)):
            v_t = b * (p ** m) * v_list[t - 1] + 1e-6
            v_list.append(v_t)

        v = torch.tensor(v_list)

        '''# Volatility process
        v = torch.zeros_like(data)
        v[0] = sigma

        for t in range(1, len(data)):
            v[t] = b * (p ** m) * v[t - 1] +  1e-6'''

        # Likelihood
        with pyro.plate('data', len(data)):

            pyro.sample('obs', dist.StudentT(df=3, loc=0, scale=v), obs=data)
            #pyro.sample('obs', dist.Normal(0, v), obs=data)

    def multi_fractal_guide(self, data):

        n = self.num_regimes
        # Variational parameters
        alpha_p = pyro.param('alpha_p', torch.tensor(1.), constraint=dist.constraints.positive)
        beta_p = pyro.param('beta_p', torch.tensor(1.), constraint=dist.constraints.positive)
        alpha_b = pyro.param('alpha_b', torch.tensor(1.), constraint=dist.constraints.positive)
        beta_b = pyro.param('beta_b', torch.tensor(1.), constraint=dist.constraints.positive)
        probs_m = pyro.param('probs_m', torch.ones(n), constraint=dist.constraints.simplex)
        scale_sigma = pyro.param('scale_sigma', torch.tensor(1.), constraint=dist.constraints.positive)

        # Variational distributions
        pyro.sample('p', dist.Beta(alpha_p, beta_p))
        pyro.sample('b', dist.Beta(alpha_b, beta_b))
        pyro.sample('m', dist.Categorical(probs_m))
        pyro.sample('sigma', dist.HalfNormal(scale_sigma))

    def multi_fractal_inference(self, data, num_steps=2500):

        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)

        #pyro.clear_param_store()
        model = self.multi_fractal_model
        guide = self.multi_fractal_guide

        # Define the optimizer and loss function
        adam = pyro.optim.Adam({'lr': 0.001})
        elbo = Trace_ELBO()

        # Define the inference algorithm
        svi = SVI(model, guide, adam, loss=elbo)

        # Run the inference algorithm
        n_steps = num_steps
        loss_list = []

        for step in trange(n_steps):
            loss = svi.step(data)
            loss_list.append(loss)
            #print(f'Step {step}, loss: {loss}')

        '''plt.plot(loss_list)
        plt.title("ELBO")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.show()'''

        # Extract the posterior distributions
        posterior_p = dist.Beta(pyro.param('alpha_p').item(), pyro.param('beta_p').item())
        posterior_b = dist.Beta(pyro.param('alpha_b').item(), pyro.param('beta_b').item())
        posterior_m = dist.Categorical(pyro.param('probs_m'))
        posterior_sigma = dist.HalfNormal(pyro.param('scale_sigma').item())

        # Generate samples from the posterior distributions
        current_state = [posterior_m.sample().item() for _ in range(5)]
        p_sample = posterior_p.sample().item()
        b_sample = posterior_b.sample().item()
        m_sample = posterior_m.sample().item()
        sigma_sample = posterior_sigma.sample().item()

        '''print("State sequence:", current_state)
        print("p:", p_sample)
        print("b:", b_sample)
        print("m:", m_sample)
        print("sigma:", sigma_sample)'''
        
        # Generate a volatility process
        v_sim = [sigma_sample]

        len_data = data.shape[0] if isinstance(data, torch.Tensor) else len(data)

        for t in range(1, len_data):  # simulate for 1000 time steps

            v_t = b_sample * (p_sample ** (m_sample)) * v_sim[t - 1] + 1e-6 
            v_sim.append(v_t)
        
        
        # Convert the simulated volatility process to a tensor
        epsilon = 1e-6
        v_sim_tensor = torch.tensor(v_sim) + epsilon

        # Compute the VaR at the 5% confidence level
        VaR_5 = dist.Normal(0, v_sim_tensor).icdf(torch.tensor(0.05)).numpy()
        
        # Compute the VaR at the 95% confidence level
        VaR_50 = dist.Normal(0, v_sim_tensor).icdf(torch.tensor(0.50)).numpy()

        # Compute the VaR at the 99% confidence level
        VaR_99 = dist.Normal(0, v_sim_tensor).icdf(torch.tensor(0.99)).numpy()

        # get mean, skew, kurtosis of  VaR at 5%, 95% and 99% confidence levels
        '''print("VaR at 5% confidence level:", VaR_5.mean(), VaR_5.std(), stats.skew(VaR_5), stats.kurtosis(VaR_5))
        print("VaR at 50% confidence level:", VaR_50.mean(), VaR_50.std(), stats.skew(VaR_50), stats.kurtosis(VaR_50))
        print("VaR at 99% confidence level:", VaR_99.mean(), VaR_99.std(), stats.skew(VaR_99), stats.kurtosis(VaR_99))'''

        var_5_params = [VaR_5.mean(), VaR_5.std(), stats.skew(VaR_5), stats.kurtosis(VaR_5)]
        var_50_params = [VaR_50.mean(), VaR_50.std(), stats.skew(VaR_50), stats.kurtosis(VaR_50)]
        var_99_params = [VaR_99.mean(), VaR_99.std(), stats.skew(VaR_99), stats.kurtosis(VaR_99)]

        # Compute the volatility forecast
        volatility_forecast = v_sim_tensor.std().numpy()
        
        params = [volatility_forecast*1000, p_sample, b_sample, m_sample, sigma_sample]

        # add the current_state to the list
        params.extend(current_state)

        # add the var_5_params,  var_50_params, var_99_params to the list
        params.extend(var_5_params)
        params.extend(var_50_params)
        params.extend(var_99_params)

        # set nan to num
        params = np.nan_to_num(params, nan=0.0, posinf=0.0, neginf=0.0)

        param_cols = ['msmmf_vol', 'msmmf_p', 'msmmf_b', 'msmmf_m', 'msmmf_sigma']
        param_cols.extend([f'msmmf_state_{i+1}' for i in range(5)])
        param_cols.extend(['msmmf_var_5_mean', 'msmmf_var_5_std', 'msmmf_var_5_skew', 'msmmf_var_5_kurt'])
        param_cols.extend(['msmmf_var_50_mean', 'msmmf_var_50_std', 'msmmf_var_50_skew', 'msmmf_var_50_kurt'])
        param_cols.extend(['msmmf_var_99_mean', 'msmmf_var_99_std', 'msmmf_var_99_skew', 'msmmf_var_99_kurt'])

        if len(param_cols) != len(params):
            raise ValueError("Length of param_cols and params do not match")

        return params, param_cols

    def multi_fractal_simulation(self, data):

        kbar = 3
        niter = 4
        temperature = 13.0
        stepsize = 1.0

        parameters, LL, niter, output = glo_min(kbar, data, niter, temperature, stepsize)

        # name parameters for later use:
        b_sim = parameters[0]
        m_0_sim = parameters[1]
        gamma_kbar_sim = parameters[2]
        sigma_sim = parameters[3]
        LL_sim = LL

        '''print("Parameters from glo_min for Simulated dataset: ", "\n"
            "kbar = ", kbar,"\n"
            'b = %.5f' % b_sim,"\n"
            'm_0 = %.5f' % m_0_sim,"\n"
            'gamma_kbar = %.5f' % gamma_kbar_sim,"\n"
            'sigma = %.5f' % (sigma_sim*np.sqrt(252)),"\n"
            'Likelihood = %.5f' % LL_sim,"\n"
            "niter = " , niter,"\n"
            "output = " , output,"\n")'''
        
        # Simulate data from estimated parameters:
        T = len(data)

        sim_data = self.simulatedata(b_sim, m_0_sim, gamma_kbar_sim, sigma_sim, kbar, T)

        params_msmmf = [sigma_sim*np.sqrt(252), b_sim, m_0_sim, gamma_kbar_sim]

        # reshape sim_data 
        sim_data = np.ravel(sim_data)

        return sim_data, params_msmmf

    def simulatedata(self, b, m0, gamma_kbar, sig,kbar, T):

        m0 = m0
        m1 = 2-m0
        g_s = np.zeros(kbar)
        M_s = np.zeros((kbar,T))
        g_s[0] = 1-(1-gamma_kbar)**(1/(b**(kbar-1)))
        
        for i in range(1,kbar):
            g_s[i] = 1-(1-g_s[0])**(b**(i))
            
        for j in range(kbar):
            M_s[j,:] = np.random.binomial(1,g_s[j],T)
            
        dat = np.zeros(T)
        tmp = (M_s[:,0]==1)*m1+(M_s[:,0]==0)*m0
        dat[0] = np.prod(tmp)
        
        for k in range(1,T):
            for j in range(kbar):
                if M_s[j,k]==1:
                    tmp[j] = np.random.choice([m0,m1],1,p = [0.5,0.5])
            dat[k] = np.prod(tmp)
            
        dat = np.sqrt(dat)*sig* np.random.normal(size = T)   # VOL TIME SCALING
        dat = dat.reshape(-1,1)

        return(dat)

    
    # uni and multi variate Heston Hawkes process
    
    def get_event_windows_mv_hawkes(self):

        # process the jump series
        df = self.df.copy()

        jumps = df['jump'].values
        slumps = df['slump'].values
        flat_tyre = df['flat_tyre'].values
        
        # Convert the lists to tensors
        event_jumps = torch.tensor(jumps)
        event_slumps = torch.tensor(slumps)
        event_flat_tyre = torch.tensor(flat_tyre)

        # Stack the tensors along a new dimension to create a 2D tensor
        event_array = torch.stack((event_jumps, event_slumps, event_flat_tyre), dim=1)
        
        return event_array

    def get_jump_durations_hawkes(self, delta = 300):

        # process the jump series
        df = self.df.copy()

        jumps = df['jump'].values
        slumps = df['slump'].values
        absBarSize = df['val_absBarSize'].values

        jump_slumps = jumps + slumps

        # count total number of zeros between two ones
        n_zeros_between_ones = 0
        events_times =[]

        for i in range(len(jump_slumps)):

            if jump_slumps[i] == 0:
                events_times.append(0)
                n_zeros_between_ones += 1

            else:
                events_times.append(n_zeros_between_ones)
                n_zeros_between_ones = 0
                
        # multiply by delta to get the time in seconds
        events_times = np.array(events_times)*delta
        event_size = np.where(jump_slumps == 1, absBarSize, 0)

        return events_times, event_size, jump_slumps
      
    def hawkes_model_pyro(self, data):

        if isinstance(data, list) or isinstance(data, np.ndarray):
            data = torch.tensor(data)

        # Prior distributions for the parameters
        mu = pyro.sample("mu", dist.Exponential(1.0))
        alpha = pyro.sample("alpha", dist.Exponential(1.0))
        beta = pyro.sample("beta", dist.Exponential(1.0))
        data_len = len(data) if isinstance(data, list) else data.shape[0]

        # Initialize the intensity
        lambda_ = torch.zeros(data_len)
        lambda_[0] = mu

        # Likelihood
        for t in pyro.plate("data", data_len):
            # Compute the intensity based on the history of the process
            if t > 0:
                lambda_[t] = mu + alpha * torch.sum(data[:t] * torch.exp(-beta * (t - torch.arange(t))))
            
            # Use a Poisson distribution for the likelihood
            pyro.sample(f"obs_{t}", dist.Poisson(lambda_[t]), obs=data[t])
    
    def hawkes_guide_pyro(self, data):

        # Variational parameters
        mu_loc = pyro.param("mu_loc", torch.tensor(0.1), constraint=dist.constraints.positive)
        mu_scale = pyro.param("mu_scale", torch.tensor(0.1), constraint=dist.constraints.positive)
        alpha_loc = pyro.param("alpha_loc", torch.tensor(0.1), constraint=dist.constraints.positive)
        alpha_scale = pyro.param("alpha_scale", torch.tensor(0.1), constraint=dist.constraints.positive)
        beta_loc = pyro.param("beta_loc", torch.tensor(0.1), constraint=dist.constraints.positive)
        beta_scale = pyro.param("beta_scale", torch.tensor(0.1), constraint=dist.constraints.positive)
                
        # Variational distributions for the parameters
        mu = pyro.sample("mu", dist.LogNormal(mu_loc, mu_scale))
        alpha = pyro.sample("alpha", dist.LogNormal(alpha_loc, alpha_scale))
        beta = pyro.sample("beta", dist.LogNormal(beta_loc, beta_scale))
    
    def hawkes_inference_pyro(self, data, n_steps=100, plot_loss=False):

        if isinstance(data, list):
            data = torch.tensor(data)

        # Set up the optimizer
        adam_params = {"lr": 0.001}
        optimizer = pyro.optim.Adam(adam_params)

        # Set up the inference algorithm
        svi = SVI(self.hawkes_model_pyro, self.hawkes_guide_pyro, optimizer, loss=Trace_ELBO())

        # Number of SVI steps
        n_steps = n_steps
        loss_list =[]

        # Do gradient steps
        for step in trange(n_steps):
            
            loss = svi.step(data)
            loss_list.append(loss)

        if plot_loss:
            plt.figure(figsize=(10, 3))
            plt.plot(loss_list)
            plt.title("ELBO")
            plt.xlabel("step")
            plt.ylabel("loss")
            plt.show()

        # Grab the learned variational parameters
        mu_loc = pyro.param("mu_loc").item()
        mu_scale = pyro.param("mu_scale").item()
        alpha_loc = pyro.param("alpha_loc").item()
        alpha_scale = pyro.param("alpha_scale").item()
        beta_loc = pyro.param("beta_loc").item()
        beta_scale = pyro.param("beta_scale").item()

        '''print(f"mu_loc = {mu_loc}, mu_scale = {mu_scale}")
        print(f"alpha_loc = {alpha_loc}, alpha_scale = {alpha_scale}")
        print(f"beta_loc = {beta_loc}, beta_scale = {beta_scale}")'''

        # Grab the learned variational parameters
        mu_est = pyro.param("mu_loc").item()
        alpha_est = pyro.param("alpha_loc").item()
        beta_est = pyro.param("beta_loc").item()

        '''print(f"Estimated mean of mu: {mu_est}")
        print(f"Estimated mean of alpha: {alpha_est}")
        print(f"Estimated mean of beta: {beta_est}")'''

        params_hawkes = [mu_est, alpha_est, beta_est]

        return params_hawkes

    def hawkes_nuts_pyro(self, data, num_samples=1000, warmup_steps=200):

        nuts_kernel = pyro.infer.NUTS(self.hawkes_model_pyro, #step_size =0.001, 
                                      adapt_step_size=True, adapt_mass_matrix=True)

        mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps)
        
        mcmc.run(data)
        posterior_samples = mcmc.get_samples()

        mu_mean = posterior_samples["mu"].mean().item()
        alpha_mean = posterior_samples["alpha"].mean().item()
        beta_mean = posterior_samples["beta"].mean().item()

        params_hawkes = [mu_mean, alpha_mean, beta_mean]

        return params_hawkes
    
    def simulate_hawkes_process(self, hawkes_params, other_params, T, model_name):

        """
        Simulate a Hawkes process.

        Parameters:
        mu: Baseline intensity
        alpha: Excitation parameter
        beta: Decay parameter
        T: Time horizon
        Returns:
        times: Times of events
        intensities: Intensity of the process at each event time

        """
        mu, alpha, beta = hawkes_params

        # Initialize the list of event times and intensities
        times = []
        intensities = []

        # Initialize the intensity and the current time
        intensity = mu
        t = 0

        while t < T:
            # Draw a waiting time from the exponential distribution
            dt = np.random.exponential(1 / intensity)

            # Update the current time
            t += dt

            # Draw a uniform random number
            u = np.random.uniform()

            # If the uniform random number is less than the relative intensity, an event occurs
            if u <= mu / intensity:
                times.append(t)
                intensities.append(intensity)

                # Update the intensity due to the event
                intensity += alpha

            # Otherwise, no event occurs and the intensity decays
            else:
                intensity = mu + (intensity - mu) * np.exp(-beta * dt)
        
        # process the intensities
        
        feature_intensity = self.extract_features(intensities, model_name, 10)
        
        for index, param in enumerate(other_params):
            param_key = f'{model_name}_param_{index}'
            feature_intensity[param_key] = param

        return feature_intensity

   
    # semi state markov model with negative duration dependence
    
    def semi_markov_model_no_dd(self, data):

        num_states = self.num_regimes

        if isinstance(data, np.ndarray) or isinstance(data, list):
            data = torch.tensor(data)

        # Define priors for the parameters of the negative binomial distribution
        nb_probs = pyro.sample("nb_probs", dist.Beta(torch.ones(num_states)*3, torch.ones(num_states)*3).to_event(1))
        nb_total_count = pyro.sample("nb_total_count", dist.Gamma(torch.ones(num_states)*15, torch.ones(num_states)*0.5).to_event(1))

        # Define priors for the transition probabilities
        trans_probs = pyro.sample("trans_probs", dist.Dirichlet(torch.ones(num_states, num_states)).to_event(1))

        # Define priors for the parameters of the asymmetric Laplace distribution
        loc = pyro.sample("loc", dist.Normal(0, 0.5).expand([num_states]).to_event(1))
        scale = pyro.sample("scale", dist.LogNormal(0, 1).expand([num_states]).to_event(1))
        asymmetry = pyro.sample("asymmetry", dist.Uniform(0, 1).expand([num_states]).to_event(1))

        # Initialize the state and duration
        state = pyro.sample("state_0", dist.Categorical(torch.ones(num_states) / num_states))
        duration = pyro.sample("duration_0", dist.NegativeBinomial(total_count=nb_total_count[state], probs=nb_probs[state]))

        for t in range(1, len(data)):
            # Sample a new state and duration at every time step
            state = pyro.sample(f"state_{t}", dist.Categorical(trans_probs[state]))
            duration = pyro.sample(f"duration_{t}", dist.NegativeBinomial(total_count=nb_total_count[state], probs=nb_probs[state]))

            # The observation at time t depends on the state
            pyro.sample(f"obs_{t}", dist.AsymmetricLaplace(loc[state], scale[state], asymmetry[state]), obs=data[t])

    def semi_markov_model(self, data):

        num_states = self.num_regimes

        if isinstance(data, np.ndarray) or isinstance(data, list):
            data = torch.tensor(data)

        # Define priors for the parameters of the negative binomial distribution
        nb_probs = pyro.sample("nb_probs", dist.Beta(torch.ones(num_states), torch.ones(num_states)).to_event(1))
        nb_total_count = pyro.sample("nb_total_count", dist.Gamma(torch.ones(num_states) * 10, torch.ones(num_states)).to_event(1))

        # Define priors for the transition probabilities
        trans_probs = pyro.sample("trans_probs", dist.Dirichlet(torch.ones(num_states, num_states)).to_event(1))

        # Define priors for the parameters of the asymmetric Laplace distribution
        loc = pyro.sample("loc", dist.Normal(0, 1).expand([num_states]).to_event(1))
        scale = pyro.sample("scale", dist.LogNormal(0, 1).expand([num_states]).to_event(1))
        asymmetry = pyro.sample("asymmetry", dist.Uniform(0, 1).expand([num_states]).to_event(1))

        # Initialize the state and duration
        state = pyro.sample("state_0", dist.Categorical(torch.ones(num_states) / num_states))
        duration = pyro.sample("duration_0", dist.NegativeBinomial(total_count=nb_total_count[state], probs=nb_probs[state]))

        for t in range(1, len(data)):
            # Sample a new state and duration at every time step
            # The transition probabilities are now a function of the duration in the current state
            trans_probs_duration = trans_probs / (duration + 1)
            state = pyro.sample(f"state_{t}", dist.Categorical(trans_probs_duration[state]))
            duration = pyro.sample(f"duration_{t}", dist.NegativeBinomial(total_count=nb_total_count[state], probs=nb_probs[state]))

            # The observation at time t depends on the state
            pyro.sample(f"obs_{t}", dist.AsymmetricLaplace(loc[state], scale[state], asymmetry[state]), obs=data[t])

    def semi_markov_guide(self, data):

        num_states = self.num_regimes

        if isinstance(data, np.ndarray) or isinstance(data, list):
            data = torch.tensor(data)

        # Define the variational parameters for the initial state distribution
        initial_state_probs = pyro.param("initial_state_probs", torch.ones(num_states) / num_states, constraint=dist.constraints.simplex)

        # Define the variational parameters for the parameters of the negative binomial distribution
        nb_probs_concentration1 = pyro.param("nb_probs_concentration1", torch.ones(num_states)*3, constraint=dist.constraints.positive)
        nb_probs_concentration0 = pyro.param("nb_probs_concentration0", torch.ones(num_states)*3, constraint=dist.constraints.positive)

        nb_total_count_concentration = pyro.param("nb_total_count_concentration", torch.ones(num_states)*15, constraint=dist.constraints.positive)
        nb_total_count_rate = pyro.param("nb_total_count_rate", torch.ones(num_states)*0.5, constraint=dist.constraints.positive)

        # Define the variational parameters for the transition probabilities
        trans_probs_concentration = pyro.param("trans_probs_concentration", torch.ones(num_states, num_states), constraint=dist.constraints.positive)

        # Define the variational parameters for the parameters of the asymmetric Laplace distribution
        loc_loc = pyro.param("loc_loc", torch.zeros(num_states))
        loc_scale = pyro.param("loc_scale", torch.ones(num_states)*0.5, constraint=dist.constraints.positive)

        scale_loc = pyro.param("scale_loc", torch.zeros(num_states))
        scale_scale = pyro.param("scale_scale", torch.ones(num_states), constraint=dist.constraints.positive)

        asymmetry_loc = pyro.param("asymmetry_loc", torch.ones(num_states)*0.5, constraint=dist.constraints.interval(0.01, 1))  # ensure it's strictly positive and within (0,1)
        asymmetry_scale = pyro.param("asymmetry_scale", torch.ones(num_states), constraint=dist.constraints.positive)

        # Sample the parameters of the negative binomial distribution
        nb_probs = pyro.sample("nb_probs", dist.Beta(nb_probs_concentration1, nb_probs_concentration0).to_event(1))
        nb_total_count = pyro.sample("nb_total_count", dist.Gamma(nb_total_count_concentration, nb_total_count_rate).to_event(1))

        # Sample the transition probabilities
        trans_probs = pyro.sample("trans_probs", dist.Dirichlet(trans_probs_concentration).to_event(1))

        # Sample the parameters of the asymmetric Laplace distribution
        loc = pyro.sample("loc", dist.Normal(loc_loc, loc_scale).to_event(1))
        scale = pyro.sample("scale", dist.LogNormal(scale_loc, scale_scale).to_event(1))
        asymmetry = pyro.sample("asymmetry", dist.Beta(asymmetry_loc, asymmetry_scale).to_event(1))

        # Initialize the state and duration
        state = pyro.sample("state_0", dist.Categorical(initial_state_probs))
        duration = pyro.sample("duration_0", dist.NegativeBinomial(total_count=nb_total_count[state], probs=nb_probs[state]))

        len_data = data.shape[0]

        for t in range(1, len(data)):

            # Sample a new state and duration at every time step
            state = pyro.sample(f"state_{t}", dist.Categorical(trans_probs[state]))
            duration = pyro.sample(f"duration_{t}", dist.NegativeBinomial(total_count=nb_total_count[state], probs=nb_probs[state]))

    def semi_markov_model_wei(self, data):

        num_states = self.num_regimes

        if isinstance(data, np.ndarray) or isinstance(data, list):
            data = torch.tensor(data)

        # Define priors for the parameters of the Weibull distribution
        weibull_scale = pyro.sample("weibull_scale", dist.LogNormal(torch.zeros(num_states), torch.ones(num_states)).to_event(1))
        weibull_concentration = pyro.sample("weibull_concentration", dist.LogNormal(torch.zeros(num_states), torch.ones(num_states)).to_event(1))

        # Define priors for the transition probabilities
        trans_probs = pyro.sample("trans_probs", dist.Dirichlet(torch.ones(num_states, num_states)).to_event(1))

        # Define priors for the parameters of the asymmetric Laplace distribution
        loc = pyro.sample("loc", dist.Normal(0, 1).expand([num_states]).to_event(1))
        scale = pyro.sample("scale", dist.LogNormal(0, 1).expand([num_states]).to_event(1))
        asymmetry = pyro.sample("asymmetry", dist.Uniform(0, 1).expand([num_states]).to_event(1))

        # Initialize the state and duration
        state = pyro.sample("state_0", dist.Categorical(torch.ones(num_states) / num_states))
        duration = pyro.sample("duration_0", dist.Weibull(weibull_scale[state], weibull_concentration[state]))

        for t in range(1, len(data)):
            # Sample a new state and duration at every time step
            state = pyro.sample(f"state_{t}", dist.Categorical(trans_probs[state]))
            duration = pyro.sample(f"duration_{t}", dist.Weibull(weibull_scale[state], weibull_concentration[state]))

            # The observation at time t depends on the state
            pyro.sample(f"obs_{t}", dist.AsymmetricLaplace(loc[state], scale[state], asymmetry[state]), obs=data[t])

    def semi_markov_guide_wei(self, data):

        num_states = self.num_regimes

        if isinstance(data, np.ndarray) or isinstance(data, list):
            data = torch.tensor(data)

        # Define the variational parameters for the initial state distribution
        initial_state_probs = pyro.param("initial_state_probs", torch.ones(num_states) / num_states, constraint=dist.constraints.simplex)

        # Define the variational parameters for the parameters of the Weibull distribution
        weibull_scale_loc = pyro.param("weibull_scale_loc", torch.zeros(num_states))
        weibull_scale_scale = pyro.param("weibull_scale_scale", torch.ones(num_states), constraint=dist.constraints.positive)
        weibull_concentration_loc = pyro.param("weibull_concentration_loc", torch.zeros(num_states))
        weibull_concentration_scale = pyro.param("weibull_concentration_scale", torch.ones(num_states), constraint=dist.constraints.positive)

        # Define the variational parameters for the transition probabilities
        trans_probs_concentration = pyro.param("trans_probs_concentration", torch.ones(num_states, num_states), constraint=dist.constraints.positive)

        # Define the variational parameters for the parameters of the asymmetric Laplace distribution
        loc_loc = pyro.param("loc_loc", torch.zeros(num_states))
        loc_scale = pyro.param("loc_scale", torch.ones(num_states)*0.5, constraint=dist.constraints.positive)
        scale_loc = pyro.param("scale_loc", torch.zeros(num_states))
        scale_scale = pyro.param("scale_scale", torch.ones(num_states), constraint=dist.constraints.positive)
        asymmetry_loc = pyro.param("asymmetry_loc", torch.ones(num_states)*0.5, constraint=dist.constraints.interval(0.01, 1))  # ensure it's strictly positive and within (0,1)
        asymmetry_scale = pyro.param("asymmetry_scale", torch.ones(num_states), constraint=dist.constraints.positive)

        # Sample the parameters of the Weibull distribution
        weibull_scale = pyro.sample("weibull_scale", dist.LogNormal(weibull_scale_loc, weibull_scale_scale).to_event(1))
        weibull_concentration = pyro.sample("weibull_concentration", dist.LogNormal(weibull_concentration_loc, weibull_concentration_scale).to_event(1))

        # Sample the transition probabilities
        trans_probs = pyro.sample("trans_probs", dist.Dirichlet(trans_probs_concentration).to_event(1))

        # Sample the parameters of the asymmetric Laplace distribution
        loc = pyro.sample("loc", dist.Normal(loc_loc, loc_scale).to_event(1))
        scale = pyro.sample("scale", dist.LogNormal(scale_loc, scale_scale).to_event(1))
        asymmetry = pyro.sample("asymmetry", dist.Beta(asymmetry_loc, asymmetry_scale).to_event(1))

        # Initialize the state and duration
        state = pyro.sample("state_0", dist.Categorical(initial_state_probs))
        duration = pyro.sample("duration_0", dist.Weibull(weibull_scale[state], weibull_concentration[state]))

        len_data = data.shape[0]

        for t in range(1, len(data)):

            # Sample a new state and duration at every time step
            state = pyro.sample(f"state_{t}", dist.Categorical(trans_probs[state]))
            duration = pyro.sample(f"duration_{t}", dist.Weibull(weibull_scale[state], weibull_concentration[state]))
 
    def semi_markov_simulations(self, data, chosen_model = 'weibull', other_params = None):

        if isinstance(data, np.ndarray) or isinstance(data, list):
            data = torch.tensor(data)

        if chosen_model == 'weibull':
            model = self.semi_markov_model_wei
            guide = self.semi_markov_guide_wei
        
        elif chosen_model == 'dd':
            model = self.semi_markov_model
            guide = self.semi_markov_guide
        
        else:
            model = self.semi_markov_model_no_dd  
            guide = self.semi_markov_guide

        # Generate the list of site names
        num_data_points = len(data)
        state_names = [f"state_{t}" for t in range(num_data_points)]

        # Run the guide forward and get samples
        predictive = Predictive(model, guide=guide, num_samples=100, return_sites=state_names)
        samples = predictive(data)

        # Get the state sequence
        state_sequence = torch.stack([samples[name] for name in state_names])

        # Get the last state in the sequence
        last_state = state_sequence[-1]

        # Get the probability of remaining in the current state
        current_state_prob = pyro.param("trans_probs_concentration")

        # Get the next projected state
        next_state = dist.Categorical(current_state_prob[last_state]).sample()

        # Create a Dirichlet distribution with the learned concentration parameter
        dirichlet = torch.distributions.Dirichlet(current_state_prob)

        # Sample from the Dirichlet distribution to get a set of transition probabilities
        trans_probs_sample = dirichlet.sample()

        #print(trans_probs_sample)
        
        #ic (current_state_prob)

        #print (next_state)
        #print (last_state)
        #print (state_sequence)
        
        next_state_lower, next_state_upper = self.truth_from_samples(next_state)
        last_state_lower, last_state_upper = self.truth_from_samples(last_state)
        state_sequence_median = self.truth_from_samples(state_sequence)

        #print (next_state_lower, next_state_upper)
        #print (last_state_lower, last_state_upper)
        #print (state_sequence_median)
        
        feature_val = []
        feature_cols = []
        
        for index_prob, array_prob in enumerate(trans_probs_sample):
            for index, val in enumerate(array_prob):
                key_name = f'trans_probs_{index_prob}_{index}'
                feature_val.append(val.item())
                feature_cols.append(key_name)
                
        for index_med, val_med in enumerate(state_sequence_median):
            key_name = f'state_median_{index_med}'
            feature_val.append(val_med)
            feature_cols.append(key_name)
            
        feature_temp_col = ['next_state_lower', 'next_state_upper', 'last_state_lower', 
                            'last_state_upper']
        
        feature_temp_val = [next_state_lower, next_state_upper, last_state_lower,
                            last_state_upper]
        
        for col, val in zip(feature_temp_col, feature_temp_val):
            feature_val.append(val)
            feature_cols.append(col)

        if chosen_model == 'weibull':
           
            weibull_concentration = pyro.sample("weibull_concentration", 
                                                dist.LogNormal(pyro.param("weibull_concentration_loc"), 
                                                               pyro.param("weibull_concentration_scale")).to_event(1))
            weibull_scale = pyro.sample("weibull_scale", 
                                        dist.LogNormal(pyro.param("weibull_scale_loc"), 
                                                       pyro.param("weibull_scale_scale")).to_event(1))

            # Sample the duration from the Weibull distribution
            duration = pyro.sample("duration", dist.Weibull(weibull_scale, weibull_concentration))

            # Calculate the expected duration
            expected_duration_weibull = duration.mean()
            expected_duration_weibull = expected_duration_weibull.detach().numpy()
            
            feature_cols.append('expected_duration')
            feature_val.append(expected_duration_weibull.item())

            #print (expected_duration_weibull)
        
        else:
            nb_probs_concentration1 = pyro.param("nb_probs_concentration1")
            nb_probs_concentration0 = pyro.param("nb_probs_concentration0")
            nb_total_count_concentration = pyro.param("nb_total_count_concentration")
            nb_total_count_rate = pyro.param("nb_total_count_rate")
            
            # Compute the expected duration for the Negative Binomial distribution
            expected_duration_nb = nb_total_count_concentration / nb_total_count_rate
            expected_duration_nb =  expected_duration_nb.detach().numpy()

            # Compute the expected value of the total_count parameter
            expected_nb_total_count = nb_total_count_concentration * nb_total_count_rate

            # Sample the probs parameter from the Beta distribution
            nb_probs = pyro.sample("nb_probs", dist.Beta(nb_probs_concentration1, \
                                                         nb_probs_concentration0).to_event(1))

            # Compute the expected duration for the Negative Binomial distribution
            expected_duration_nb_prob = expected_nb_total_count * (1 - nb_probs) / nb_probs
            expected_duration_nb_prob = expected_duration_nb_prob.detach().numpy()

            #print (expected_duration_nb)
            #print (expected_duration_nb_prob)
            
            for index, val in  enumerate(expected_duration_nb):
                
                key_name = f'expected_duration_{index}'
                feature_val.append(float(val))
                feature_cols.append(key_name)
                
            for index, val in  enumerate(expected_duration_nb_prob):
                
                key_name = f'expected_duration_prob_{index}'
                feature_val.append(float(val))
                feature_cols.append(key_name)
            
        # prefix feature cols with chosen model
        feature_cols = [f'smm_{chosen_model}_{col}' for col in feature_cols]
        
        # create a dict of feature cols and values
        feature_dict = dict(zip(feature_cols, feature_val))
        
        if other_params is not None:
            
            for key, val in other_params[0].items():
                
                # if val is an array with more than one value:
                if isinstance(val, (torch.Tensor)):
                    
                    # convert to a numpy array
                    val = val.detach().numpy() # type: ignore
                    
                if key == 'trans_probs_concentration':
                    pass
                
                else:
                    
                    if val.shape[0] > 1:
                        
                        for index_op, val_op in enumerate(val):
                            
                            key_op = f'smm_{chosen_model}_{key}_{index_op}'
                            feature_dict[key_op] = float(val_op)
                    
                    else:

                        feature_dict[key] = val
        
        # round off all values in feature dict
        feature_dict = {key: np.around(val, 4) for key, val in feature_dict.items()}    

        return feature_dict            
        
    def truth_from_samples(self, sample_array):

        if isinstance(sample_array, (torch.Tensor)):
            sample_array = sample_array.detach().numpy() # type: ignore

        # check if array is 2d
        if sample_array.ndim == 2:
            
            median = np.median(sample_array, axis=0)
            #print (median)
            return median[-5:]

        else:

            # get the mode of the samples
            mode = stats.mode(sample_array)

            # Find the indices of the samples that are equal to the mode
            indices = np.where(sample_array == mode.mode[0])

            # Extract the samples that are equal to the mode
            mode_samples = sample_array[indices]

            # Compute the lower and upper percentiles for the mode samples
            lower = np.percentile(mode_samples, 2.5)
            upper = np.percentile(mode_samples, 97.5)

            #print(f"The 95% confidence interval for the mode is ({lower}, {upper})")
            
            return lower, upper
          
    def semi_markov_svi_inference(self, data, num_steps = 1000, 
                                  plot_loss = True, chosen_model = 'dd'):

        # Define the optimizer
        optimizer = pyro.optim.Adam({"lr": 0.0005})

        # Define the loss function
        loss_p = Trace_ELBO()

        if chosen_model == 'weibull':
            model = self.semi_markov_model_wei
            guide = self.semi_markov_guide_wei
        
        elif chosen_model == 'dd':
            model = self.semi_markov_model
            guide = self.semi_markov_guide
        
        else:
            model = self.semi_markov_model_no_dd  
            guide = self.semi_markov_guide

        # Define the SVI algorithm
        svi = SVI(model, guide, optimizer, loss_p)

        # Run the SVI algorithm for a certain number of steps
        num_steps = num_steps

        loss_history = []

        for step in trange(num_steps):
            loss = svi.step(data)
            loss_history.append(loss)
            
        if plot_loss:
            plt.plot(loss_history)
            plt.title("ELBO")
            plt.xlabel("step")
            plt.ylabel("loss")
            plt.show()

        if isinstance(data, np.ndarray) or isinstance(data, list):
            data = torch.tensor(data)

        # Run the guide with a trace
        trace = pyro.poutine.trace(guide).get_trace(data)
        
        # Extract the learned parameters 
        learned_parameters = {name: trace.nodes[name]["value"].detach().numpy() for name in trace.param_nodes}

        params_semi_markov = [learned_parameters]

        return params_semi_markov  

    # scaling process
    
    def get_quinary_components(self, data):

        # set seed 
        np.random.seed(42)
        np.seterr(all='ignore')
        
        try:
        
            observed_data = np.array(data).reshape(-1, 1)
            num_states = self.num_quinary_components

            # Create and train Gaussian HMM

            model = hmm.GaussianHMM(n_components=num_states, covariance_type="diag")
            model.fit(observed_data)

            # Predict the optimal sequence of internal hidden state
            hidden_states = model.predict(observed_data)
            
        except:
            
            hidden_states = np.random.randint(0, 5, len(data))

        return hidden_states
        
    def get_macro_states(self, range_window):

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
        price_data = df['typical_price'].values[range_window]
        atr_ver_3 = df['atr_ver_three'].values[range_window]

        rsi_data_3 = df['rsi_ver_three'].values[range_window]
        rsi_data_2 = df['stoch_rsi_ver_two'].values[range_window]
        rsi_data_1 = df['dynamicRsi'].values[range_window]

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
        
        '''plt.scatter (range(len(price_data)), price_data, c= states, label='multi-conditions', cmap='tab10')
        plt.plot(price_data, label='price')
        plt.colorbar()
        plt.legend()
        plt.show()'''

        macro_states = np.select([c_bull, c_bull_cont, c_bear, c_bear_cont, c_chop], 
                           [0, 1, 2, 3, 4], 5)
            
        return macro_states
    
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
   
    def msm_scaling_process(self, quinary_components, theta, hurst_exp, range_window):

        # supress convergence warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        # Calculate the base scaling process using the binary components and theta
        
        base_scaling_process = np.power(quinary_components, theta)

        # get values of each column in the base scaling process
        scaling_process_list = [base_scaling_process]

        scaling_process_list_fdiff = []
        
        # for each colum in the base scaling process, apply fractional differencing
        # If the Hurst exponent has been provided, apply fractional differencing
        if hurst_exp is not None:
            
            for scaling_process in scaling_process_list:

                scaling_process_t = np.array(scaling_process)
                #scaling_process_t = self.fractional_differencing(scaling_process_t, hurst_exp)
                scaling_process_list_fdiff.append(scaling_process_t)

        # If covariates are provided, adjust the scaling process based on the covariates
        scaling_process_cov = []
        time_series = self.price

        for scaling_process in scaling_process_list_fdiff:

            scaling_process_q = self.adjust_for_covariates(scaling_process, time_series, range_window)
            scaling_process_cov.append(scaling_process_q)
           
        return scaling_process_cov[0]
    
    def fractional_differencing(self, series, hurst_exp):
        """
        Implements fractional differencing on a time series using the Hurst exponent.

        Args:
            series: Time series data
            hurst: The Hurst exponent

        Returns:
            The fractionally differenced series
        """
        
        if self.components == 'data':
            # scale the series
            series = self.pre_process_data(series)

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

    def garchVolModel_ext(self, series):

        #  NaN or inf values to 0
        series = np.nan_to_num(series)
        
        am = arch_model(series, x=None,  #power=1.0,
                                mean='HARX', lags= 1, 
                                vol='GARCH', p=1, o=0, q=1, power = 2,
                                dist='studentst', hold_back=None, rescale=True)

        volatility_model = am.fit(update_freq=5, disp=False)

        garch_vol = volatility_model.conditional_volatility

        # set nan values to mean of first 10 values
        garch_vol[np.isnan(garch_vol)] = np.mean(garch_vol[-20:])
        garch_vol = np.nan_to_num(garch_vol)

        return volatility_model, garch_vol

    def run_scaling_process(self, range_window, components= 'data', model_name_prefix = 'model_name'):

        data = self.df['typical_price'].values
        data = self.pre_process_data(data)
        data = data[range_window]
        
        price_ma_diff = self.price - self.series
        price_ma_diff = price_ma_diff[range_window]
        
        n_last = 3
        initial_theta = 2.2
        
        feature_cols = ['quinary_comp', 'hurst', 'garch_vol', 'cov', #'fracdiff_series_last',
                            'bscp', 'vol_sp',]
        
        # prefix the column names with model_name_prefix
        feature_cols = [model_name_prefix + '_' + col for col in feature_cols]

        garch_model, garch_vol = self.garchVolModel_ext(data)

        if components == 'data':
            quinary_components = self.get_quinary_components(data) # macro states
            hurst_exponent = self.estimate_hurst_exponent_wav(data)
            #fracdiff_series = self.fractional_differencing(data, hurst_exponent)

        elif components == 'data_macro':
            quinary_components = self.get_macro_states(range_window)  
            hurst_exponent = self.estimate_hurst_exponent_wav(data)
            #fracdiff_series = self.fractional_differencing(data, hurst_exponent)

        else:
            quinary_components = self.get_quinary_components(garch_vol) 
            hurst_exponent = self.estimate_hurst_exponent_wav(garch_vol)
            #fracdiff_series = self.fractional_differencing(garch_vol, hurst_exponent)
        
        try:
            # find covariates
            quinary_components = np.array([i+1 for i in quinary_components])
            base_scaling_process = np.power(quinary_components, initial_theta)
            cov_adjustment = self.adjust_for_covariates(base_scaling_process, data, range_window)

            # individual features
            garch_vol_last = np.mean(garch_vol[-n_last:])

            #fracdiff_series_last = np.mean(fracdiff_series[-n_last:])
            cov_adjustment_last = np.mean(cov_adjustment[-n_last:])
            base_scaling_process_last = np.mean(base_scaling_process[-n_last:])

            vol_scaling_process = self.msm_scaling_process(quinary_components, initial_theta, 
                                                           hurst_exponent, range_window)
            
            feature_array = np.array([quinary_components[-1], hurst_exponent, garch_vol_last, 
                                      cov_adjustment_last, #fracdiff_series_last, 
                                      base_scaling_process_last, vol_scaling_process[-1]])
            
            # create dictionary from feature array and feature cols
            feature_dict = dict(zip(feature_cols, feature_array))

        except:
            
            feature_array = np.zeros(6)
            vol_scaling_process = np.zeros(len(quinary_components))
            feature_dict = dict(zip(feature_cols, feature_array))
            
        extracted_features_dict = self.extract_features(vol_scaling_process, model_name_prefix)
        
        # combine extracted features with feature dict
        feature_dict.update(extracted_features_dict)
        feature_dict['m_scaling_target'] = price_ma_diff[-1]
        
        return feature_dict

    # multi model processes
    
    def neural_prophet(self, data):

        # Example time series data
        data = pd.DataFrame({
            'ds': pd.date_range(start='2023-01-01', periods=len(data)),
            'y': data
            #'y': np.sin(np.linspace(0, 20, 100)) + np.random.normal(0, 0.1, 100)
        })

        # Fit NeuralProphet model
        '''np_model = NeuralProphet(growth="off",
                                yearly_seasonality=False,
                                weekly_seasonality=False,
                                daily_seasonality=False,
                                n_lags=3 * 24,
                                ar_layers=[32, 32, 32, 32],
                                learning_rate=0.01)'''
                                
        np_model = Prophet()

        metrics = np_model.fit(data)

        # Make forecasts
        future = np_model.make_future_dataframe(periods= len(data))
        forecast = np_model.predict(future)
        
        # actuals data based on the first n time steps of the original dataframe
        forecast_actuals = forecast[:len(data)]
        forecast_residuals = forecast_actuals['yhat'].values - data['y'].values
        
        # Calculate residuals for historical data 
        #actuals = forecast.loc[forecast['ds'].isin(data['ds']), 'yhat'].values
        #residuals = data['y'].values - forecast.loc[forecast['ds'].isin(data['ds']), 'yhat'].values # type: ignore
        
        '''plt.scatter(range(len(forecast_actuals)), forecast_actuals, label='actuals')
        plt.scatter(range(len(data['y'])), data['y'], label='data')
        plt.legend()
        plt.title('Residuals')
        plt.show()'''
        
        # Get futures dataframe 
        forecast_future = forecast[len(data):]
        
        # Prepare dataset for BNN
        X = pd.DataFrame({'residuals': forecast_residuals, 'trend': forecast_actuals['trend'].values, 
                          'additive_terms': forecast_actuals['additive_terms'].values})
        y = data['y']

        return X
    
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
            
            try:
                dH = np.diff(np.log10(DWT_std))
            except:
                dH=[1e-8]

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
        
        
        try:
            hurstExponent = np.polyfit(np.log(lags), np.log(tau), 1)[0]
            hurstExponent = np.around(hurstExponent, 4)
        except:
            hurstExponent = 0.5
            
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
    
    def generate_fbm_series(self, length, hurst_exponent, num_realizations=1):

        n = length - 1
        covariance = np.empty((n + 1, n + 1))
        
        for i in range(n + 1):
            for j in range(n + 1):
                ij = i + j if i + j <= n else n - (i + j)
                covariance[i, j] = 0.5 * (abs(i - 1)**(2 * hurst_exponent) + abs(j - 1)**(2 * hurst_exponent) - abs(ij - 1)**(2 * hurst_exponent))
        
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues[eigenvalues < 0] = 0
        diagonal = np.diag(np.sqrt(eigenvalues))
        transform = eigenvectors @ diagonal

        white_noise = np.random.randn(n + 1, num_realizations)
        fbm_series = transform @ white_noise

        fgn_series = np.diff(fbm_series, axis=0)
        
        return fbm_series, fgn_series
 
    def kalman_filter_with_features(self, observed_price):
        # Kalman Filter parameters
        initial_state = observed_price[0]
        initial_state_uncertainty = 1
        state_noise_variance = 0.0001
        observation_noise_variance = 0.001
        n = len(observed_price)
        prices = observed_price

        # Initialize state estimates and uncertainties
        state_estimate = initial_state
        state_uncertainty = initial_state_uncertainty

        # Initialize arrays to store estimates and features
        filtered_prices = np.zeros(n)
        filtered_prices[0] = initial_state
        filtered_uncertainties = np.zeros(n)
        filtered_uncertainties[0] = initial_state_uncertainty
        kalman_gains = np.zeros(n)
        residuals = np.zeros(n)

        # Kalman Filter algorithm
        for t in range(1, n):
            # Prediction step
            predicted_state = state_estimate
            predicted_uncertainty = state_uncertainty + state_noise_variance

            # Update step
            observed_return = prices[t] - prices[t - 1]
            residual = observed_return - (predicted_state - state_estimate)
            kalman_gain = predicted_uncertainty / (predicted_uncertainty + observation_noise_variance)
            state_estimate = predicted_state + kalman_gain * residual
            state_uncertainty = (1 - kalman_gain) * predicted_uncertainty

            # Store estimates
            filtered_prices[t] = state_estimate
            filtered_uncertainties[t] = state_uncertainty
            kalman_gains[t] = kalman_gain
            residuals[t] = residual

        # Calculate z-scores
        z_scores = (prices - filtered_prices) / np.sqrt(filtered_uncertainties)
        long_signals = z_scores < -1
        short_signals = z_scores > 1

        # Kalman Filtered Returns
        kalman_filtered_returns = np.diff(filtered_prices, prepend=filtered_prices[0])

        # Return features
        return {
            "filtered_prices": filtered_prices,
            "filtered_uncertainties": filtered_uncertainties,
            "kalman_gains": kalman_gains,
            "residuals": residuals,
            "z_scores": z_scores,
            "long_signals": long_signals,
            "short_signals": short_signals,
            "kalman_filtered_returns": kalman_filtered_returns
        }
  
    def calculate_fbm_features(self, series, window=10):

        """
        Calculates features for a given fBm series.
        
        Parameters:
        - series: The fBm series.
        - window: The window size for rolling statistics.
        
        Returns:
        A dictionary containing the calculated features.
        """
        if not isinstance(series, (np.ndarray, list)) or np.ndim(series) != 1:
            raise ValueError("series must be a 1-dimensional array or list")
        
        if not np.issubdtype(np.array(series).dtype, np.number):
            raise ValueError("series must contain numerical data")
        
        # Hurst Exponent (estimated using range over standard deviation method)
        lag1 = np.log(np.arange(2, 20))
        tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in range(2, 20)]
        hurst_exponent = np.polyfit(lag1, np.log(tau), 1)[0]

        hurst_exponent_2 = self.estimate_hurst_exponent_wav(series)
        
        # Volatility Scaling (range over standard deviation)
        volatility_scaling = np.ptp(series) / np.std(series)
        
        # Autocorrelation Function
        autocorrelations = acf(series, nlags=5)  # First 5 lags
        
        # Moving Averages
        short_term_ma = np.convolve(series, np.ones(window), 'valid') / window
        long_term_ma = np.convolve(series, np.ones(2 * window), 'valid') / (2 * window)
        
        # Differences
        differences = np.diff(series)
        
        # Rolling Statistics
        rolling_mean = np.convolve(series, np.ones(window), 'valid') / window
        rolling_var = [np.var(series[i:i+window]) for i in range(len(series) - window + 1)]
        
        # Power Spectral Density
        f, pxx = signal.welch(series)
        power_spectral_density = pxx
        
        # Bundle the features in a dictionary
        features = {
            "hurst_exponent": hurst_exponent,
            "hurst_exponent_2": hurst_exponent_2,
            "volatility_scaling": volatility_scaling,
            "autocorrelations": autocorrelations,
            "short_term_ma": short_term_ma,
            "long_term_ma": long_term_ma,
            "differences": differences,
            "rolling_mean": rolling_mean,
            "rolling_var": rolling_var,
            "power_spectral_density": power_spectral_density
        }
        
        return features

    def auto_arima(self, data):

        # Use auto_arima to estimate the optimal ARIMA model parameters
        model = pm.auto_arima(data, seasonal=False, trace=False, 
                            error_action='ignore', suppress_warnings=True)
        
        #print(model.summary())

        model_residuals = model.resid()

        # Print the estimated optimal values of the AR, I, and MA parameters
        #print('AR parameters:', model.params()[1:model.order[0]+1])
        #print('I parameter:', model.order)
        #print('MA parameters:', model.params()[-model.order[2]:])
        
        return model_residuals

    def run_multi_model_processes(self, range_window):

        price_data = self.df['typical_price'].values[range_window]
        returns_data = self.df['priceGrad'].values[range_window]
        price_ma_diff = self.price - self.series
        price_ma_diff = price_ma_diff[range_window]
        
        feature_set_np_res = []

        np_residuals = self.neural_prophet(price_data)
        
        for col in np_residuals.columns:
            feature_set_np_res.append(np_residuals[col].values[-1])

        data_arima = self.auto_arima(price_data)

        data_arima[0] = np.mean(data_arima[10:25])

        hurst_exponent = self.estimate_hurst_exponent_wav(price_data)

        fBm_series, fGn_series = self.generate_fbm_series(len(price_data), hurst_exponent, num_realizations=1)
        
        fg_mean = np.mean(fGn_series[:3])
        fGn_series = np.pad(fGn_series, (1, 0), mode= 'constant', constant_values=fg_mean)
        
        # reshape fbm series
        fBm_series = fBm_series.reshape(-1)
        fBm_residuals = fBm_series - price_data 
        
        features = self.calculate_fbm_features(fBm_series, window=10)
        features_kalman = self.kalman_filter_with_features(price_data)

        # normalize the series
        #normalized_fbm = (fBm_series - fBm_series.mean()) / fBm_series.std()
        #normalized_returns = (returns_data - returns_data.mean()) / returns_data.std()
        
        '''print (features["hurst_exponent"])
        print (features["hurst_exponent_2"])
        print (features["volatility_scaling"])
        print (features["autocorrelations"])
        print (features["short_term_ma"][-1])
        print (features["long_term_ma"][-1])
        print (features["differences"][-1])
        print (features["rolling_mean"][-1])
        print (features["rolling_var"][-1])
        print (features["power_spectral_density"][-1])
        print (features_kalman["filtered_prices"][-1])
        print (price_data[-1])
        print (features_kalman['filtered_uncertainties'][-1])

        print (features_kalman['kalman_gains'][-1])
        print (features_kalman['residuals'][-1])
        print (features_kalman['z_scores'][-1])
        print (features_kalman['long_signals'][-1])
        print (features_kalman['short_signals'][-1])
        print (features_kalman['kalman_filtered_returns'][-1])
        
        plt.plot(data_arima, label='ARIMA Residuals')
        plt.plot(fBm_residuals, label='fBm Residuals')
        plt.plot(np_residuals['residuals'].values, label='Neural Prophet Residuals')
        plt.plot(returns_data, label='Returns')
        plt.legend()
        plt.show()'''

        # features out
        
        features_all = [hurst_exponent, features["volatility_scaling"], 
                        features["short_term_ma"][-1], features["long_term_ma"][-1], features["differences"][-1],
                        features["rolling_mean"][-1], features["rolling_var"][-1], features["power_spectral_density"][-1],
                        features_kalman["filtered_prices"][-1], price_data[-1], features_kalman['filtered_uncertainties'][-1],
                        features_kalman['kalman_gains'][-1], features_kalman['residuals'][-1], features_kalman['z_scores'][-1],
                        int(features_kalman['long_signals'][-1]), int(features_kalman['short_signals'][-1]), 
                        features_kalman['kalman_filtered_returns'][-1], fBm_residuals[-1], data_arima[-1], price_ma_diff[-1],
        ]
        
        features_col_names = ['mm_hurst_exp', 'mm_vol_scl', 'mm_short_ma', 'mm_long_ma', 'mm_diff',
                              'mm_rolling_mean', 'mm_rolling_var', 'mm_psd', 'mm_kalman_prices', 'mm_price',
                              'mm_kalman_uncrt', 'mm_kalman_gains', 'mm_kalman_residuals', 'mm_z_scores', 'mm_long_signals', 'mm_short_signals',
                              'mm_kalman_returns', 'mm_fbm_residuals', 'mm_arima_residuals', 'mm_target']
        
        for index, item in enumerate(features["autocorrelations"][1:]): 
            features_all.append(item)
            item_name = f"mm_acf_{index}" 
            features_col_names.append(item_name)
            #print(f"Index: {index}, Item: {item}, {item_name}")
        
        for index, item in enumerate(feature_set_np_res):
            features_all.append(item)
            item_name = f"mm_prophet_val_{index}"
            features_col_names.append(item_name)
            #print(f"Index: {index}, Item: {item}, {item_name}")
            
        features_all = [np.around(x, 4) for x in features_all]
        
        features_dict = dict(zip(features_col_names, features_all))
        
        return features_dict
        
    # run various models
        
    def get_features_df_window_multi_model_processes(self, window_size = 1500, series_name = 'typical_price'):

        range_windows = self.sliding_windows(np.array(range(len(self.df))), window_size = window_size)

        # set the torch seed
        torch.manual_seed(42)
        
        output_dict = []
        placeholder = st.empty()

        for window in tqdm(range_windows[0:5]):
    
            multi_model_processes = self.run_multi_model_processes(window)
            
            output_dict.append(multi_model_processes)
            
            df = pd.DataFrame(output_dict)
            
            with placeholder.container():
            
                st.title(df.shape[0])
                # Iterating through the columns of the DataFrame
                
                for col_name in df.columns:
                    # Creating two columns
                    col1, col2 = st.columns(2)
                    
                    # Creating a line chart for the current column
                    line_fig = px.line(df, x=df.index, y=col_name, title=f"{col_name} Line Chart")
                    
                    # Displaying the line chart in the first column
                    col1.plotly_chart(line_fig)
                    
                    # Creating a histogram chart for the current column
                    hist_fig = px.histogram(df, x=col_name, title=f"{col_name} Histogram")
                    
                    # Displaying the histogram chart in the second column
                    col2.plotly_chart(hist_fig)
        
        
        df_features_mm = pd.DataFrame(output_dict)
        file_ext_name = f'mm_features_{window_size}.csv'
        
        filename_2 = os.path.join(os.path.expanduser('~'), 'Desktop', 'code_base', 'm_r_code', file_ext_name)
        df_features_mm.to_csv(filename_2, index=False)
        
        filename_temp = os.path.join(os.path.expanduser('~'), 'Desktop', 'code_base', 'm_r_code', 'mm_features_1500_temp.csv')
        df_features_mm = pd.read_csv(filename_temp, index_col=False)
        
        # shift df_features_mm['target'] by 1 row
        df_features_mm['mm_target'] = df_features_mm['mm_target'].shift(-1)

        # drop the last row

        df_features_mm = df_features_mm.dropna()
        target = df_features_mm.pop('mm_target')
        target = target.values
        
        selected_features = lucy_markov.feature_selector_boruta_shap_xgb(
                                                df_features_mm, target, 
                                                num_trials = 100,
                                                params =  None,
                                                verbose = True, 
                                                target_name = 'price_ma_diff')
        
        st.write(selected_features)
        
    def get_features_df_window_msm_scaling(self, window_size = 1500, series_name = 'typical_price'):

        range_windows = self.sliding_windows(np.array(range(len(self.df))), window_size = window_size)
        #pyro.clear_param_store()

        feature_dicts_array = []
        placeholder = st.empty()
        
        # set the torch seed
        torch.manual_seed(42)

        for window in tqdm(range_windows[0:1500]):

            # process series with data macro
            self.components= 'data_macro'
            feature_array_dm = self.run_scaling_process(window, components= self.components,
                                                        model_name_prefix = 'm_scaling_dm')

            # process series with garch volatility
            self.components= 'garch'
            feature_array_garch = self.run_scaling_process(window, components= self.components, 
                                                            model_name_prefix = 'm_scaling_garch')

            # combine the feature_array_dm and feature_array_garch dictionaries
            feature_dict = {**feature_array_dm, **feature_array_garch}
            
            feature_dicts_array.append(feature_dict)
            
            df_feature = pd.DataFrame(feature_dicts_array)
            charts = self.create_st_plotly_charts(placeholder, df_feature)
            
            
        df_features_m_scaling = pd.DataFrame(feature_dicts_array)
        file_ext_name = f'm_scaling_features_{window_size}.csv'
        
        filename_2 = os.path.join(os.path.expanduser('~'), 'Desktop', 'code_base', 'm_r_code', file_ext_name)
        df_features_m_scaling.to_csv(filename_2, index=False)
        
        filename_temp = os.path.join(os.path.expanduser('~'), 'Desktop', 'code_base', 'm_r_code', file_ext_name)
        
        feature_imp = self.get_feature_importance(filename_temp, 'm_scaling_target')
    
    def get_features_df_window_semi_markov(self, window_size = 1500, series_name = 'typical_price'):

        data = self.df[series_name].values
        price_ma_diff = self.price - self.series
        

        data_windows = self.sliding_windows(data, window_size = window_size)
        target_windows = self.sliding_windows(range(len(price_ma_diff)), window_size = window_size)
        
        feature_dicts_array_dd = []
        feature_dicts_array_wb = []
        placeholder = st.empty()

        #pyro.clear_param_store()

        #for window in tqdm(data_windows[0:5]):
        for i, window in enumerate(tqdm(data_windows[0:1000])):
            
            # set the torch seed
            pyro.clear_param_store()
            torch.manual_seed(42)
            
            current_window = target_windows[i]
            target_data = price_ma_diff[current_window]

            #dist = self.find_my_distribution(window)
            chosen_model = ['weibull', 'dd'] #, 

            for model in chosen_model:
                
                params_semi_markov = self.semi_markov_svi_inference(window, num_steps = 50, 
                                                                    plot_loss = False, 
                                                                    chosen_model = model)

                #st.write(params_semi_markov)

                simulated_values = self.semi_markov_simulations(window, chosen_model = model,
                                                                other_params = params_semi_markov)
                
                # append target data
                simulated_values['semi_markov_target'] = np.around(target_data[-1], 4)
                
                #st.write(simulated_values)
                
                if model == 'dd':
                
                    feature_dicts_array_dd.append(simulated_values)
                    df_feature_dd = pd.DataFrame(feature_dicts_array_dd)
                    charts_dd = self.create_st_plotly_charts(placeholder, df_feature_dd)
                    st.dataframe(df_feature_dd)
                    
                else:
                    
                    feature_dicts_array_wb.append(simulated_values)
                    df_feature_wb = pd.DataFrame(feature_dicts_array_wb)
                    charts_wb = self.create_st_plotly_charts(placeholder, df_feature_wb)
                    st.dataframe(df_feature_wb)
                
                
        df_features_smm_dd = pd.DataFrame(feature_dicts_array_dd)
        df_features_smm_wb = pd.DataFrame(feature_dicts_array_wb)
        
        df_features_smm = pd.concat([df_features_smm_dd, df_features_smm_wb], axis=1)
        
        file_ext_name = f'semi_markov_model_{window_size}.csv'
        
        filename_2 = os.path.join(os.path.expanduser('~'), 'Desktop', 'code_base', 'm_r_code', file_ext_name)
        df_features_smm.to_csv(filename_2, index=False)
        
        filename_temp = os.path.join(os.path.expanduser('~'), 'Desktop', 'code_base', 'm_r_code', file_ext_name)
        
        feature_imp = self.get_feature_importance(filename_temp, 'semi_markov_target')

    def get_features_df_window_heston(self, window_size = 1500, series_name = 'typical_price'):

        returns_data = self.df[series_name].values
        price_ma_diff = self.price - self.series
        

        data_windows = self.sliding_windows(returns_data, window_size = window_size)

        pyro.set_rng_seed(42)
        pyro.clear_param_store()

        # set the torch seed
        #torch.manual_seed(42)
        # Generate the paths

        events_times, event_size, jump_slumps = self.get_jump_durations_hawkes(delta = 300)

        events_windows = self.sliding_windows(jump_slumps, window_size = window_size)
        
        features_dict = []
        
        placeholder = st.empty()

        # multi-variate hawkes process
        #mv_windows = self.get_event_windows_mv_hawkes()

        range_windows = self.sliding_windows(np.array(range(len(self.df))), window_size = window_size)

        for window, event, range_win in tqdm(zip(data_windows[0:1500], events_windows[0:1500], 
                                                    range_windows[0:1500])):
            
            pyro.clear_param_store()
            
            # other params
            
            model_name = 'heston_hawkes'
            target = price_ma_diff[range_win]
            target = np.around(target[-1], 4)
            event_last = event[-1]
            event_size_last = event_size[range_win[-1]]
            events_times_last = events_times[range_win[-1]]

            #find_my_dist_2 = self.find_my_distribution(event)

            # multi-variate hawkes process
            #mv_window = mv_windows[range_win]
            
            #params_hawkes_mv_nuts = self.hawkes_inference_nuts_mv(mv_window, num_samples=100, warmup_steps=20)
            #print (params_hawkes_mv_nuts)

            #params_hawkes_mv = self.hawkes_inference_svi_mv(mv_window, num_steps=500, plot_loss=True)
            #print (params_hawkes_mv)

            # uni-variate hawkes process

            #param_hawkes_nuts = self.hawkes_nuts_pyro(event, num_samples= 250, warmup_steps=100)
            #print (param_hawkes_nuts)

            param_hawkes = self.hawkes_inference_pyro(event, n_steps=250, plot_loss=False)
            #print (param_hawkes)

            intensities = self.simulate_hawkes_process(param_hawkes, param_hawkes, T=window_size, 
                                                        model_name=model_name)
            
            # update intensitites dict with other params
            intensities[f'{model_name}_target'] = target
            intensities[f'{model_name}_event'] = event_last
            intensities[f'{model_name}_event_size'] = event_size_last
            intensities[f'{model_name}_event_times'] = float(events_times_last)
            
            # round all values to 4 decimal places
            intensities = {k: np.around(v, 4) for k, v in intensities.items()}
            
            features_dict.append(intensities)
            
            #st.write(features_dict)
            
            df_features_hh = pd.DataFrame(features_dict)
            charts_wb = self.create_st_plotly_charts(placeholder, df_features_hh)
            
        ff_features_hh_full = pd.DataFrame(features_dict)
        
        file_ext_name = f'heston_hawkes_model_{window_size}.csv'
        
        filename_2 = os.path.join(os.path.expanduser('~'), 'Desktop', 'code_base', 'm_r_code', file_ext_name)
        ff_features_hh_full.to_csv(filename_2, index=False)
        
        filename_temp = os.path.join(os.path.expanduser('~'), 'Desktop', 'code_base', 'm_r_code', file_ext_name)
        
        feature_imp = self.get_feature_importance(filename_temp, 'heston_hawkes_target')

    def get_features_df_window_msmmf_pyro(self, window_size = 1500, series_name = 'priceGrad'):

        returns_data = self.df[series_name].values
        price_ma_diff = self.price - self.series

        data_windows = self.sliding_windows(returns_data, window_size = window_size)
        range_windows = self.sliding_windows(np.array(range(len(self.df))), window_size = window_size)
        
        inference_params = []
        simulation_params = []
        extracted_features_dicts = []
        placeholder = st.empty()
        
        #pyro.clear_param_store()

        for index, window in enumerate(tqdm(data_windows[0:1500])):

            # set the torch seed
            torch.manual_seed(42)
            #pyro.clear_param_store()
            
            current_range = range_windows[index]
            target_data = price_ma_diff[current_range]
            target = np.around(target_data[-1], 4)

            params_inference, param_cols_inference = self.multi_fractal_inference(data = window, num_steps=500)
            
            # round off values in params to 4 decimal places if value is float
            params_inference = [np.around(x, 4) if isinstance(x, float) else x for x in params_inference]
            inference_params.append(params_inference)

            data = window[:,np.newaxis]
            cols_sim = ['msmmf_sigma_sim', 'msmmf_b_sim', 'msmmf_m_sim', 'msmmf_gamma_kbar_sim']
            values_sim, params_sim = self.multi_fractal_simulation(data)
            params_sim = [np.around(x, 4) if isinstance(x, float) else x for x in params_sim]
            simulation_params.append(params_sim)

            extracted_features = self.extract_features(values_sim, model_name_prefix='msmmf')
            # add target to extracted features df
            extracted_features['msmmf_target'] = target
            extracted_features_dicts.append(extracted_features)
            
            # create dataframes from inference and simulation params
            df_inference = pd.DataFrame(inference_params, columns=param_cols_inference)
            df_simulation = pd.DataFrame(simulation_params, columns=cols_sim)
            df_extracted_features = pd.DataFrame(extracted_features_dicts)
            
            # concat all feature dfs
            df_features = pd.concat([df_inference, df_simulation, df_extracted_features], axis=1)
            
            # check for duplicate col names
            assert len(df_features.columns) == len(set(df_features.columns)), 'Duplicate column names found'
            
            # plot the charts
            charts_wb = self.create_st_plotly_charts(placeholder, df_features)
            st.dataframe(df_inference)
            
            # Collect garbage
            gc.collect()
            
        file_ext_name = f'multi_fractal_model_{window_size}.csv'
        
        filename_2 = os.path.join(os.path.expanduser('~'), 'Desktop', 'code_base', 'm_r_code', file_ext_name)
        df_features.to_csv(filename_2, index=False)
        
        filename_temp = os.path.join(os.path.expanduser('~'), 'Desktop', 'code_base', 'm_r_code', file_ext_name)
        
        feature_imp = self.get_feature_importance(filename_temp, 'msmmf_target')
                   
        return df_features

    def get_features_df_window_rsoup_pyro(self, window_size = 750, series_name = 'priceGrad'):

        # set the torch seed
        torch.manual_seed(42)

        returns_data = self.df[series_name].values
        price_ma_diff = self.price - self.series
        
        data_windows = self.sliding_windows(returns_data, window_size = window_size)
        range_windows = self.sliding_windows(np.array(range(len(self.df))), window_size = window_size)

        rsoup_params_list = []
        rsoup_ext_features_dict = []
        feature_columns = []
        placeholder = st.empty()
        
        parameters = ['mu', 'theta', 'sigma', 'last_state']
        model_name = 'rsoup'

        for parameter in parameters:
            
            for i in range(self.num_regimes):
                
                parameter_name = model_name + '_' + parameter + '_state_' + str(i)

                feature_columns.append(parameter_name)
        
        for index, window in enumerate(tqdm(data_windows[0:10])):
            
            current_range = range_windows[index]
            target_data = price_ma_diff[current_range]
            target = np.around(target_data[-1], 4)
            target_name = model_name + '_target'

            params = self.rsoup_inference(data = window, num_steps=100)
            
            # flatten list
            updated_params = [item for sublist in params for item in sublist]
            
            rsoup_features = self.simulate_ou_process(params, window, num_sim=500)
            rsoup_features = np.nan_to_num(rsoup_features, nan = 0, posinf = 0, neginf = 0)
            
            rsoup_features[target_name]= target
            
            assert len(updated_params) == len(feature_columns), \
            'length of updated_params and feature_columns do not match'

            rsoup_params_list.append(updated_params)
            rsoup_ext_features_dict.append(rsoup_features)
            
            # create dataframes
            df_params_rsoup = pd.DataFrame(rsoup_params_list, columns = feature_columns)
            df_features_sim_rsoup = pd.DataFrame(rsoup_ext_features_dict)
            
            # concat all feature dfs
            df_features_rsoup = pd.concat([df_params_rsoup, df_features_sim_rsoup], axis=1)
            
            charts = self.create_st_plotly_charts(placeholder, df_features_rsoup)
       
        # get feature importance 
        
        file_ext_name = f'rsoup_model_{window_size}.csv'
        
        filename_2 = os.path.join(os.path.expanduser('~'), 'Desktop', 'code_base', 'm_r_code', file_ext_name)
        df_features_rsoup.to_csv(filename_2, index=False)
        
        filename_temp = os.path.join(os.path.expanduser('~'), 'Desktop', 'code_base', 'm_r_code', file_ext_name)
        
        feature_imp = self.get_feature_importance(filename_temp, target_name)

        return df_features_rsoup
            
    def get_features_df_window_levy(self, window_size = 1200, series_type = 'jump'): #'flat_tyre'

        # set tf seed
        tf.random.set_seed(42)

        # flat_tyre: log normal 'loc': 4.658575675582702, 'scale': 2.9271130930085207
        # jump: 'laplace_asymmetric': {'kappa': 0.3115627893729736, 'loc': 5.949996839614894, 'scale': 2.8436031557862815
        # slump: tfp.distributions.NormalInverseGaussian
        
        df_full = self.df_o
        self.series_type = series_type
        len_df = df_full.shape[0]

        skewness= -3.0
        tailweight= 1.3
        loc = 3.4
        scale = 1.8
        learning_rate = 0.001
        num_iterations = 750

        # get sliding windows
        data_windows = self.sliding_windows(np.array(range(len_df)), window_size = window_size)
        other_params_list = []
        inference_params_list = []
        levy_ext_features_dict = []
        placeholder = st.empty()
        
        price_ma_diff = self.price - self.series

        for window in data_windows:

            df = df_full.iloc[window]
            target_series = price_ma_diff[window]

            if series_type == 'jump':
                
                # clear param store pyro
                pyro.clear_param_store()
                
                # process the jump series
                returns = df['barReturn'].values
                price = df['typical_price'].values
                time_since_last_jump = df['time_since_last_jump'].values    
                jumps = df['jump'].values
                absBarSize = df['val_absBarSize'].values
                target = target_series[-1]
                model_prefix = 'levy_jump'

                jump_durations = self.get_jump_durations(time_since_last_jump)
                mean_jump_durations = np.around(np.mean(jump_durations), 2)
                self.jump_intensity = np.around(1 / mean_jump_durations, 8)
                jump_sizes = absBarSize[jumps == 1]
                jump_counts = np.sum(jumps)

                event_counts = jump_counts
                event_sizes = jump_sizes
                
                data = price

                params = self.stoch_var_inference_levy(price, num_steps = 25)

                param_cols_levy = ['mu', 'sigma', 'lambda_opt', 'alpha', 'beta']
                params_val_levy = [params[0], params[1], params[2], params[3], params[4]]
                
                # round off values of params 
                params_val_levy = [np.around(param, 4) for param in params_val_levy] 
                
                other_params = [target, absBarSize[-1], jumps[-1], time_since_last_jump[-1], 
                                jump_durations[-1], mean_jump_durations, 
                                self.jump_intensity, event_counts]
                
                # append last 5 event_sizes to other_params
                other_params.extend(event_sizes[-5:])
                
                other_params_cols = ['target', 'absBarSize', 'jumps', 'tsl', 
                                     'durations', 'mean_durations',
                                     'intensity', 'counts']
                
                # append event size name: event_size_i for each event size
                other_params_cols.extend([f'event_size_{i}' for i in range(1, 6)])
                                
                # print params as numpy array
                mu, sigma, lambda_opt, alpha, beta = params
                num_steps= window_size

                #levy_process = self.get_levy_process(params)
                levy_process = self.simulate_levy_process(mu, sigma, lambda_opt, 
                                                          alpha, beta, num_steps)
                
                levy_process_features = self.extract_features( levy_process,
                                                              model_name_prefix=model_prefix)

                
                time_to_next_jump, jump_size = self.predict_jump(lambda_opt, alpha, beta)
                
                other_params.extend([time_to_next_jump.item(), jump_size.item()])
                other_params_cols.extend(['ttn', 'size'])
                
                # round off values of params
                other_params = [np.around(param, 4) for param in other_params]
                
                other_params_list.append(other_params)
                inference_params_list.append(params_val_levy)
                levy_ext_features_dict.append(levy_process_features)  
                
                # prefix model name to each column name
                other_params_cols = [f'{model_prefix}_{col}' for col in other_params_cols]
                param_cols_levy = [f'{model_prefix}_{col}' for col in param_cols_levy]
                
                df_other_params = pd.DataFrame(other_params_list, columns = other_params_cols)
                df_inference_params = pd.DataFrame(inference_params_list, columns = param_cols_levy)
                df_levy_ext_features = pd.DataFrame(levy_ext_features_dict)
                
                # combine all dataframes
                df_levy_features = pd.concat([df_other_params, df_inference_params, df_levy_ext_features], axis = 1)
                
                # create charts for levy process
                charts = self.create_st_plotly_charts(placeholder, df_levy_features)
                
                st.line_chart(levy_process)
                
            elif series_type == 'slump':
                
                # clear param store pyro
                pyro.clear_param_store()

                # process the jump series
                model_prefix = 'levy_slump'
                target = target_series[-1]
                
                # process the jump series
                returns = df['barReturn'].values
                price = df['typical_price'].values
                time_since_last_slump = df['time_since_last_slump'].values    
                slumps = df['slump'].values
                absBarSize = df['val_absBarSize'].values

                slump_durations = self.get_jump_durations(time_since_last_slump)
                mean_slump_durations = np.around(np.mean(slump_durations), 2)
                self.slump_intensity = np.around(1 / mean_slump_durations, 8)
                slump_sizes = absBarSize[slumps == 1]
                slump_counts = np.sum(slumps)

                self.jump_intensity = self.slump_intensity
                event_counts = slump_counts
                event_sizes = slump_sizes
                
                data = price

                params = self.stoch_var_inference_levy(price, num_steps = 25)

                param_cols_levy = ['mu', 'sigma', 'lambda_opt', 'alpha', 'beta']
                params_val_levy = [params[0], params[1], params[2], params[3], params[4]]
                
                # round off values of params 
                params_val_levy = [np.around(param, 4) for param in params_val_levy] 
                
                other_params = [target, absBarSize[-1], slumps[-1], time_since_last_slump[-1], 
                                slump_durations[-1], mean_slump_durations, 
                                self.slump_intensity, event_counts]
                
                # append last 5 event_sizes to other_params
                other_params.extend(event_sizes[-5:])
                
                other_params_cols = ['target', 'absBarSize', 'slumps', 'tsl', 
                                     'durations', 'mean_durations',
                                     'intensity', 'counts']
                
                # append event size name: event_size_i for each event size
                other_params_cols.extend([f'event_size_{i}' for i in range(1, 6)])
                                
                # print params as numpy array
                mu, sigma, lambda_opt, alpha, beta = params
                num_steps= window_size

                #levy_process = self.get_levy_process(params)
                levy_process = self.simulate_levy_process(mu, sigma, lambda_opt, 
                                                          alpha, beta, num_steps)
                
                levy_process_features = self.extract_features( levy_process,
                                                              model_name_prefix=model_prefix)

                
                time_to_next_slump, slump_size = self.predict_jump(lambda_opt, alpha, beta)
                
                other_params.extend([time_to_next_slump.item(), slump_size.item()])
                other_params_cols.extend(['ttn', 'size'])
                
                # round off values of params
                other_params = [np.around(param, 4) for param in other_params]
                
                other_params_list.append(other_params)
                inference_params_list.append(params_val_levy)
                levy_ext_features_dict.append(levy_process_features)  
                
                # prefix model name to each column name
                other_params_cols = [f'{model_prefix}_{col}' for col in other_params_cols]
                param_cols_levy = [f'{model_prefix}_{col}' for col in param_cols_levy]
                
                df_other_params = pd.DataFrame(other_params_list, columns = other_params_cols)
                df_inference_params = pd.DataFrame(inference_params_list, columns = param_cols_levy)
                df_levy_ext_features = pd.DataFrame(levy_ext_features_dict)
                
                # combine all dataframes
                df_levy_features = pd.concat([df_other_params, df_inference_params, df_levy_ext_features], axis = 1)
                
                # create charts for levy process
                charts = self.create_st_plotly_charts(placeholder, df_levy_features)
                
                st.line_chart(levy_process)

            else:
                
                # clear param store pyro
                pyro.clear_param_store()

                # process the jump series
                model_prefix = 'levy_ft'
                returns = df['barReturn'].values
                price = df['typical_price'].values
                time_since_last_flat_tyre = df['time_in_flat_tyre_zone'].values    
                flat_tyres = df['flat_tyre'].values
                absBarSize = df['val_absBarSize'].values
                target = target_series[-1]

                flat_tyre_durations = self.get_jump_durations(time_since_last_flat_tyre)
                mean_flat_tyre_durations = np.around(np.mean(flat_tyre_durations), 2)
                self.flat_tyre_intensity = np.around(1 / mean_flat_tyre_durations, 8)
                flat_tyre_sizes = absBarSize[flat_tyres == 1]
                flat_tyre_counts = np.sum(flat_tyres)

                self.jump_intensity = self.flat_tyre_intensity
                event_counts = flat_tyre_counts
                event_sizes = flat_tyre_sizes

                data = price

                params = self.stoch_var_inference_levy(price, num_steps = 25)

                param_cols_levy = ['mu', 'sigma', 'lambda_opt', 'alpha', 'beta']
                params_val_levy = [params[0], params[1], params[2], params[3], params[4]]
                
                # round off values of params 
                params_val_levy = [np.around(param, 4) for param in params_val_levy] 
                
                other_params = [target, absBarSize[-1], flat_tyres[-1], time_since_last_flat_tyre[-1], 
                                flat_tyre_durations[-1], mean_flat_tyre_durations, 
                                self.flat_tyre_intensity, event_counts]
                
                # append last 5 event_sizes to other_params
                other_params.extend(event_sizes[-5:])
                
                other_params_cols = ['target', 'absBarSize', 'flat_tyres', 'tsl_ft', 
                                     'ft_durations', 'mean_ft_durations',
                                     'ft_intensity', 'ft_counts']
                
                # append event size name: event_size_i for each event size
                other_params_cols.extend([f'event_size_{i}' for i in range(1, 6)])
                                
                # print params as numpy array
                mu, sigma, lambda_opt, alpha, beta = params
                num_steps= window_size

                #levy_process = self.get_levy_process(params)
                levy_process = self.simulate_levy_process(mu, sigma, lambda_opt, 
                                                          alpha, beta, num_steps)
                
                #levy_process = levy_process.detach().numpy()
                
                levy_process_features = self.extract_features( levy_process,
                                                              model_name_prefix=model_prefix)

                
                time_to_next_flat_tyre, flat_tyre_size = self.predict_jump(lambda_opt, alpha, beta)
                
                other_params.extend([time_to_next_flat_tyre.item(), flat_tyre_size.item()])
                other_params_cols.extend(['ttn_ft', 'ft_size'])
                
                # round off values of params
                other_params = [np.around(param, 4) for param in other_params]
                
                other_params_list.append(other_params)
                inference_params_list.append(params_val_levy)
                levy_ext_features_dict.append(levy_process_features)  
                
                # prefix model name to each column name
                other_params_cols = [f'{model_prefix}_{col}' for col in other_params_cols]
                param_cols_levy = [f'{model_prefix}_{col}' for col in param_cols_levy]
                
                df_other_params = pd.DataFrame(other_params_list, columns = other_params_cols)
                df_inference_params = pd.DataFrame(inference_params_list, columns = param_cols_levy)
                df_levy_ext_features = pd.DataFrame(levy_ext_features_dict)
                
                # combine all dataframes
                df_levy_features = pd.concat([df_other_params, df_inference_params, df_levy_ext_features], axis = 1)
                
                # create charts for levy process
                charts = self.create_st_plotly_charts(placeholder, df_levy_features)
                
    def get_features_df_window_tvoup(self, window_size = 750, series_name = 'typical_price'):
        
        # set tf seed
        tf.random.set_seed(42)

        if series_name == 'time_since_last_jump' or series_name == 'time_since_last_slump':

            time_since_last_slump = self.df['time_since_last_slump'].values
            time_since_last_jump = self.df['time_since_last_jump'].values    
            jumps = self.df['jump'].values
            absBarSize = self.df['val_absBarSize'].values

            jump_durations = self.get_jump_durations(time_since_last_jump)
            mean_jump_durations = np.around(np.mean(jump_durations), 2)
            self.jump_intensity = np.around(1 / mean_jump_durations, 8)
            jump_sizes = absBarSize[jumps == 1]
            jump_counts = np.sum(jumps)

        elif series_name == 'typical_price':
            # preprocess data
            price = self.df['typical_price'].values
            price = self.pre_process_data(price)
            returns_data = price
            returns_data_valid = self.validation_df[series_name].values
        
        else:
            returns_data = self.df[series_name].values
            returns_data_valid = self.validation_df[series_name].values
        
        price_ma_diff = self.price - self.series

        # optimization parameters
        learning_rate = 0.001
        num_iterations = 150
        num_samples = 1000
        num_burnin_steps = 100
        num_leapfrog_steps = 30
        step_size = 0.01

        tvoup_params_list = []
        tvoup_features_dict = []
        
        model_name_prefix = 'tvoup' + '_' + str(window_size)
        placeholder = st.empty()
        
        data_windows = self.sliding_windows(returns_data, window_size = window_size)
        range_windows = self.sliding_windows(range(len(self.df)), window_size = window_size)
        

        for index, window in enumerate(tqdm(data_windows[0:1000])):
            
            current_window = range_windows[index]
            target_series = price_ma_diff[current_window]
            target = target_series[-1]
            

            params, extracted_features = self.run_ou_tv_process(data = window, 
                                                           valid_data = returns_data_valid,
                                                            learning_rate = learning_rate, 
                                                            num_iterations = num_iterations)
            
            columns_params = ['b1', 'b2', 's1', 's2', 's3', 's_diff', 'k_diff', 'sigma', 
                              'loc', 'kappa', 'scale', 'sim_val', 'target']
            
            params = np.nan_to_num(params, nan = 0, posinf = 0, neginf = 0)
            
            # append target to params
            params = np.append(params, target)
            
            # create a dict of params, add model name prefix to each param col
            columns_params = [f'{model_name_prefix}_{col}' for col in columns_params]
            tvoup_params_list.append(params)
            
            tvoup_features_dict.append(extracted_features)
            
            # create dataframe from list of params
            df_params = pd.DataFrame(tvoup_params_list, columns = columns_params)
            df_extracted_features = pd.DataFrame(tvoup_features_dict)
            
            # concat the dfs
            df_tvoup_features = pd.concat([df_params, df_extracted_features], axis = 1)
            
            charts = self.create_st_plotly_charts(placeholder, df_tvoup_features)


        return df_tvoup_features
    
    def get_features_df_window_ploup(self, window_size = 750, series_name = 'typical_price'):

        # set tf seed
        tf.random.set_seed(42)

        if series_name == 'time_since_last_jump' or series_name == 'time_since_last_slump':

            time_since_last_slump = self.df['time_since_last_slump'].values
            time_since_last_jump = self.df['time_since_last_jump'].values    
            jumps = self.df['jump'].values
            absBarSize = self.df['val_absBarSize'].values

            jump_durations = self.get_jump_durations(time_since_last_jump)
            mean_jump_durations = np.around(np.mean(jump_durations), 2)
            self.jump_intensity = np.around(1 / mean_jump_durations, 8)
            jump_sizes = absBarSize[jumps == 1]
            jump_counts = np.sum(jumps)

        elif series_name == 'typical_price':
            # preprocess data
            price = self.df['typical_price'].values
            price = self.pre_process_data(price)
            returns_data = price
            returns_data_valid = self.validation_df[series_name].values

        else:
            returns_data = self.df[series_name].values
            returns_data_valid = self.validation_df[series_name].values
            
        price_ma_diff = self.price - self.series

        # optimization parameters
        learning_rate = 0.001
        num_iterations = 150
        num_samples = 1000
        num_burnin_steps = 100
        num_leapfrog_steps = 30
        step_size = 0.01
        
        plou_params_list = []
        plou_extracted_features = []
        model_name_prefix = 'ploup' + '_' + str(window_size)
        placeholder = st.empty()
        
        data_windows = self.sliding_windows(returns_data, window_size = window_size)
        range_windows = self.sliding_windows(range(len(self.df)), window_size = window_size)

        for index, window in enumerate(tqdm(data_windows[0:1000])):
            
            current_window = range_windows[index]
            target_series = price_ma_diff[current_window]
            target = target_series[-1]

            feature_list, extracted_features = self.run_plou_process(data_window = window, 
                                                                model_name_prefix = model_name_prefix)
            
            # append target to feature list
            feature_list = np.append(feature_list, target)
            
            plou_param_col = ['theta', 'sigma', 'L', 'U', 'kappa_0', 'kappa_1', 'kappa_2', 
                                'mean_reversion_speed', 'sim_val', 'target']
            
            # prefix each column with model name
            plou_param_col = [f'{model_name_prefix}_{col}' for col in plou_param_col]
            
            plou_process_features = np.nan_to_num(feature_list, nan = 0, posinf = 0, neginf = 0)
            
            plou_params_list.append(plou_process_features)
            plou_extracted_features.append(extracted_features)
            
            # create df from list of params
            df_ploup_features = pd.DataFrame(plou_params_list, columns = plou_param_col)
            df_ploup_extracted_features = pd.DataFrame(plou_extracted_features)
            
            # concatenate the dfs
            df_ploup_features = pd.concat([df_ploup_features, df_ploup_extracted_features], axis = 1)
            
            charts = self.create_st_plotly_charts(placeholder, df_ploup_features)

        return df_ploup_features
    
    def get_features_df_window_oup(self, window_size = 750, series_name = 'typical_price'):

        # set tf seed
        tf.random.set_seed(42)

        # data inputs 
        
        data = self.df[series_name].values
        price_data = self.df['typical_price'].values
        returns_data = self.df['priceGrad'].values
        returns_data_valid = self.validation_df['priceGrad'].values
        absBarSize = self.df['val_absBarSize'].values
        jumps = self.df['jump'].values
        time_since_last_slump = self.df['time_since_last_slump'].values 
        time_since_last_jump = self.df['time_since_last_jump'].values
        
        price_ma_diff = self.price - self.series
        jump_sizes = absBarSize[jumps == 1]
        jump_counts = np.sum(jumps)
        
        jump_durations = self.get_jump_durations(time_since_last_jump)
        mean_jump_durations = np.around(np.mean(jump_durations), 2)
        self.jump_intensity = np.around(1 / mean_jump_durations, 8)
        
        if series_name == 'typical_price':
            # preprocess data
            price = self.pre_process_data(data)
            returns_data_valid = self.validation_df[series_name].values
            data = price

        else:
            data = data

        # optimization parameters
        learning_rate = 0.001
        num_iterations = 150
        num_samples = 1000
        num_burnin_steps = 100
        num_leapfrog_steps = 30
        step_size = 0.01
        
        ou_params_list = []
        extracted_features_list = []
        model_prefix = 'oup' + '_' + str(window_size)
        placeholder = st.empty()
        
        data_windows = self.sliding_windows(data, window_size = window_size)
        range_windows = self.sliding_windows(range(len(self.df)), window_size = window_size)

        for index, window in enumerate(tqdm(data_windows[0:2000])):
            
            current_window = range_windows[index]
            target_series = price_ma_diff[current_window]
            target = target_series[-1]
            
            ou_param_col = ['mu', 'kappa', 'sigma', 'base_oup', 'target']
            
            # prefix each column with model name
            ou_param_col = [f'{model_prefix}_{col}' for col in ou_param_col]
             
            ou_params, extracted_features = self.run_base_ou_process(data = window, 
                                                                     model_prefix = model_prefix)
            ou_params = np.append(ou_params, target)
            
            ou_params_list.append(ou_params)
            extracted_features_list.append(extracted_features)
            
            # create df from list of params
            df_ou_params = pd.DataFrame(ou_params_list, columns = ou_param_col)
            df_extracted_features = pd.DataFrame(extracted_features_list)
            
            # concatenate the dfs
            df_ou_features = pd.concat([df_ou_params, df_extracted_features], axis = 1)
            
            charts = self.create_st_plotly_charts(placeholder, df_ou_features)
            
        return df_ou_features
            
# get paths for data: Linux
#filename_1 = os.path.join(os.path.expanduser('~'), 'Desktop', 'alpha', 'ts_upload_raw.csv')
filename_2 = os.path.join(os.path.expanduser('~'), 'Desktop', 'code_base', 'm_r_code', 'ts_non_discrete.csv.csv')

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

list_covariates = ['xbom_zone_1', 'xbom_zone_2', 'xbom_zone_3', 'xbom_zone_4', 'xbom_zone_5',
                'xbom_zone_6', 'xbom_zone_7', 'xbom_zone_8', 'xbom_zone_9', 'xbom_zone_10',
                'hurstExponent', 'atr_ver_three', 'rsi_ver_three', 'cumSumPrice']

jumps_slumps = ['val_absBarSize', 'jump', 'slump', 'flat_tyre', 'val_bodySize', 
                'time_since_last_slump', 'time_since_last_jump', 'time_in_flat_tyre_zone']

feature_set = features_bar_returns + ma_price + jumps_slumps

# get paths for data: Linux
#filename_1 = os.path.join(os.path.expanduser('~'), 'Desktop', 'alpha', 'ts_upload_raw.csv')
#filename_2 = os.path.join(os.path.expanduser('~'), 'Desktop', 'alpha', 'ts_discrete_upload.csv')

df_original = pd.read_csv(filename_2, index_col=0)

df_original = df_original[feature_set]

lucy_markov = Lucy(5, 3, 3, 3, df_original, 1000, list_covariates)

# streamlit dashboard title
page_title = "Real-Time / Live Model Data Dashboard"

st.set_page_config(page_title=page_title, page_icon=None, layout="wide", 
                   initial_sidebar_state="auto", menu_items=None)

st.title("Real-Time / Live Model Data Dashboard")
