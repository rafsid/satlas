"""
Implementation of fitting routines specialised for BaseModel objects. Note that not all functions are loaded into the global satlas namespace.

.. moduleauthor:: Wouter Gins <wouter.gins@kuleuven.be>
.. moduleauthor:: Ruben de Groote <ruben.degroote@kuleuven.be>
"""
import copy
import os
import numdifftools as nd

from satlas.stats import emcee as mcmc

import lmfit as lm
from satlas import loglikelihood as llh
from satlas import tqdm
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.misc import derivative
from scipy.stats import chi2


__all__ = ['chisquare_spectroscopic_fit', 'chisquare_fit', 'calculate_analytical_uncertainty',
           'likelihood_fit', 'likelihood_walk', 'create_band', 'process_walk']
chisquare_warning_message = "The supplied dictionary for {} did not contain the necessary keys 'value' and 'uncertainty'."

###############################
# CHI SQUARE FITTING ROUTINES #
###############################

def chisquare_model(params, f, x, y, yerr, xerr=None, func=None):
    r"""Model function for chisquare fitting routines as established
    in this module.

    Parameters
    ----------
    params: lmfit.Parameters
        Instance of lmfit.Parameters object, to be assigned to the model object.
    f: :class:`.BaseModel`
        Callable instance with the correct methods for the fitmethods.
    x: array_like
        Experimental data for the x-axis.
    y: array_like
        Experimental data for the y-axis.
    yerr: array_like
        Experimental errorbars on the y-axis.

    Other parameters
    ----------------
    xerr: array_like, optional
        Given an array with the same size as *x*, the error is taken into
        account by using the method of estimated variance. Defaults to *None*.
    func: function, optional
        Given a function, the errorbars on the y-axis is calculated from
        the fitvalue using this function. Defaults to *None*.

    Returns
    -------
    NumPy array
        Array containing the residuals for the given parameters, divided by the
        uncertainty.

    Note
    ----
    If a custom function is to be used for the calculation of the residual,
    this function should be overwritten.

    The method of estimated variance calculates the chisquare in the following way:

        .. math::

            \sqrt{\chi^2} = \frac{y-f(x)}{\sqrt{\sigma_x^2+f'(x)^2\sigma_x^2}}"""
    f.params = params
    model = np.hstack(f(x))
    if func is not None:
        yerr = func(model)
    if xerr is not None:
        x = np.array(x)
        if len(x.shape) > 1:
            xerr = derivative(lambda x: np.hstack(f.seperate_response(x)), x, dx=1E-5) * xerr
        else:
            xerr = derivative(f, x, dx=1E-5) * xerr
        bottom = np.sqrt(yerr * yerr + xerr * xerr)
    else:
        bottom = yerr
    return_value = (y - model) / bottom
    appended_values = f.get_chisquare_mapping()
    if appended_values is not None:
        return_value = np.append(return_value, appended_values)
    return return_value

def chisquare_spectroscopic_fit(f, x, y, xerr=None, func=np.sqrt, verbose=True, hessian=False, method='leastsq'):
    """Use the :func:`chisquare_fit` function, automatically estimating the errors
    on the counts by the square root.

    Parameters
    ----------
    f: :class:`.BaseModel`
        Model object containing all the information about the fit;
        will be fitted to the given data.
    x: array_like
        Experimental data for the x-axis.
    y: array_like
        Experimental data for the y-axis.

    Other parameters
    ----------------
    xerr: array_like, optional
        Error bars on *x*.
    func: function, optional
        This function is applied to the model values to calculate the errorbars.
    verbose: boolean, optional
        When set to *True*, a tqdm-progressbar in the terminal is maintained.
        Defaults to *True*.
    hessian: boolean, optional
        When set to *True*, the SATLAS implementation of the Hessian uncertainty estimate
        is calculated, otherwise the LMFIT version is used. Defaults to *False*.
    method: string, optional
        Sets the method to be used by lmfit for the fitting. See lmfit for all options.

    Return
    ------
    success, message: tuple
        Boolean indicating the success of the convergence, and the message
        from the optimizer."""
    y = np.hstack(y)
    yerr = np.sqrt(y)
    yerr[np.isclose(yerr, 0.0)] = 1.0
    return chisquare_fit(f, x, y, yerr=yerr, xerr=xerr, func=func, verbose=verbose, hessian=hessian, method=method)

def chisquare_fit(f, x, y, yerr=None, xerr=None, func=None, verbose=True, hessian=False, method='leastsq'):
    """Use a non-linear least squares minimization (Levenberg-Marquardt)
    algorithm to minimize the chi-square of the fit to data *x* and
    *y* with errorbars *yerr*.

    Parameters
    ----------
    f: :class:`.BaseModel`
        Model object containing all the information about the fit;
        will be fitted to the given data.
    x: array_like
        Experimental data for the x-axis.
    y: array_like
        Experimental data for the y-axis.
    yerr: array_like
        Uncertainties on *y*.

    Other parameters
    ----------------
    xerr: array_like, optional
        Uncertainties on *x*.
    func: function, optional
        Uses the provided function on the fitvalue to calculate the
        errorbars.
    verbose: boolean, optional
        When set to *True*, a tqdm-progressbar in the terminal is maintained.
        Defaults to *True*.
    hessian: boolean, optional
        When set to *True*, the SATLAS implementation of the Hessian uncertainty estimate
        is calculated, otherwise the LMFIT version is used. Defaults to *False*.
    method: string, optional
        Sets the method to be used by lmfit for the fitting. See lmfit for all options.

    Return
    ------
    success, message: tuple
        Boolean indicating the success of the convergence, and the message
        from the optimizer."""

    params = f.params

    if verbose:
        def iter_cb(params, iter, resid, *args, **kwargs):
            progress.set_description('Chisquare fitting in progress (' + str((resid**2).sum()) + ')')
            progress.update(1)
        progress = tqdm.tqdm(desc='Chisquare fitting in progress', leave=True)
    else:
         def iter_cb(params, iter, resid, *args, **kwargs):
            pass

    result = lm.minimize(chisquare_model, params, args=(f, x, np.hstack(y), np.hstack(yerr), xerr, func), iter_cb=iter_cb, method=method)
    f.params = copy.deepcopy(result.params)
    f.chisqr_chi = copy.deepcopy(result.chisqr)

    success = False
    counter = 0
    while not success:
        result = lm.minimize(chisquare_model, result.params, args=(f, x, np.hstack(y), np.hstack(yerr), xerr, func), iter_cb=iter_cb, method=method)
        f.params = copy.deepcopy(result.params)
        success = np.isclose(result.chisqr, f.chisqr_chi)
        f.chisqr_chi = copy.deepcopy(result.chisqr)
        if counter > 10 and not success:
            break
    if verbose:
        progress.set_description('Chisquare fitting done')
        progress.close()

    f.ndof_chi = copy.deepcopy(result.nfree)
    f.redchi_chi = copy.deepcopy(result.redchi)
    f.chisq_res_par = copy.deepcopy(f.params)
    f.aic_chi = copy.deepcopy(result.aic)
    f.bic_chi = copy.deepcopy(result.bic)
    if hessian:
        if verbose:
            progress = tqdm.tqdm(desc='Starting Hessian calculation', leave=True, miniters=1)
        else:
            progress = None
        assign_hessian_estimate(lambda *args: (chisquare_model(*args)**2).sum(), f, f.chisq_res_par, x, np.hstack(y), np.hstack(yerr), xerr, func, progress=progress)
    else:
        for key in f.params.keys():
            if f.params[key].stderr is not None:
                f.params[key].stderr /= f.redchi_chi**0.5
                f.chisq_res_par[key].stderr /= f.redchi_chi**0.5

    return success, result.message

##########################################
# MAXIMUM LIKELIHOOD ESTIMATION ROUTINES #
##########################################


class PriorParameter(lm.Parameter):

    # Extended the Parameter class from LMFIT to incorporate prior boundaries.

    def __init__(self, name=None, value=None, vary=True, min=None, max=None,
                 expr=None, priormin=None, priormax=None):
        super(PriorParameter, self).__init__(name=name, value=value,
                                             vary=vary, min=min,
                                             max=max, expr=expr)
        self.priormin = priormin
        self.priormax = priormax

    def __getstate__(self):
        return_value = super(PriorParameter, self).__getstate__()
        return_value += (self.priormin, self.priormax)
        return return_value

    def __setstate__(self, state):
        state_pass = state[:-2]
        self.priormin, self.priormax = state[-2:]
        super(PriorParameter, self).__setstate__(state_pass)

sqrt2pi = np.sqrt(2*np.pi)

def likelihood_x_err(f, x, y, xerr, func):
    """Calculates the loglikelihood for a model given
    x and y values. Incorporates a common given error on
    the x-axis.

    Parameters
    ----------
    f: :class:`.BaseModel`
        Model object set to current parameters.
    x: array_like
        Experimental data for the x-axis.
    y: array_like
        Experimental data for the y-axis.
    xerr: float
        Experimental uncertainty on *x*.
    func: function
        Function taking (*y_data*, *y_model*) as input,
        and returning the loglikelihood that the data is
        drawn from a distribution characterized by the model.

    Returns
    -------
    array_like"""
    import scipy.integrate as integrate
    return (np.log([
        integrate.quad(
            lambda theta: (np.exp(func(Y, f, X+theta * XERR)[i]) * np.exp(-theta*theta/2)/sqrt2pi),
            -np.inf, np.inf)[0] for i, (X, Y, XERR) in enumerate(zip(x, y, xerr))
        ]).sum())

def likelihood_lnprob(params, f, x, y, xerr, func):
    """Calculates the logarithm of the probability that the data fits
    the model given the current parameters.

    Parameters
    ----------
    params: lmfit.Parameters object with satlas.PriorParameters
        Group of parameters for which the fit has to be evaluated.
    f: :class:`.BaseModel`
        Model object containing all the information about the fit;
        will be fitted to the given data.
    x: array_like
        Experimental data for the x-axis.
    y: array_like
        Experimental data for the y-axis.
    xerr: array_like
        Uncertainty values on *x*.
    func: function
        Function calculating the loglikelihood of y_data being drawn from
        a distribution characterized by y_model.

    Note
    ----
    The prior is first evaluated for the parameters. If this is
    not finite, the values are rejected from consideration by
    immediately returning -np.inf."""
    # Handle old-style BaseModel children by using .lnprior().
    try:
        lp = f.get_lnprior_mapping(params)
    except AttributeError:
        lp = f.lnprior()
    f.params = params
    if not np.isfinite(lp):
        return -np.inf
    res = lp + np.sum(likelihood_loglikelihood(f, x, y, xerr, func))
    return res

def likelihood_loglikelihood(f, x, y, xerr, func):
    """Given a parameters object, a Model object, experimental data
    and a loglikelihood function, calculates the loglikelihood for
    all data points.

    Parameters
    ----------
    f: :class:`.BaseModel`
        Model object containing all the information about the fit;
        will be fitted to the given data.
    x: array_like
        Experimental data for the x-axis.
    y: array_like
        Experimental data for the y-axis.
    xerr: array_like
        Experimental data on *x*.
    func: function
        Function calculating the loglikelihood of y_data being drawn from
        a distribution characterized by y_model.

    Returns
    -------
    array_like
        Array containing the loglikelihood for each seperate datapoint."""
    # If a value is given to the uncertainty on the x-values, use the adapted
    # function.
    y = np.hstack(y)
    if xerr is None or np.allclose(0, xerr):
        return_value = func(y, f, x)
    else:
        return_value = likelihood_x_err(f, x, y, xerr, func)
    return return_value

def likelihood_fit(f, x, y, xerr=None, func=llh.poisson_llh, method='nelder-mead', method_kws={}, walking=False, walk_kws={}, verbose=True, hessian=True):
    """Fits the given model to the given data using the Maximum Likelihood Estimation technique.
    The given function is used to calculate the loglikelihood. After the fit, the message
    from the optimizer is printed and returned.

    Parameters
    ----------
    f: :class:`.BaseModel`
        Model to be fitted to the data.
    x: array_like
        Experimental data for the x-axis.
    y: array_like
        Experimental data for the y-axis.

    Other parameters
    ----------------
    xerr: array_like, optional
        Estimated value for the uncertainty on the x-values.
        Set to *None* to ignore this uncertainty. Defaults to *None*.
    func: function, optional
        Used to calculate the loglikelihood that the data is drawn
        from a distribution given a model value. Should accept
        input as (y_data, y_model). Defaults to the Poisson
        loglikelihood.
    method: str, optional
        Selects the algorithm to be used by the minimizer used by LMFIT.
        See LMFIT documentation for possible values. Defaults to 'nelder-mead'.
    method_kws: dict, optional
        Dictionary containing the keywords to be passed to the
        minimizer.
    walking: boolean, optional
        If True, the uncertainty on the parameters is estimated
        by performing a random walk in parameter space and
        evaluating the loglikelihood. Defaults to False.
    walk_kws: dictionary
        Contains the keywords for the :func:`.likelihood_walk`
        function, used if walking is set to True.
    verbose: boolean, optional
        When set to *True*, a tqdm-progressbar in the terminal is maintained.
        Defaults to *True*.
    hessian: boolean, optional
        When set to *True*, the Hessian estimate of the uncertainty will be
        calculated.

    Returns
    -------
    success, message: tuple
        Boolean indicating the success of the optimization and
        the message from the optimizer."""

    def negativeloglikelihood(*args, **kwargs):
        return_val = -likelihood_lnprob(*args, **kwargs)
        return return_val

    def iter_cb(params, iter, resid, *args, **kwargs):
        if verbose:
            progress.update(1)
            progress.set_description('Likelihood fitting in progress (' + str(resid) + ')')
        else:
            pass

    y = np.hstack(y)
    params = copy.deepcopy(f.params)
    # Eliminate the estimated uncertainties
    for p in params:
        params[p].stderr = None
    if verbose:
        progress = tqdm.tqdm(leave=True, desc='Likelihood fitting in progress')

    result = lm.Minimizer(negativeloglikelihood, params, fcn_args=(f, x, y, xerr, func), iter_cb=iter_cb)
    result = result.minimize(method=method, params=params, **method_kws)
    f.params = copy.deepcopy(result.params)
    val = negativeloglikelihood(f.params, f, x, y, xerr, func)
    success = False
    counter = 0
    while not success:
        result = lm.Minimizer(negativeloglikelihood, result.params, fcn_args=(f, x, y, xerr, func), iter_cb=iter_cb)
        result.scalar_minimize(method=method, **method_kws)
        counter += 1
        f.params = copy.deepcopy(result.params)
        new_val = negativeloglikelihood(f.params, f, x, y, xerr, func)
        success = np.isclose(val, new_val)
        val = new_val
        if not success and counter > 10:
            break
    if verbose:
        progress.set_description('Likelihood fitting done')
        progress.close()
    f.ndof_mle = copy.deepcopy(result.nfree)
    f.fit_mle = copy.deepcopy(result.params)
    f.result_mle = result.message
    f.likelihood_mle = negativeloglikelihood(f.params, f, x, y, xerr, func)
    try:
        f.chisqr_mle = np.sum(-2 * likelihood_loglikelihood(f, x, y, xerr, func) + 2 * likelihood_loglikelihood(lambda i: y, x, y, xerr, func))
    except AttributeError:
        f.chisqr_mle = np.nan
    # if np.isnan(f.chisqr_mle):
    #     print('Used loglikelihood does not allow calculation of reduced chisquare for these data points! Does it contain 0 or negative numbers?')
    try:
        f.redchi_mle = f.chisqr_mle / f.ndof_mle
    except:
        f.redchi_mle = f.chisqr_mle / (len(y) - len([p for p in f.params if f.params[p].vary]))

    if hessian:
        if verbose:
            progress = tqdm.tqdm(leave=True, desc='Starting Hessian calculation')
        else:
            progress = None

        assign_hessian_estimate(likelihood_lnprob, f, f.fit_mle, x, y, xerr, func, likelihood=True, progress=progress)
        f.params = copy.deepcopy(f.fit_mle)

    if walking:
        likelihood_walk(f, x, y, xerr=xerr, func=func, **walk_kws)
    return success, result.message

############################
# UNCERTAINTY CALCULATIONS #
############################

def _parameterCostfunction(f, params, func, *args, **kwargs):
    # Creates a costfunction for the given model and arguments/data for the different methods.
    # Is used for calculation the derivative of the cost function for the different parameters.
    likelihood = kwargs.pop('likelihood', False)
    var_names = []
    vars = []
    for key in params.keys():
        if params[key].vary:
            var_names.append(key)
            vars.append(params[key].value)
    if vars == []:
        return
    groupParams = lm.Parameters()
    for key in params.keys():
        groupParams[key] = PriorParameter(key,
                                          value=params[key].value,
                                          vary=params[key].vary,
                                          expr=None,
                                          priormin=params[key].min,
                                          priormax=params[key].max)
    for key in params.keys():
        groupParams[key].expr = params[key].expr

    def listfunc(fvars):
        for val, n in zip(fvars, var_names):
            groupParams[n].value = val
        return func(groupParams, f, *args)
    return listfunc

def assign_hessian_estimate(func, f, params, *args, **kwargs):
    """Calculates the Hessian of the model at the given parameters,
    and associates uncertainty estimates based on the inverted Hessian matrix.
    Note that, for estimation for chisquare methods, the inverted matrix is
    multiplied by 2.

    Parameters
    ----------
    func: function
        Function used as cost function. Use :func:`.chisquare_model` for chisquare estimates,
        and :func:`.likelihood_lnprob` for likelihood estimates.
    f: :class:`.BaseModel`
        Model for which the estimates need to be made.
    params: Parameters
        LMFIT parameters for which the Hessian estimate needs to be made.
    args: args for func
        Arguments for the defined cost function *func*.
    likelihood: boolean
        Set to *True* if a likelihood approach is used.
    progress: progressbar
        TQDM progressbar instance to update.

    Returns
    -------
    None"""
    likelihood = kwargs.pop('likelihood', False)
    progress = kwargs.pop('progress', False)
    if progress is not None:
        progress.set_description('Parsing parameters')
        progress.update(1)

    var_names = []
    vars = []
    for key in params.keys():
        if params[key].vary:
            var_names.append(key)
            vars.append(params[key].value)
    if vars == []:
        if progress is not None:
            progress.set_description('No parameters to vary')
            progress.update(1)
            progress.close()
        return

    if progress is not None:
        progress.set_description('Creating Hessian function')
        progress.update(1)
    Hfun = nd.Hessian(_parameterCostfunction(f, params, func, *args, **kwargs))
    if progress is not None:
        progress.set_description('Calculating Hessian matrix')
        progress.update(1)
    hess_vals = Hfun(vars)
    if progress is not None:
        progress.set_description('Inverting matrix')
        progress.update(1)
    hess_vals = np.linalg.inv(hess_vals)
    if likelihood:
        hess_vals = -hess_vals
        multiplier = 1
    else:
        multiplier = 2
    if progress is not None:
        progress.set_description('Assigning uncertainties')
        progress.update(1)
    for name, hess in zip(var_names, np.diag(multiplier*hess_vals)):
        params[name].stderr = np.sqrt(hess)

    if progress is not None:
        progress.set_description('Assigning correlations')
        progress.update(1)
    for i, name in enumerate(var_names):
        params[name].correl = {}
        for j, name2 in enumerate(var_names):
            if name != name2:
                params[name].correl[name2] = hess_vals[i, j] / np.sqrt(hess_vals[i, i]*hess_vals[j, j])
    if progress is not None:
        progress.set_description('Finished Hessian calculation')
        progress.update(1)
        progress.close()

def create_band(f, x, x_data, y_data, yerr, xerr=None, method='chisquare', func_chi=None, func_llh=llh.poisson_llh, kind='prediction'):
    r"""Calculates prediction or confidence bounds at the 1 :math:`\sigma` level.
    The method used is based on the Delta Method: at the requested prediction points *x*, the bound is calculated as

    .. math::
        \sqrt{G'(\beta, x)^T H^{-1}(\beta) G'(\beta, x)}

    with G the cost function, H the Hessian matrix and :math:`\beta` the vector of parameters.
    The resulting bound needs to be subtracted and added to the value given by the model to get the confidence interval.

    For a prediction interval, the value before taking the square root is increased by 1


    Parameters
    ----------
    f: :class:`.BaseModel`
        Model for which the bound needs to be calculated.
    x: array_like
        Selection of values for which a prediction needs to be made.
    x_data: array_like
        Experimental data for the x-axis.
    y_data: array_like
        Experimental data for the y-axis.
    yerr: array_like
        Experimental uncertainty for the y-axis.
    xerr: array_like
        Experimental uncertainty for the x-axis. Defaults to *None*.
    method: {'mle', 'chisquare'}
        Selected method for which the cost function is selected.
    func_chi: function, optional
        Is passed on to the chisquare methods in order to calculate the
        experimental uncertainty from the modelvalue. Defaults to *None*,
        which uses *yerr*.
    func_llh: function
        Is passed on to the likelihood fitting method to define the
        likelihood function. Defaults to :func:`satlas.loglikelihood.poisson_llh`.
    kind: {'prediction', 'confidence'}
        Selects which type of bound is calculated.

    Returns
    -------
    bound: array_like
        Array describing the deviation from the model value as can be expected
        for the selected parameters at the 1:math:`\sigma` level."""
    method_mapping = {'mle': likelihood_lnprob,
                      'chisquare': lambda *args: (chisquare_model(*args)**2).sum()}
    if method == 'chisquare':
        args = x_data, np.hstack(y_data), np.hstack(yerr), xerr, func_chi
    else:
        args = x_data, y_data, xerr, func_llh, False
    func = method_mapping.pop(method)
    var_names = []
    vars = []
    for key in f.params.keys():
        if f.params[key].vary:
            var_names.append(key)
            vars.append(f.params[key].value)

    backup = copy.deepcopy(f.params)
    Hfun = nd.Hessian(_parameterCostfunction(f, f.params, func, *args, likelihood=method.lower()=='mle'))
    if method.lower()=='mle':
        hess_vals = -np.linalg.inv(Hfun(vars))
    else:
        hess_vals = np.linalg.inv(Hfun(vars))*2
    groupParams = lm.Parameters()
    for key in f.params.keys():
        groupParams[key] = PriorParameter(key,
                                          value=f.params[key].value,
                                          vary=f.params[key].vary,
                                          expr=f.params[key].expr,
                                          priormin=f.params[key].min,
                                          priormax=f.params[key].max)
    def listfunc(fvars):
        for val, n in zip(fvars, var_names):
            groupParams[n].value = val
        f.params = groupParams
        return f(x)
    jacob = nd.Jacobian(listfunc)
    result = np.zeros(len(x))
    for i, row in enumerate(jacob(vars)):
        result[i] = np.dot(row.T, np.dot(hess_vals, row))
    f.params = backup
    if kind.lower()=='prediction':
        return (result+1)**0.5
    else:
        return result**0.5

params_map = {'mle': 'fit_mle', 'chisquare_spectroscopic': 'chisq_res_par', 'chisquare': 'chisq_res_par'}
fit_mapping = {'mle': likelihood_fit, 'chisquare_spectroscopic': chisquare_spectroscopic_fit, 'chisquare': chisquare_fit}
attr_mapping = {'mle': 'likelihood_mle', 'chisquare_spectroscopic': 'chisqr_chi', 'chisquare': 'chisqr_chi'}

def calculate_updated_statistic(value, params_name, f, x, y, method='chisquare', func_args=tuple(), func_kwargs={}, pbar=None, orig_stat=0):
    params = copy.deepcopy(f.params)
    func = fit_mapping[method.lower()]
    attr = attr_mapping[method.lower()]

    try:
        for v, n in zip(value, params_name):
            params[n].value = v
            params[n].vary = False
    except:
        params[params_name].value = value
        params[params_name].vary = False

    f.params = params
    success = False
    counter = 0
    while not success:
        try:
            success, message = func(f, x, y, *func_args, **func_kwargs)
        except ValueError:
            f.params.pretty_print()
            print(value, f.params['Background0'].value)
            raise
        counter += 1
        if counter > 10:
            return np.nan
    return_value = getattr(f, attr) - orig_stat
    try:
        try:
            params_name = ' '.join(params_name)
        except:
            pass
        pbar.set_description(params_name + ' (' + str(value, return_value) + ')')
        pbar.update(1)
    except:
        pass
    return return_value

def _find_boundary(step, param_name, bound, f, x, y, function_kwargs={'method': 'chisquare_spectroscopic'}, verbose=True):
    method = function_kwargs['method']
    attr = attr_mapping[method.lower()]
    orig_stat = getattr(f, attr)
    value = f.params[param_name].value
    search_value = value
    if step < 0:
        direction = 'left'
        boundary = f.params[param_name].min
        boundary_test = lambda val: val < boundary
    else:
        direction = 'right'
        boundary = f.params[param_name].max
        boundary_test = lambda val: val > boundary
    if verbose:
        pbar = tqdm.tqdm(leave=True, desc=param_name + ' (searching ' + direction + ')', miniters=1)
    else:
        pbar = None
    backup = copy.deepcopy(f.params)
    backup_fit = params_map[method.lower()]
    function_kwargs['pbar'] = pbar
    function_kwargs['orig_stat'] = orig_stat
    while True:
        search_value += step
        if boundary_test(search_value):
            try:
                pbar.set_description(desc=param_name + ' (' + direction + ' limit reached)')
                pbar.update(1)
            except:
                pass
            result = boundary
            success = True
            break
        new_value = calculate_updated_statistic(search_value, param_name, f, x, y, **function_kwargs)
        try:
            pbar.set_description(desc=param_name + ' (searching ' + direction + ':  {:.3g}, change of {:.3f}, at {:.3f}%)'.format(search_value, new_value, new_value/bound*100))
            pbar.update(1)
        except:
            pass
        if new_value > bound:
            try:
                pbar.set_description(desc=param_name + ' (finding root, between {:.3g} and {:.3g})'.format(value, search_value))
                pbar.update(1)
            except:
                pass
            result, output = optimize.bisect(lambda v: calculate_updated_statistic(v, param_name, f, x, y, **function_kwargs) - bound,
                                             search_value - step, search_value,
                                             full_output=True)
            try:
                pbar.set_description(desc=param_name + ' (root found: {:.3g})'.format(result))
                pbar.update(1)
            except:
                pass
            success = output.converged
            break
    result_value = calculate_updated_statistic(result, param_name, f, x, y, **function_kwargs)
    try:
        pbar.set_description(desc=param_name + ' (root found: {:.3g}, change of {:.3f})'.format(result, result_value))
        pbar.update(1)
        pbar.close()
    except:
        pass
    f.params = copy.deepcopy(backup)
    setattr(f, attr, orig_stat)
    return result, success

def _get_state(f, method='mle'):
    if method.lower() == 'mle':
        return (copy.deepcopy(f.fit_mle), f.result_mle, f.likelihood_mle, f.chisqr_mle, f.redchi_mle)
    else:
        return (copy.deepcopy(f.chisq_res_par), f.chisqr_chi, f.ndof_chi, f.redchi_chi)

def _set_state(f, state, method='mle'):
    if method.lower() == 'mle':
        f.fit_mle = copy.deepcopy(state[0])
        f.params = copy.deepcopy(state[0])
        f.result_mle, f.likelihood_mle, f.chisqr_mle, f.redchi_mle = state[1:]
    else:
        f.chisq_res_par = copy.deepcopy(state[0])
        f.params = copy.deepcopy(state[0])
        f.chisqr_chi, f.ndof_chi, f.redchi_chi = state[1:]

def calculate_analytical_uncertainty(f, x, y, method='chisquare_spectroscopic', filter=None, fit_args=tuple(), fit_kws={}):
    """Calculates the analytical errors on the parameters, by changing the value for
    a parameter and finding the point where the chisquare for the refitted parameters
    is one greater. For MLE, an increase of 0.5 is sought. The corresponding series
    of parameters of the model is adjusted with the values found here.

    Parameters
    ----------
    f: :class:`.BaseModel`
        Instance of a model which is to be fitted.
    x: array_like
        Experimental data for the x-axis.
    y: array_like
        Experimental data for the y-axis.

    Other parameters
    ----------------
    method: {'chisquare_spectroscopic', 'chisquare', 'mle'}
        Select for which method the analytical uncertainty has to be calculated.
        Defaults to 'chisquare_spectroscopic'.
    filter: list of strings, optional
        Select only a subset of the variable parameters to calculate the uncertainty for.
        Defaults to *None* (all parameters).
    fit_kws: dictionary, optional
        Dictionary of keywords to be passed on to the selected fitting routine.

    Note
    ----
    The function edits the parameters of the given instance. Furthermore,
    it only searches for the uncertainty in the neighbourhood of the starting
    point, which is taken to be the values of the parameters as given in
    the instance. This does not do a full exploration, so the results might be
    from a local minimum!"""

    # Save the original goodness-of-fit and parameters for later use
    mapping = {'chisquare_spectroscopic': (chisquare_spectroscopic_fit, 'chisqr', 'chisq_res_par'),
               'chisquare': (chisquare_fit, 'chisqr', 'chisq_res_par'),
               'mle': (likelihood_fit, 'likelihood_mle', 'fit_mle')}
    func, attr, save_attr = mapping.pop(method.lower(), (chisquare_spectroscopic_fit, 'chisqr', 'chisq_res_par'))
    fit_kws['verbose'] = False
    fit_kws['hessian'] = False

    func(f, x, y, *fit_args, **fit_kws)

    state = _get_state(f, method=method.lower())
    orig_params = state[0]
    f.params = copy.deepcopy(state[0])

    ranges = {}

    # Select all variable parameters, generate the figure
    param_names = []
    no_params = 0
    for p in orig_params:
        if orig_params[p].vary and (filter is None or any([f in p for f in filter])):
            no_params += 1
            param_names.append(p)

    params = copy.deepcopy(f.params)
    chifunc = lambda x: chi2.cdf(x, 1) - 0.682689492 # Calculate 1 sigma boundary
    bound = optimize.root(chifunc, 1).x[0] * 0.5 if method.lower() == 'mle' else optimize.root(chifunc, 1).x[0]
    for i in range(no_params):
        # _set_state(f, state, method=method.lower())
        ranges[param_names[i]] = {}
        # Select starting point to determine error widths.
        value = orig_params[param_names[i]].value
        stderr = orig_params[param_names[i]].stderr
        stderr = stderr if stderr is not None else 0.01 * np.abs(value)
        stderr = stderr if stderr != 0 else 0.01 * np.abs(value)

        function_kws = {'method': method.lower(), 'func_args': fit_args, 'func_kwargs': fit_kws}

        result_left, success_left = _find_boundary(stderr, param_names[i], bound, f, x, y, function_kwargs=function_kws)
        result_right, success_right = _find_boundary(-stderr, param_names[i], bound, f, x, y, function_kwargs=function_kws)
        success = success_left * success_right
        ranges[param_names[i]]['left'] = result_left
        ranges[param_names[i]]['right'] = result_right

        if not success:
            print("Warning: boundary calculation did not fully succeed for " + param_names[i])
        right = np.abs(ranges[param_names[i]]['right'] - value)
        left = np.abs(ranges[param_names[i]]['left'] - value)
        ranges[param_names[i]]['uncertainty'] = max(right, left)
        ranges[param_names[i]]['value'] = orig_params[param_names[i]].value

    # First, clear all uncertainty estimates
    for p in orig_params:
        orig_params[p].stderr = None
    for param_name in ranges.keys():
        orig_params[param_name].stderr = ranges[param_name]['uncertainty']
        orig_params[param_name].value = ranges[param_name]['value']
    state = list(state)
    state[0] = copy.deepcopy(orig_params)
    state = tuple(state)
    _set_state(f, state, method=method.lower())

def likelihood_walk(f, x, y, xerr=None, func=llh.poisson_llh, nsteps=2000, walkers=20, filename=None):
    """Calculates the uncertainty on MLE-optimized parameter values
    by performing a random walk through parameter space and comparing
    the resulting loglikelihood values. For more information,
    see the emcee package. The data from the random walk is saved in a
    file, as defined with the *filename*.

    Parameters
    ----------
    f: :class:`.BaseModel`
        Model to be fitted to the data.
    x: array_like
        Experimental data for the x-axis.
    y: array_like
        Experimental data for the y-axis.

    Other parameters
    ----------------
    func: function, optional
        Used to calculate the loglikelihood that the data is drawn
        from a distribution given a model value. Should accept
        input as (y_data, y_model). Defaults to the Poisson
        loglikelihood.
    nsteps: integer, optional
        Determines how many steps each walker should take.
        Defaults to 2000 steps.
    walkers: integer, optional
        Sets the number of walkers to be used for the random walk.
        The number of walkers should never be less than twice the
        number of parameters. For more information on this, see
        the emcee documentation. Defaults to 20 walkers.
    filename: string, optional
        Filename where the random walk has to be saved. If *None*,
        the current time in seconds since January 1970 is used.

    Note
    ----
    The parameters associated with the MLE fit are not updated
    with the uncertainty as estimated by this method."""

    params = f.params
    var_names = []
    vars = []
    for key in params.keys():
        if params[key].vary:
            var_names.append(key)
            vars.append(params[key].value)
    ndim = len(vars)
    pos = mcmc.utils.sample_ball(vars, [1e-4] * len(vars), size=walkers)
    for i in range(pos.shape[1]):
        pos[:, i] = np.where(pos[:, i] < params[var_names[i]].min, params[var_names[i]].min+(1E-5), pos[:, i])
        pos[:, i] = np.where(pos[:, i] > params[var_names[i]].max, params[var_names[i]].max-(1E-5), pos[:, i])

    def lnprobList(fvars, groupParams, f, x, y, xerr, func):
        for val, n in zip(fvars, var_names):
            groupParams[n].value = val
        return likelihood_lnprob(groupParams, f, x, y, xerr, func)

    p = f.params.copy()
    groupParams = f.params.copy()
    for key in params.keys():
        groupParams[key] = PriorParameter(key,
                                          value=params[key].value,
                                          vary=params[key].vary,
                                          expr=params[key].expr,
                                          priormin=params[key].min,
                                          priormax=params[key].max)
    sampler = mcmc.EnsembleSampler(walkers, ndim, lnprobList,
                                   args=(groupParams, f, x, y, xerr, func))

    if filename is None:
        import time
        filename = '{}.h5'.format(time.time())
    else:
        filename = '.'.join(filename.split('.')[:-1]) + '.h5'

    if not os.path.isfile(filename):
        with h5py.File(filename, 'w') as store:
            dset = store.create_dataset('data', (nsteps * walkers, ndim), dtype='float', chunks=True, compression='gzip', maxshape=(None, ndim))
            dset.attrs['format'] = np.array([f.encode('utf-8') for f in var_names])
    else:
        with h5py.File(filename, 'a') as store:
            dset = store['data']
            pos = dset[-walkers:, :]

    with tqdm.tqdm(total=nsteps, desc='Walk', leave=True) as pbar:
        for i, result in enumerate(sampler.sample(pos, iterations=nsteps, storechain=False)):
            with h5py.File(filename, 'a') as store:
                dset = store['data']
                dset.resize(((i+1)*walkers, ndim))
                dset[i*walkers:(i+1)*walkers,:] = result[0]
                pbar.update(1)

    f.fit_mle = copy.deepcopy(params)
    f.params = copy.deepcopy(params)

def process_walk(model, filename, selection=(0, 100)):
    r"""Given a model and H5 file with the results of a random walk,
    the parameters varied in the walk are set to the 50% percentile,
    and the uncertainty is set to either the difference between the
    50% and 16% percentile, or between the 84% and 50%, whichever is
    the largest.

    Parameters
    ----------
    model: :class:`.BaseModel`
        Object which has a params attribute.
    filename: str
        Filename of the corresponding H5 file.

    Other parameters
    ----------------
    selection: tuple
        Sets the lower and upper boundary of the percentage of the random walk
        to take into account. Can be used to eliminate burn-in.
        Defaults to (0, 100)."""
    p = model.params.copy()
    with h5py.File(filename, 'r') as store:
        columns = store['data'].attrs['format']
        columns = [f.decode('utf-8') for f in columns]
        with tqdm.tqdm(total=len(columns)+(len(columns)**2-len(columns))/2, leave=True) as pbar:
            dataset_length = store['data'].shape[0]
            first, last = int(np.floor(dataset_length/100*selection[0])), int(np.ceil(dataset_length/100*selection[1]))
            for i, val in enumerate(columns):
                pbar.set_description(val)
                i = columns.index(val)
                x = store['data'][first:last, i]

                q = [16.0, 50.0, 84.0]
                q16, q50, q84 = np.percentile(x, q)

                value, std_dev = q50, max(q50-q16, q84-q50)
                p[val].value = value
                p[val].stderr = std_dev
                pbar.update(1)
    model.params = p.copy()
    model.fit_mle = p.copy()
    model.chisqr_mle = np.nan
    model.redchi_mle = np.nan
