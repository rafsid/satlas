import numpy as np
import scipy.optimize as optimize
import lmfit as lm

class Fitter:
    def __init__(self):
        super().__init__()
        self.sources = []
        self.pars = {}
        self.bounds = optimize.Bounds([], [])

    def addSource(self, source, name=None):
        if name is None:
            name = source.name
        self.sources.append((name, source))

    def createParameters(self):
        for name, source in self.sources:
            self.pars[name] = source.params()

    def createBounds(self):
        lower = []
        upper = []
        for source_name in self.pars.keys():
            p = self.pars[source_name]
            for model_name in p.keys():
                pars = p[model_name]
                for parameter_name in pars.keys():
                    parameter = pars[parameter_name]
                    if not parameter.vary:
                        l = parameter.value
                        u = parameter.value
                    else:
                        l = parameter.min
                        u = parameter.max
                    lower.append(l)
                    upper.append(u)
        self.bounds = optimize.Bounds(lower, upper)

    def createLmParameters(self):
        lmpars = lm.Parameters()
        for source_name in self.pars.keys():
            p = self.pars[source_name]
            for model_name in p.keys():
                pars = p[model_name]
                for parameter_name in pars.keys():
                    parameter = pars[parameter_name]
                    n = '___'.join([source_name, model_name, parameter_name])
                    lmpars.add(n, value=parameter.value, min=parameter.min, max=parameter.max, vary=parameter.vary)
        self.lmpars = lmpars

    def createParameterList(self):
        x = []
        for source_name in self.pars.keys():
            p = self.pars[source_name]
            for model_name in p.keys():
                pars = p[model_name]
                for parameter_name in pars.keys():
                    x.append(pars[parameter_name].value)
        return x

    def f(self):
        f = []
        for name, source in self.sources:
            f.append(source.f())
        return f

    def y(self):
        y = []
        for _, source in self.sources:
            y.append(source.y)
        return y

    def yerr(self):
        yerr = []
        for _, source in self.sources:
            yerr.append(source.yerr)
        return yerr

    def setParameters(self, x):
        j = 0
        for source_name in self.pars.keys():
            p = self.pars[source_name]
            for model_name in p.keys():
                pars = p[model_name]
                for parameter_name in pars.keys():
                    pars[parameter_name].value = x[j]
                    j += 1

    def setLmParameters(self, params):
        for p in params.keys():
            source_name, model_name, parameter_name = p.split('___')
            self.pars[source_name][model_name][parameter_name].value = params[p].value

    def setUncertainties(self, x):
        j = 0
        for source_name in self.pars.keys():
            p = self.pars[source_name]
            for model_name in p.keys():
                pars = p[model_name]
                for parameter_name in pars.keys():
                    pars[parameter_name].unc = x[j]
                    j += 1

    def resid(self):
        model_calcs = np.hstack(self.f())
        return (model_calcs-self.temp_y)/self.temp_yerr

    def optimizeFunc(self, x):
        self.setParameters(x)
        resid = self.resid()
        return resid

    def optimizeFuncLM(self, params):
        self.setLmParameters(params)
        resid = self.resid()
        return resid

    def fitLm(self):
        self.createParameters()
        self.createBounds()
        self.createLmParameters()
        self.temp_y = np.hstack(self.y())
        self.temp_yerr = np.hstack(self.yerr())

        self.result = lm.minimize(self.optimizeFuncLM, self.lmpars)

        print(lm.fit_report(self.result))
        del self.temp_y
        del self.temp_yerr

    def fit(self):
        self.createParameters()
        self.createBounds()
        self.createLmParameters()
        self.temp_y = np.hstack(self.y())
        self.temp_yerr = np.hstack(self.yerr())
        x0 = self.createParameterList()
        self.result = optimize.least_squares(self.optimizeFunc, x0=x0, bounds=(self.bounds.lb, self.bounds.ub))

        self.success = self.result['success']
        self.msg = self.result['message']
        print(self.result)
        # print(self.result['hess_inv'].matvec(np.ones(len(x0))))
        if self.success:
            try:
                covar = np.dual.inv(np.dot(self.result['jac'].T, self.result['jac']))
                unc = np.sqrt(np.diag(covar))
                self.setUncertainties(unc)
            except Exception as e:
                print(e)
                pass
        else:
            print(self.msg)

        self.residuals = self.result['fun']
        self.chisqr = (self.residuals**2).sum()
        self.nfree = len(self.temp_y) - len(x0)
        self.redchi = self.chisqr / self.nfree
        del self.temp_y
        del self.temp_yerr

class Source:
    def __init__(self, x, y, xerr=None, yerr=1, name=None):
        super().__init__()
        self.x = x
        self.y = y
        self.xerr = xerr
        self.yerr = yerr
        if self.yerr is 1:
            self.yerr = np.ones(self.x.shape)
        if name is not None:
            self.name = name
        self.models = []

    def addModel(self, model, name=None):
        if name is None:
            name = model.name
        self.models.append((name, model))

    def params(self):
        params = {}
        for name, model in self.models:
            params[name] = model.params
        return params

    def f(self):
        for name, model in self.models:
            try:
                f += model.f(self.x)
            except UnboundLocalError:
                f = model.f(self.x)
        return f

class Model:
    def __init__(self, name=None):
        super().__init__()
        if name is not None:
            self.name = name

    def params(self):
        return {}

    def setValues(self, params):
        raise NotImplemented

    def setBounds(self, name, bounds):
        raise NotImplemented

    def setVary(self, name, vary):
        raise NotImplemented

    def f(self, x, params):
        raise NotImplemented

class Parameter:
    def __init__(self, value=0, min=-np.inf, max=np.inf, vary=True, link=False):
        super().__init__()
        self.value = value
        self.min = min
        self.max = max
        self.vary = vary
        self.link = link
        self.unc = None

    def __repr__(self):
        return '{}+/-{} ({} max, {} min, vary={})'.format(self.value, self.unc, self.max, self.min, self.vary)

class Linear(Model):
    def __init__(self, a, b, name=None):
        super().__init__(name=name)
        self.params = {
                'a': Parameter(value=a, min=-np.inf, max=np.inf, vary=True),
                'b': Parameter(value=b, min=-np.inf, max=np.inf, vary=True),
                }

    def setValues(self, params):
        self.a = params['a'].value
        self.b = params['b'].value

    def setBounds(self, name, bounds):
        if name == 'a':
            self.params['a'].min = min(bounds)
            self.params['a'].max = max(bounds)
        elif name == 'b':
            self.params['b'].min = min(bounds)
            self.params['b'].max = max(bounds)

    def setVary(self, name, vary):
        if name == 'a':
            self.params['a'].vary = vary
        elif name == 'b':
            self.params['b'].vary = vary

    def f(self, x):
        a = self.params['a'].value
        b = self.params['b'].value
        return a*x+b
