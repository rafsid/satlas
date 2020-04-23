from fitter import *
np.random.seed(0)
f = Fitter()
x = np.linspace(1, 10, 3)
y = 3*x+5
# y += np.random.randn(x.shape[0])

data1 = Source(x, y, yerr=0.98*np.ones(x.shape), name='File1')
model1 = Linear(1, 1, name='Background')

x = np.linspace(10, 12, 3)
y = 3*x+5
data2 = Source(x, y, name='File2')
model2 = Linear(0, 1, name='Background')
model3 = Linear(0, 1, name='Signal')

data1.addModel(model1)
data2.addModel(model2)
data2.addModel(model3)
model2.setVary('a', False)
model2.setVary('b', False)
model2.setBounds('b', (0, 3))
# model3.setBounds('a', (-5, 2.5))
f.addSource(data1)
f.addSource(data2)

f.fitLm()
# f.createParameters()
# f.createLmParameters()
# print(f.lmpars)
# f.fit()
# import json
# print(f.pars)
# print(json.dumps(f.pars, indent=4, default=str))
