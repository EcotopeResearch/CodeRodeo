# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 16:30:49 2021

@author: paul
"""
import time
from pytictoc import TicToc

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot
import plotly.express as px

# Plot all the data
import matplotlib.pyplot as plt


class tracker(object):
    def __init__(self, fct, noiselvl = 0.):
        self.function = fct
        
        self.noiselevel = noiselvl
        
        self.storex = []
        self.storef = []
        self.convergencex = []
        self.convergence = []
        self.count = 0
        
        self.print_call = False
        
        
    def __call__(self, x, *args):
        if self.print_call:
            t = TicToc()
            t.tic()
        self.storex.append(x)
        
        ret = self.function(x, self.noiselevel)
        self.storef += [ret]
        
        self.checkConv(x, ret)
        self.count += 1

        if self.print_call: t.toc("Finished Call " + str(self.count) +",")
        return ret
    
    def checkConv(self,x, ret):
        if self.count == 0: 
            self.convergencex += x if type(x) is list else [x]
            self.convergence +=  ret if type(ret) is list else [ret]
        elif ret < self.convergence[-1]:
            self.convergencex += x
            self.convergence += [ret] 
        
    def unpackLists(self):
        self.storex = [x[0] for x in self.storex]
        self.storef = [x[0] for x in self.storef]
        
   # def plotConvergence(self):
        
    def parallelPlot(self, title = ""):
        df = pd.DataFrame(self.storex)
        df['cost'] = self.storef
        fig = px.parallel_coordinates(df, color="cost", title = title,
                                     #color_continuous_scale=px.colors.diverging.Tealrose)
                                     color_continuous_scale=px.colors.sequential.Plotly3)
        plot(fig)  
        time.sleep(1)

# =============================================================================
#         
# =============================================================================

noiselevel = 0.1
# Let's define the objective function f(x), well do a 1D problem and a 2D problem for funs.
def objectiveFct(x, noiselevel=noiselevel):
    if type(x) is list:
        x = x[0]
    return (1.+np.sin(x**2.)**2.)*(1.+x**2.) + \
        np.random.randn() * noiselevel

def objectiveFct2(X, noiselevel=noiselevel):
    x = X[0]
    y = X[1]
    return (1.+np.sin(x**2.)**2.)*(1.+x**2) * (1.+y**2.)*(1.+np.sin(y**2.)**2.) + \
        np.random.randn() * noiselevel

def plotObj2(x, y, retfig = True):
    
    x, y = np.meshgrid(x, y)
    z = objectiveFct2([x,y])
    #https://plotly.com/python/3d-surface-plots/
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                      highlightcolor="limegreen", project_z=True))
    #fig.update_layout(title='2 Varible Objective Function')
    if retfig:
        return fig
    plot(fig)   
    time.sleep(1)

def plotObj(x, retfig = True):
    z = objectiveFct(x)
    fig = go.Figure(data=go.Scatter(x=x, y=z))
    if retfig:
        return fig
    plot(fig)    # will pop up in browser as html
    time.sleep(1)

   
# =============================================================================
# 
# =============================================================================
side = 10
xmin = -side; xmax = side
ymin = -side; ymax = side

points = 1000
x, y = np.linspace(xmin, xmax, points), np.linspace(ymin, ymax, points) # what silly syntax

plotObj(x, False)
#plotObj2(x2,y2, False)

    
from scipy.optimize import Bounds
bounds = Bounds([xmin], [xmax])
bounds2 = Bounds([xmin, ymin], [xmax, ymax])





# =============================================================================
# Start with stardard newton/gradient descent
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares
# Gauss–Newton algorithm 
# =============================================================================
from scipy import optimize
objls = tracker(objectiveFct)
x0 = 5#(np.random.rand()-0.5) * 10
resls = optimize.least_squares(objls, [x0])

objls.unpackLists()
fig = plotObj(x)
fig.add_trace(go.Scatter(x=objls.storex, y=objls.storef,
                    mode='lines+markers'))
plot(fig)

objls.parallelPlot()

# =============================================================================
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_cg.html?highlight=fmin_cg#scipy.optimize.fmin_cg
# https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cg.html#optimize-minimize-cg
# This conjugate gradient algorithm is based on that of Polak and Ribiere [1].
# Conjugate gradient methods tend to work better when:
# f has a unique global minimizing point, and no local minima or other stationary points,
# f is, at least locally, reasonably well approximated by a quadratic function of the variables,
# f is continuous and has a continuous gradient,
# fprime is not too large, e.g., has a norm less than 1000,
# The initial guess, x0, is reasonably close to f ‘s global minimizing point, xopt.
# =============================================================================
objgd = tracker(objectiveFct)
x0 = 7#(np.random.rand()-0.5) * 10
resgd = optimize.minimize(objgd, [x0], method='CG', 
                          options={'maxiter':100, 'disp':True, 'return_all': True})

objgd.unpackLists()
fig = plotObj(x)
fig.add_trace(go.Scatter(x=objgd.storex, y=objgd.storef,
                    mode='lines+markers'))
plot(fig)

# =============================================================================
# Works well but what if I have a large expensive problem ---> look at that freakin jacobian.
# =============================================================================
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html
#
# annealing says lets look all over, and store the best point, as we keep looking there's some change 
# well jump out of the minimum that we're in but this chance and how far away from the minimum will 
# decrease with time. 
# Note it could jump away from the global minimum!
# Also watch out, now we've got hyperparameters. 
# =============================================================================
from scipy.optimize import dual_annealing
global xa, fa
def printf(x, f, context):
    global xa, fa
    xa.extend(x)
    fa.extend(f)
    print(str(context))

xa = []; fa = []

obja = tracker(objectiveFct)
resa = dual_annealing(obja, bounds=list(zip([xmin], [xmax])), seed=4, 
                       maxfun = 100, 
                       initial_temp=5230.0, # If temperature is low less likely to jump
                       restart_temp_ratio=2e-05, # Restarts temperature when it's low. 
                       visit=2.62, # Visit controls how far we look out. 
                       no_local_search = True, # with no_local_search = True this is traditional Generalized Simulated Annealing we'll get to this next
                       callback=printf)
obja.unpackLists()
fig = plotObj(x)
fig.add_trace(go.Scatter(x=obja.storex, y=obja.storef,
                    mode='markers', opacity=0.5))
fig.add_trace(go.Scatter(x=xa, y=fa,
                    mode='lines+markers'))
plot(fig)

# Let's look at a different wayto look at this.
obja.parallelPlot("Generalized Simulated Annealing")

fig = px.histogram(obja.storex, title = "Generalized Simulated Annealing")
fig.show()

# =============================================================================
# Dual annealing or basin hopping. Lets check the local variable. 
# =============================================================================
xa = []; fa = []

objda = tracker(objectiveFct)
resda = dual_annealing(objda, bounds=list(zip([xmin], [xmax])), seed=4, 
                       maxfun = 100, 
                       initial_temp=5230.0, # If temperature is low less likely to jump
                       restart_temp_ratio=2e-05, # Restarts temperature when it's low. 
                       visit=2.62, # Visit controls how far we look out. 
                       no_local_search = False, # with no_local_search = True this is traditional Generalized Simulated Annealing we'll get to this next
                       callback=printf)
objda.unpackLists()
fig = plotObj(x)
fig.add_trace(go.Scatter(x=objda.storex, y=objda.storef,
                    mode='markers', opacity=0.5))
fig.add_trace(go.Scatter(x=xa, y=fa,
                    mode='lines+markers'))
plot(fig)

# Let's look at a different wayto look at this.
objda.parallelPlot("Dual Annealing")

fig = px.histogram(objda.storex, title = "Dual Annealing")
fig.show()


# =============================================================================
# Lets try it in 2d
# =============================================================================

# xa = []; fa = []

# objda = tracker(objectiveFct2)
# resda = dual_annealing(objda, bounds=list(zip([xmin, ymin], [xmax, ymax])), seed=4, 
#                        maxfun = 100, 
#                        initial_temp=5230.0, # If temperature is low less likely to jump
#                        restart_temp_ratio=2e-05, # Restarts temperature when it's low. 
#                        visit=2.62, # Visit controls how far we look out. 
#                        no_local_search = False)#, # with no_local_search = True this is traditional Generalized Simulated Annealing we'll get to this next
#                        #callback=printf)

#objda.unpackLists()
# fig = plotObj2(x,y)
# fig.add_trace(go.Scatter(x=objda.storex, y=objda.storef,
#                     mode='markers', opacity=0.5))
# fig.add_trace(go.Scatter(x=xa, y=fa,
#                     mode='lines+markers'))
# plot(fig)
# plotObj2(x,y,False)


# =============================================================================
# # http://blog.dlib.net/2017/12/a-global-optimization-algorithm-worth.html
# # http://dlib.net/global_optimization.py.html
# # https://github.com/jdb78/lipo
# =============================================================================
# import dlib
# from lipo import GlobalOptimizer
# objlipo = tracker(objectiveFct)

# search = GlobalOptimizer(
#         objlipo,
#         lower_bounds={"x": -10.0},
#         upper_bounds={"x": 10.0},
#         evaluations=[(9., objlipo(9.))],
#         maximize=False
#     )

# xa = []; fa = []
# def objFct(x,y):
#     global xa, fa
#     ret =  objectiveFct2(x,0)
#     xa += [x]
#     fa.extend(ret)
#     return ret

# # Grrrrrrrrrrr 
# x,y = dlib.find_min_global(objFct, 
#                         [xmin],  # Lower bound constraints on x0 and x1 respectively
#                         [xmax],    # Upper bound constraints on x0 and x1 respectively
#                         80)         # The number of times find_min_global() will call holder_table()
    
# fig = plotObj2(x,y)

# =============================================================================
# Bayes Opto
# https://scikit-optimize.github.io/stable/auto_examples/bayesian-optimization.html#sphx-glr-auto-examples-bayesian-optimization-py
# Good when
# f is a black box for which no closed form is known (nor its gradients);
# f is expensive to evaluate;
# and evaluations of f may be noisy.
# =============================================================================
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective, plot_evaluations, plot_gaussian_process, plot_regret

xmin = -4; xmax = 4

objbay = tracker(objectiveFct)
objbay.print_call = False

#https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html#skopt.gp_minimize
res = gp_minimize(objbay,           # the function to minimize
                  list(zip([xmin],[xmax])),             # the bounds on each dimension of x
                  acq_func="EI",      # the acquisition function"LCB" for lower confidence bound. "EI" for negative expected improvement. "PI" for negative probability of improvement.
                  kappa = 2000, #Controls how much of the variance in the predicted values should be taken into account. If set to be very high, then we are favouring exploration over exploitation and vice versa. Used when the acquisition is "LCB".
                  xi = 1000,
                  n_calls=200,      # the number of evaluations of f
                  n_random_starts=4,  # the number of random initialization points
                  random_state=4,   # the random seed
                  acq_optimizer ="lbfgs") # "sampling" or "lbfgs",
#objda.unpackLists()

for n_iter in range(5):
    # Plot true function.
    plt.subplot(5, 2, 2*n_iter+1)

    if n_iter == 0:
        show_legend = True
    else:
        show_legend = False

    ax = plot_gaussian_process(res, n_calls=4+n_iter,
                               objective=objbay,
                               noise_level=0,
                               show_legend=show_legend, show_title=False,
                               show_next_point=False, show_acq_func=False)
    ax.set_ylabel("")
    ax.set_xlabel("")
    # Plot EI(x) !!!!
    plt.subplot(5, 2, 2*n_iter+2)
    ax = plot_gaussian_process(res, n_calls=4+n_iter,
                               show_legend=show_legend, show_title=True,
                               show_mu=False, show_acq_func=True,
                               show_observations=False,
                               show_next_point=True)
    ax.set_ylabel("")
    ax.set_xlabel("")

plt.show()


#plt.figure()
#plot_evaluations(res)
#plt.figure()
plot_objective(res,  sample_source='result', n_points=100)
plt.figure()
plot_convergence(res)
objda.parallelPlot("Bayes Optomization")

plot_regret(res)
# =============================================================================
# 2D!
# =============================================================================
# points = 200
# x, y = np.linspace(xmin, xmax, points), np.linspace(ymin, ymax, points) # what silly syntax


# objbay = tracker(objectiveFct2)
# objbay.print_call = True

# #https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html#skopt.gp_minimize
# res = gp_minimize(objbay,           # the function to minimize
#                   list(zip([xmin, ymin],[xmax, ymax])),             # the bounds on each dimension of x
#                   acq_func="LCB",      # the acquisition function"LCB" for lower confidence bound. "EI" for negative expected improvement. "PI" for negative probability of improvement.
#                   kappa = 2000, #Controls how much of the variance in the predicted values should be taken into account. If set to be very high, then we are favouring exploration over exploitation and vice versa. Used when the acquisition is "LCB".
#                   n_calls=100,      # the number of evaluations of f
#                   n_random_starts=5,  # the number of random initialization points
#                   random_state=1234,   # the random seed
#                   acq_optimizer ="sampling") # "sampling" or "lbfgs",


# #plt.figure()
# plot_evaluations(res)
# #plt.figure()
# plot_objective(res,  sample_source='result', n_points=100)
# #plt.figure()
# plot_convergence(res)