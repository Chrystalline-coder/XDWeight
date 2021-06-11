'''
Program written by cschuermann@chemie.uni-goettingen.de

This program reads an FCO and refines the Weighting scheme by a Gradient Gescent function. The Weighting scheme is refined
against the deviation Rsq of the Q-Q-plot from a normal distribution. In Shelx, he Weighting scheme is fit to the GoF/GoFw.

'''


import re
from sys import version_info, exit
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from time import strftime
import scipy.stats as stats
import numpy as np
import sys



refine = True #Refine scaling parameters?

# initial Constants


f = 0.33333
lsm_out = 'xd_lsm.out'
weightfile = 'XDweight.txt'

#

try:
  a_ini = float(sys.argv[1])
except IndexError:
  a_ini = 0.0105
  num = None
try:
  b_ini = float(sys.argv[2])
except IndexError:
  b_ini = 0.0101
  num = None
try:
  fco = sys.argv[3]
except IndexError:
  fco = 'xd.fco'
  num = None


# Gradient Descent Parameters

try:
  epsilon = sys.argv[4]
except:
  epsilon = 0.0001 # delta a for numerical derivation
  num = None
try:
  alpha = sys.argv[5]
except:
  alpha = 0.0001   # a = a - (alpha * grad_a)
  num = None
try:
  ep = sys.argv[6]
except:
  num = None
  ep = 0.00001	 # if abs(Rsq_new - Rsq_old) <= ep -> converged!
try:
  max_iter = sys.argv[7]
except:
  num = None
  max_iter = 200	 # maximum Number of Iterationstry:

if num == None:
  print '\n  Note: XDweight variables not specified! \n   USAGE: XDweight.py a_ini[0.01] b_ini[0.01] .fco[xd.fco] epsilon[0.0001] alpha[0.0001] converged[0.00001] max_iter[200] \n'

# Definitions

R = {}  # list the reflection data, format: value: list[F_sq(calc),F_sq(obs),F_sq(sigma),sin(t/l),err]
E = {}  # list the reflection error
w = {}
g1 = {}
g2 ={}


regexp = re.compile('\s+([ \-]\d+)\s+(\d+)\s+(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)')

def checkpython():
    print '     Python version is %s.%s.%s \n' % (version_info[0], version_info[1], version_info[2])
    if int(version_info[0]) <= 2 and int(version_info[1]) < 7:
        print 'Your python version is %s.%s.%s Please Use version 2.7.10 or greather, but not >= 3.0.x! \n older versions cal lead to Plotting errors - nevertheless, the weighting scheme will be printed\n' \
              % (version_info[0], version_info[1], version_info[2])
        exit(-1)
    if  int(version_info[2]) < 10:
        plot = False
        print '    NOTE: Graphical Outpus is only supported for Python 2.7.10 or higher.\n'
    else:
        plot = True
    return plot

def get_npar(lsm_out):
    try:
        with open(lsm_out) as rfile:
            lines = rfile.readlines()
            for line in range(len(lines)):
                if lines[line].startswith(' SCALE   ') and lines[line+1].startswith('--------------------------------------------------') and lines[line-1].startswith(' OVTHP  '):
                    np = int(lines[line].split(' ')[-1])
                elif lines[line].startswith('  Rank of Q ='):
                    np -= int(lines[line].split()[4])
        return np
    except IOError:
        np = float(raw_input('Not able to get Numper of Parameter from xd_lsm.out.\nNumber of Parameter: '))
        return np

def import_fco(fco):
    #    FCO = raw_input('FCO filename? [xd.fco]') or 'xd.fco'
    with open(fco) as rfile:
        lines = rfile.readlines()
    for line in lines:
        matchobj = re.match(regexp, line)
        if matchobj:
            h, k, l, Fc, Fo, Fs, res = [int(i) if n < 3 else float(i) for n, i in enumerate(matchobj.groups())]
            R[(h, k, l)] = (Fc, Fo, Fs, res)
    return R

def cal_Rsq(a, b, f, R):
    for h, k, l in R:
        Fc, Fo, Fs = R[(h, k, l)][0], R[(h, k, l)][1], R[(h, k, l)][2]
        p = (f * Fo) + ((1 - f) * Fc)
        w = 1 / (Fs ** 2 + (a * p) ** 2 + (b * p))
        E[(h, k, l)] = (Fc - Fo) * np.sqrt(w)
    (x, y) = stats.probplot(E.values(), dist="norm", fit=False)
    r = 1 - sum((y-x)**2)/sum((y-np.mean(y))**2)
    return r

def cal_Goof(a, b, f, R):
    for h, k, l in R:
        Fc, Fo, Fs = R[(h, k, l)][0], R[(h, k, l)][1], R[(h, k, l)][2]
        p = (f * Fo) + ((1 - f) * Fc)
        w = 1 / (Fs ** 2 + (a * p) ** 2 + (b * p))
        g1[h, k, l] = w*((Fo - Fc) ** 2)
        g2[h, k, l] = ((Fo - Fc)/Fs) ** 2
    gofw = np.sqrt(np.sum(g1.values())/(len(R)-npar))
    gof = np.sqrt(np.sum(g2.values())/(len(R)-npar))
    return gof, gofw

def cal_def_Goof(a, b, f, R):
    for h, k, l in R:
        Fc, Fo, Fs = R[(h, k, l)][0], R[(h, k, l)][1], R[(h, k, l)][2]
        p = (f * Fo) + ((1 - f) * Fc)
        w = 1 / (Fs ** 2 + (a * p) ** 2 + (b * p))
        g1[h, k, l] = w*((Fo - Fc) ** 2)
        g2[h, k, l] = ((Fo - Fc)/Fs) ** 2
    def_gofw = np.sqrt(np.sum(g1.values())/(len(R)-npar))-1
    return def_gofw

def fitGOOF(a_ini, b_ini, f, R, epsilon, alpha, ep, max_iter):
    converged = False
    iter = 0
    a = a_ini
    b = b_ini
    # calculate old Rsq
    dev_Goof_old = cal_def_Goof(a, b, f, R)
    # iteration loop
    while not converged:
        # calculate gradeient
        grad_a = (cal_def_Goof(a - epsilon, b, f, R) - cal_def_Goof(a + epsilon, b, f, R)) / (2 * epsilon)
        grad_b = (cal_def_Goof(a, b - epsilon, f, R) - cal_def_Goof(a, b + epsilon, f, R)) / (2 * epsilon)
        #
        if dev_Goof_old > 0:
            grad_a *= -1
            grad_b *= -1
        # update a and b
        a = a - (alpha *  grad_a)
        b = b - (alpha *  grad_b)
        if b < 0:
            b = -0.1*b
        # calculate Goof
        GoF, GoFw = cal_Goof(a, b, f, R)
        dev_Goof_new = cal_def_Goof(a, b, f, R)
        # print output
        print "iter %s | a = %.4f, b = %.4f | GoF = %.4f | GoFw = %.4f " % (iter, a, b, GoF, GoFw)
        #damping
        alpha = alpha * 0.9
        # test if old and new Rsq diverge more than the minimal value ep
        if abs(dev_Goof_new) <= ep or b == 0:
            print '\n Congratulations: Gradient Descent for an optimal fit of GOFw did converge after %s iterations !!!\n Please check the GoFw to be sensible \n\n Suggested XD weighting scheme: \n WEIGHT   %.4f %.4f 0.0 0.0 0.0 %.4f' % (iter, a, b, f)
            ofile = open(weightfile, 'a')
            now = (strftime("%y/%m/%d-%H:%M:%S"))
            ofile.write('%s GoF = %.4f GoFw = %.4f WEIGHT   %.4f %.4f 0.0 0.0 0.0 %.4f\n' % (now, GoF, GoFw, a, b, f))
            ofile.close()
            converged = True
        dev_Goof_old = dev_Goof_new  # update Rsq
        iter += 1  # update iter
        if iter == max_iter:
            print 'Max_iterations exceeded. If XDweight did not converge, you should adjust the Gradient Descent parameters.\n  Note: XDweight variables can be specified! \n   USAGE: XDweight.py a_ini[0.01] b_ini[0.01] .fco[xd.fco] epsilon[0.0001] alpha[0.0001] converged[0.00001] max_iter[200] \n'
            converged = True
    return a, b


def gradientDescent(a_ini, b_ini, f, R, epsilon, alpha, ep, max_iter):
    converged = False
    iter = 0
    a = a_ini
    b = b_ini
    # calculate old Rsq
    Rsq_old = cal_Rsq(a, b, f, R)
    # iteration loop
    while not converged:
        # calculate gradeient
        grad_a = (cal_Rsq(a - epsilon, b, f, R) - cal_Rsq(a + epsilon, b, f, R)) / (2 * epsilon)
        grad_b = (cal_Rsq(a, b - epsilon, f, R) - cal_Rsq(a, b + epsilon, f, R)) / (2 * epsilon)
        # update a and b
        a = a - (alpha * grad_a)
        b = b - (alpha * grad_b)
        if b < 0:
            b = -0.1*b
        # calculate new Rsq
        Rsq_new = cal_Rsq(a, b, f, R)
        # calculate Goof
        GoF, GoFw = cal_Goof(a, b, f, R)
        # print output
        print "iter %s | a = %.4f, b = %.4f | Rsq = %.4f | GoF = %.4f | GoFw = %.4f " % (iter, a, b, Rsq_new, GoF, GoFw)
        # test if old and new Rsq diverge more than the minimal value ep
        if abs(Rsq_new - Rsq_old) <= ep or b == 0:
            print '\n Congratulations: Gradient Descent for an optimal fit of the Q-Q-plot did converge after %s iterations !!!\n Please check the GoFw to be sensible \n\n Suggested XD weighting scheme: \n WEIGHT   %.4f %.4f 0.0 0.0 0.0 %.4f' % (iter, a, b, f)
            ofile = open(weightfile, 'a')
            now = (strftime("%y/%m/%d-%H:%M:%S"))
            ofile.write('%s Rsq = %.4f GoF = %.4f GoFw = %.4f WEIGHT   %.4f %.4f 0.0 0.0 0.0 %.4f\n' % (now, Rsq_new, GoF, GoFw, a, b, f))
            ofile.close()
            converged = True
        Rsq_old = Rsq_new  # update Rsq
        iter += 1  # update iter
        if iter == max_iter:
            print 'Max_iterations exceeded. If XDweight did not converge, you should adjust the Gradient Descent parameters.\n  Note: XDweight variables can be specified! \n   USAGE: XDweight.py a_ini[0.01] b_ini[0.01] .fco[xd.fco] epsilon[0.0001] alpha[0.0001] converged[0.00001] max_iter[200] \n'
            converged = True
    return a, b

def do_Plot(a, b, f, R):
    for h, k, l in R:
        Fc, Fo, Fs = R[(h, k, l)][0], R[(h, k, l)][1], R[(h, k, l)][2]
        p = (f * Fo) + ((1 - f) * Fc)
        w = 1 / (Fs ** 2 + (a * p) ** 2 + (b * p))
        E[(h, k, l)] = (Fc - Fo) * np.sqrt(w)
    (x, y) = stats.probplot(E.values(), dist="norm", fit=False)
    Rsq = 1 - sum((y-x)**2)/sum((y-np.mean(y))**2)
    # PLOT
    fsize = 11
    lsize = 10
    plt.xlim((-4,4))
    plt.ylim((-4,4))
    plt.plot(x,y,'ro',markersize=4, markevery=0.01)
    plt.axhline(color='k', lw=1)
    plt.axvline(color='k', lw=1)
    plt.plot((-4,4),(-4,4), color='b')
    plt.title('Q-Q-Plot of Dataset with suggested weighting factors', fontsize=fsize)
    plt.annotate('Scaling factors a = %.4f, b = %.4f, f = %.4f and Rsq = %.4f ' % (a, b, f, Rsq),fontsize=lsize, xycoords='axes fraction', xy=(0.5,0.958), va="center", ha="center",bbox=dict(boxstyle="round", fc="w"))
    plt.ylabel('Experimental DR')
    plt.xlabel('Expected DR')
    plt.savefig('XDweight_out.png', dpi=300, bbox_inches='tight')

# Main Section

Check = checkpython()

R = import_fco(fco)  # imports .fco

npar = get_npar(lsm_out)
print 'number of Parameter = %.f \n' % (npar)
GoF_ini, GoFw_ini = cal_Goof(a_ini,b_ini,f,R)
print 'initial GoF = %.4f GoFw = %.4f'%(GoF_ini,GoFw_ini)
mode = raw_input('Please select parameter, thar the weighting scheme should be optimised for. Q for Q-Q-Plot and G for GOOFw. [Q]') or 'Q'
if mode == 'Q':
    if refine:
        (a,b) = gradientDescent(a_ini, b_ini, f, R, epsilon, alpha, ep, max_iter)
    else:
        (a,b) = a_ini, b_ini
        print 'Weighting scheme not refined\nWEIGHT   %.4f %.4f 0.0 0.0 0.0 %.4f' % (iter, a, b, f)
        print '\n  Note: XDweight variables can be specified! \n   USAGE: XDweight.py a_ini[0.01] b_ini[0.01] .fco[xd.fco] epsilon[0.0001] alpha[0.0001] converged[0.00001] max_iter[200] \n'

else:
    (a,b) = fitGOOF(a_ini, b_ini, f, R, epsilon, alpha, ep, max_iter)
if Check:
    P = do_Plot(a, b, f, R)
