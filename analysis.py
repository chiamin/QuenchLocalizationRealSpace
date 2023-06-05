import pylab as pl
from collections import OrderedDict
import sys
from math import pi, acos
sys.path.append ('/home/chiamin/mypy/')
import plotsetting as ps
import numpy as np
import fitfun as ff
import matplotlib.colors as colors
import cmasher as cmr

def Luttinger_g (V):
    return 0.5 * pi / acos(-0.5*V)

def exactG (V, tp):
    g = Luttinger_g (V)
    #if V != 0:
    #    return 0.5 * pi / acos(-0.5*V)
    #else:
    return g * 4*tp**2/(1+tp**2)**2

def get_para (fname, key, typ, last=False, n=1):
    with open(fname) as f:
        for line in f:
            if key in line:
                val = list(map(typ,line.split()[-n:]))
                if n == 1: val = val[0]
                if not last:
                    return val
        return val

def get_data (fname):
    dt = get_para (fname, 'dt', float, last=True)
    t_contactL = get_para (fname, 't_contactL', float)
    t_contactR = get_para (fname, 't_contactR', float)

    I_time_space = OrderedDict()
    dens = OrderedDict()
    iL_dev = OrderedDict()
    iR_dev = OrderedDict()
    bonddim = []
    with open(fname) as f:
        for line in f:
            line = line.lstrip()
            if line.startswith ('step ='):
                tmp = line.split()
                step = int(tmp[-1])
                I_time_space[step] = OrderedDict()
                I_t = I_time_space[step]
                dens[step] = OrderedDict()
                ns = dens[step]
            elif line.startswith ('device site ='):
                idevL = int(line.split()[-2])
                idevR = int(line.split()[-1])
                iL_dev[step] = idevL
                iR_dev[step] = idevR
            elif line.startswith('current'):
                tmp = line.split()
                ilink = int(tmp[1])

                I = float(tmp[-1].strip('()').split(',')[0])
                cc = 2*pi
                if ilink == idevL-1:
                    cc *= t_contactL
                elif ilink == idevR:
                    cc *= t_contactR
                I_t[ilink] = I * cc
            elif line.startswith('n '):
                tmp = line.split()
                i = int(tmp[1])
                n = float(tmp[-1])
                ns[i] = n
            elif line.startswith('Largest link dim'):
                tmp = line.split()
                dim = float(tmp[-1])
                bonddim.append (dim)

    xss, tss, Iss, nss = [],[],[],[]
    I_L, I_R, I_mean, ts = [],[],[],[]
    w = 1
    for step in I_time_space:
        t = step*dt
        Is = list(I_time_space[step].values())
        xs = list(I_time_space[step].keys())
        central_site = int((iL_dev[step] + iR_dev[step] -1)/2)
        xss += [i - central_site for i in xs]
        tss += [t for i in range(len(Is))]
        Iss += Is
        nss += dens[step].values()

        siteL, siteR = iL_dev[step]-1, iR_dev[step]
        if siteL in xs and siteR in xs:
            iL, iR = xs.index(siteL), xs.index(siteR)
            if iL < len(Is) and iR < len(Is):
                I_L.append (Is[iL])
                I_R.append (Is[iR])
                whf = int(w/2)
                I_mean.append (np.mean(Is))#[imp_site-whf:imp_site+whf+1]))
                ts.append (t)

    return xss, tss, Iss, nss, ts, I_L, I_R, I_mean, bonddim

def get_average_current (ts, js, ax=None, tbeg=0., tend=float('inf'), **args):
    ibeg, iend = 0, len(js)
    for i in range(len(js)):
        if ts[i] > tbeg:
            ibeg = i
            break
    for i in range(len(js)):
        if ts[i] > tend:
            iend = i
            break
    G = np.mean (js[ibeg:iend])
    err = np.std (js[ibeg:iend])

    if ax != None:
        ax.plot (ts, js, **args)
        ax.axhline (G, ls='--', c='k')
        ax.axvline (tbeg, ls=':', c='gray')
    return G, err

def to_imag_data (xss, tss, Iss):
    tmap = dict()
    ti = 0
    tpre = np.nan
    for t in tss:
        if t != tpre:
            tmap[t] = ti
            tpre = t
            ti += 1
    Nt = len(tmap)
    tmin = min(tss)
    tmax = max(tss)

    xmin = min(xss)
    xmax = max(xss)
    Nx = xmax-xmin+1

    Idata = np.empty ((Nt,Nx))
    Idata.fill (np.nan)
    for x,t,I in zip(xss,tss,Iss):
        ti = tmap[t]
        xi = x-xmin
        Idata[ti,xi] = I
    return Idata, tmin, tmax, xmin, xmax

def plot_data (fname, f2, ax2, tbeg):
    with open(fname) as f:
        for line in f:
            if 'Largest link dim' in line:
                m = int(line.split()[-1])
    V_lead = get_para (fname, 'V_lead', float)
    g = Luttinger_g(V_lead)
    print ('g =',g)

    xss, tss, Iss, nss, ts, I_Lt, I_Rt, I_mean, bonddim = get_data (fname)

    f1,ax1 = pl.subplots()
    Idata, tmin, tmax, xmin, xmax = to_imag_data (xss, tss, Iss)
    Imax, Imin = max(Iss), min(Iss)
    Ilim = max([abs(Imax),abs(Imin)])
    midnorm = ps.MidpointNormalize (vmax=Ilim, vmin=-Ilim, vzero=0.)
    sc = ax1.imshow (Idata, origin='lower', extent=[xmin, xmax, tmin, tmax], aspect='auto', cmap='cmr.pride_r',norm=midnorm)
    #sc = ax1.scatter (xss, tss, c=Iss)
    cb = pl.colorbar (sc)
    ax1.set_title ('$m='+str(m)+'$')
    ax1.set_xlabel ('site')
    ax1.set_ylabel ('time')
    cb.ax.set_ylabel ('current')
    #cb.ax.set_ylim (Imin, Imax)

    f,ax = pl.subplots()
    Idata, tmin, tmax, xmin, xmax = to_imag_data (xss, tss, nss)
    sc = ax.imshow (Idata, origin='lower', extent=[xmin, xmax, tmin, tmax], aspect='auto')
    #sc = ax.scatter (xss, tss, c=nss)
    cb = pl.colorbar (sc)
    ax.set_title ('$m='+str(m)+'$')
    ax.set_xlabel ('site')
    ax.set_ylabel ('time')
    cb.ax.set_ylabel ('density')

    muL = get_para (fname, 'mu_biasL', float)
    muR = get_para (fname, 'mu_biasR', float)
    Vg = muL - muR
    I_Lt = [i/Vg for i in I_Lt]
    I_Rt = [i/Vg for i in I_Rt]
    IL, errL = get_average_current (ts, I_Lt, ax=ax2, tbeg=tbeg, marker='.', label='left')
    IR, errR = get_average_current (ts, I_Rt, ax=ax2, tbeg=tbeg, marker='.', label='right')
    print (IL, errL)
    print (IR, errR)
    ax2.set_xlabel ('time')
    ax2.set_ylabel ('current')

    f,ax = pl.subplots()
    ax.plot (range(len(bonddim)), bonddim)
    ax.set_xlabel ('time step')
    ax.set_ylabel ('bond dimension')


    ps.set([ax, ax1, ax2])
    if '-pdf' in sys.argv:
        f1.savefig (fname.replace('.out','')+'_I_space.pdf')
        f1.savefig (fname.replace('.out','')+'_den_space.pdf')
        f2.savefig (fname.replace('.out','')+'_I.pdf')

    #tp = get_para (fname, 't_contactR', float)
    #Gex = exactG (V_lead, tp)
    #print ('Gex',Gex)
    #ax2.axhline (Gex*Vg,ls='--',c='gray')

if __name__ == '__main__':
    fc,axc = pl.subplots()
    tbeg = 20

    files = [i for i in sys.argv[1:] if i[0] != '-']
    files = sorted(files)
    for fname in files:
        print (fname)

        plot_data (fname, fc, axc, tbeg)
    #axc.set_xlim (0, 30)
    axc.legend()
    ps.set(axc)
    if '-pdf' in sys.argv:
        fc.savefig ('I.pdf')
    pl.show()
