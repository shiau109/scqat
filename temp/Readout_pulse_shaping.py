
#%% Square/Two-step/three-step pulse correction for both qubit states, respectively.

import matplotlib.pyplot as plt
import numpy as np 
from tqdm import tqdm

def Readout_cal_amp_phase_correction(para):
    tp = int(10)
    hbar = 1.054571800*1e-34
    ke = 2*np.pi*para['ke']*1e9
    ki = 2*np.pi*para['ki']*1e9
    keff = ke+ki
    Wc = 2*np.pi*para['fc']*1e9
    Wrf = 2*np.pi*para['frf']*1e9
    delta = Wc-Wrf
    P_in = para['P_in']
    X = -2*np.pi*para['X_eff']*1e9
    
    amp = np.sqrt(10**(P_in/10)/1000/hbar/Wrf)
    
    t = np.linspace(0, para['time'], tp*int(para['time'])+1)*1e-9
    ti = para['ti']*1e-9
    Du = para['Du']*1e-9
    first_step_Du = para['first_step_Du']*1e-9
    
    a_g = np.zeros_like(t, dtype=np.complex128)
    a_e = np.zeros_like(t, dtype=np.complex128)
    pulse_timeamp_g = np.zeros_like(t, dtype=np.complex128) 
    pulse_timeamp_e = np.zeros_like(t, dtype=np.complex128)

    dg = -1j*(delta-X)-keff/2
    de = -1j*(delta+X)-keff/2

    def g_cal(A, t_s, a0, tv):
        return A*np.sqrt(ke/2)/dg + (a0 - np.sqrt(ke/2)*A/dg)*np.exp(dg*(tv-t_s))

    def e_cal(A, t_s, a0, tv):
        return A*np.sqrt(ke/2)/de + (a0 - np.sqrt(ke/2)*A/de)*np.exp(de*(tv-t_s))

    if para['pulse_shape'] == 'Square':
        m1 = (t >= ti) & (t <= ti+Du)
        m2 = t > ti+Du
        pulse_timeamp_g[m1] = amp
        pulse_timeamp_e[m1] = amp
        a_g[m1] = g_cal(amp, ti, 0, t[m1])
        a_e[m1] = e_cal(amp, ti, 0, t[m1])
        a_g0, a_e0 = g_cal(amp, ti, 0, ti+Du), e_cal(amp, ti, 0, ti+Du)
        a_g[m2], a_e[m2] = g_cal(0, ti+Du, a_g0, t[m2]), e_cal(0, ti+Du, a_e0, t[m2])
        tf = ti+Du
        pulse_timeamp_g[m1] =  1
        pulse_timeamp_e[m1] =  1

    elif para['pulse_shape'] == 'Two_step':
        if para['phase_correction']== True:
            k_opt_g = 1 / (1 - np.exp(dg * first_step_Du))
            k_opt_e = 1 / (1 - np.exp(de * first_step_Du))
        else:
            k_opt_g = np.abs(1 / (1 - np.exp(dg * first_step_Du)))
            k_opt_e = np.abs(1 / (1 - np.exp(de * first_step_Du)))

        t1 = ti + first_step_Du
        t2 = Du
        m1, m2, m3 = (t >= ti) & (t <= t1), (t > t1) & (t <= t2), t > t2
        
        pulse_timeamp_g[m1], pulse_timeamp_g[m2] = amp*k_opt_g, amp
        pulse_timeamp_e[m1], pulse_timeamp_e[m2] = amp*k_opt_e, amp
        
        a_g[m1], a_e[m1] = g_cal(amp*k_opt_g, ti, 0, t[m1]), e_cal(amp*k_opt_e, ti, 0, t[m1])
        ag1, ae1 = g_cal(amp*k_opt_g, ti, 0, t1), e_cal(amp*k_opt_e, ti, 0, t1)
        
        a_g[m2], a_e[m2] = g_cal(amp, t1, ag1, t[m2]), e_cal(amp, t1, ae1, t[m2])
        ag2, ae2 = g_cal(amp, t1, ag1, t2), e_cal(amp, t1, ae1, t2)
        
        a_g[m3], a_e[m3] = g_cal(0, t2, ag2, t[m3]), e_cal(0, t2, ae2, t[m3])
        tf = t2

        pulse_timeamp_g[m1], pulse_timeamp_g[m2] = k_opt_g, 1
        pulse_timeamp_e[m1], pulse_timeamp_e[m2] = k_opt_e, 1

        print("First_step_amplitude_g=",k_opt_g)
        print("First_step_amplitude_e=",k_opt_e)

    elif para['pulse_shape'] == 'Three_step':
        third_step_Du = para['third_step_Du'] * 1e-9
        if para['phase_correction']== True:
            k_opt_g = 1 / (1 - np.exp(dg * first_step_Du))
            k_opt_e = 1 / (1 - np.exp(de * first_step_Du))
            k_down_g = -1 / (np.exp(-dg * third_step_Du)-1)
            k_down_e = -1 / (np.exp(-de * third_step_Du)-1)
        else:
            k_opt_g = np.abs(1 / (1 - np.exp(dg * first_step_Du)))
            k_opt_e = np.abs(1 / (1 - np.exp(de * first_step_Du)))
            k_down_g = -np.abs(1 / (np.exp(-dg * third_step_Du)-1))
            k_down_e = -np.abs(1 / (np.exp(-de * third_step_Du)-1))

        
  
        t1 = ti + first_step_Du
        t2 = ti + Du
        t3 = t2 + third_step_Du
        
        m1 = (t >= ti) & (t <= t1)
        m2 = (t > t1) & (t <= t2)
        m3 = (t > t2) & (t <= t3)
        m4 = t > t3
        
        pulse_timeamp_g[m1], pulse_timeamp_g[m2], pulse_timeamp_g[m3] = amp*k_opt_g, amp, amp *k_down_g
        pulse_timeamp_e[m1], pulse_timeamp_e[m2], pulse_timeamp_e[m3] = amp*k_opt_e, amp, amp *k_down_e
      
        a_g[m1], a_e[m1] = g_cal(amp*k_opt_g, ti, 0, t[m1]), e_cal(amp*k_opt_e, ti, 0, t[m1])
        ag1, ae1 = g_cal(amp*k_opt_g, ti, 0, t1), e_cal(amp*k_opt_e, ti, 0, t1)
        
        a_g[m2], a_e[m2] = g_cal(amp, t1, ag1, t[m2]), e_cal(amp, t1, ae1, t[m2])
        ag2, ae2 = g_cal(amp, t1, ag1, t2), e_cal(amp, t1, ae1, t2)
        
        a_g[m3], a_e[m3] = g_cal(amp*k_down_g, t2, ag2, t[m3]), e_cal(amp*k_down_e, t2, ae2, t[m3])
        ag3, ae3 = g_cal(amp*k_down_g, t2, ag2, t3), e_cal(amp*k_down_e, t2, ae2, t3)
        
        a_g[m4], a_e[m4] = g_cal(0, t3, ag3, t[m4]), e_cal(0, t3, ae3, t[m4])
        
        tf = t3

        pulse_timeamp_g[m1], pulse_timeamp_g[m2], pulse_timeamp_g[m3] = k_opt_g, 1, k_down_g
        pulse_timeamp_e[m1], pulse_timeamp_e[m2], pulse_timeamp_e[m3] = k_opt_e, 1, k_down_e
        
        print("First_step_amplitude_g=",k_opt_g)
        print("First_step_amplitude_e=",k_opt_e)
        print("Third_step_amplitude_g=",k_down_g)
        print("Third_step_amplitude_e=",k_down_e)

    return dict(t=t, n_g=np.abs(a_g)**2, n_e=np.abs(a_e)**2, pulse_timeamp_g=pulse_timeamp_g, pulse_timeamp_e=pulse_timeamp_e, ti=ti, tf=tf)


# para_test={'ki':0,'ke':0.0004,'fc':5.113100000,'frf':5.113100000, 'X_eff':0.00009,
#            'pulse_shape':'Three_step', 
#            'phase_correction':False,
#            'P_in':-120,
#            'time':8000, 
#            'ti':0,
#            'Du':4000,    
#            'first_step_Du': 100,
#            'third_step_Du': 100 
#           } 
if __name__ == "__main__":
    para_test={'ki':0,'ke':0.0004,'fc':3.078602573,'frf':3.078602573, 'X_eff':0.000075,
           'pulse_shape':'Three_step', 
           'phase_correction':False,
           'P_in':-120,
           'time':6000, 
           'ti':0,
           'Du':4000,    
           'first_step_Du': 100,
           'third_step_Du': 100 
          }
    # para_test={'ki':0,'ke':0.004,'fc':5.95,'frf':5.95, 'X_eff':0.002,
    #            'pulse_shape':'Three_step', 
    #            'phase_correction':False,
    #            'P_in':-120,
    #            'time':2000+1000, 
    #            'ti':0,
    #            'Du':2000,    
    #            'first_step_Du': 100,
    #            'third_step_Du': 100 
    #           } 
    L = Readout_cal_amp_phase_correction(para_test)
    print("max_photon_avg_of_state=", max(L['n_g']),max(L['n_e']))
    print("last_photon_avg_of_state=", L['n_g'][-1], L['n_e'][-1])

    dt = (L['t'][1] - L['t'][0])
    dn_g_dt = np.gradient(L['n_g'], dt)
    dn_e_dt = np.gradient(L['n_e'], dt)

    fig, ax0 = plt.subplots(nrows=5, figsize=(6,8), dpi=200)  

    ax0[0].plot(L['t']*1e9/1000, np.real(L['pulse_timeamp_g']), label=r'$I$', color="b", alpha=0.5, lw=2)   
    ax0[0].plot(L['t']*1e9/1000, np.imag(L['pulse_timeamp_g']), label=r'$Q$', color="r", alpha=0.5, lw=2)   
    ax0[0].set_ylabel(r"$V_g(t)$", size='12') 
    ax0[0].set_xlim(0, max(L['t']*1e9/1000))
    ax0[0].legend(fontsize=10, loc='lower right')
    ax0[0].tick_params('both', labelsize='8', labelbottom=True)

    ax0[1].plot(L['t']*1e9/1000, np.real(L['pulse_timeamp_e']), label=r'$I$', color="b", alpha=0.5, lw=2)   
    ax0[1].plot(L['t']*1e9/1000, np.imag(L['pulse_timeamp_e']), label=r'$Q$', color="r", alpha=0.5, lw=2)   
    ax0[1].set_ylabel(r"$V_e(t)$", size='12') 
    ax0[1].set_xlim(0, max(L['t']*1e9/1000))
    ax0[1].legend(fontsize=10, loc='lower right')
    ax0[1].tick_params('both', labelsize='8', labelbottom=True)


    ax0[2].plot(L['t']*1e9/1000, L['n_g'], color="b", label=r'$|g\rangle$', alpha=0.5, lw=2)   
    ax0[2].plot(L['t']*1e9/1000, L['n_e'], color="r", label=r'$|e\rangle$', alpha=0.5, lw=2)
    ax0[2].axvline(x=L['ti']*1e9/1000, ymin=0, ymax=1, color='k', linestyle='dashed', lw=1) 
    ax0[2].axvline(x=L['tf']*1e9/1000, ymin=0, ymax=1, color='k', linestyle='dashed', lw=1) 
    ax0[2].set_ylabel(r"$n$", size='15')
    ax0[2].set_xlim(0, max(L['t']*1e9/1000))
    ax0[2].tick_params('both', labelsize='8', labelbottom=True)

    ax0[3].plot(L['t']*1e9/1000, dn_g_dt/1e9, color="b", alpha=0.5, lw=2, label=r'$|g\rangle$')  
    ax0[3].plot(L['t']*1e9/1000, dn_e_dt/1e9, color="r", alpha=0.5, lw=2, label=r'$|e\rangle$')  
    ax0[3].set_xlabel(r'$ t\ (\mu$s)', size='15')
    ax0[3].set_ylabel(r'$\Gamma_{r}\ (n/\mathrm{ns})$', size='12')
    ax0[3].axhline(y=0, color='k', linestyle='dashed', lw=1) 
    ax0[3].legend(fontsize=10, loc='lower right')

    ax0[4].plot(L['n_g'], dn_g_dt/1e9, color="b", alpha=0.5, lw=2, label=r'$|g\rangle$')  
    ax0[4].plot(L['n_g'], dn_e_dt/1e9, color="r", alpha=0.5, lw=2, label=r'$|e\rangle$')  
    ax0[4].set_xlabel(r'$n$', size='15')
    ax0[4].set_ylabel(r'$\Gamma_{r}\ (n/\mathrm{ns})$', size='12')
    ax0[4].axhline(y=0, color='k', linestyle='dashed', lw=1) 

    fig.tight_layout()
    plt.show()

    print("max_photon_avg_of_state=", (max(L['n_g'])+max(L['n_e']))/2)
    print("steady_photon_avg_of_state=", ((L['n_g'])[-1]+(L['n_e'])[-1])/2)
