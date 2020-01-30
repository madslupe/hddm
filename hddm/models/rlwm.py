
"""
"""

from copy import copy
import numpy as np
import pymc
import wfpt

from kabuki.hierarchical import Knode
from kabuki.utils import stochastic_from_dist
from hddm.models import HDDM
from wfpt import wiener_like_rlwm
from collections import OrderedDict

#the  RL  learning  rate 𝛼,  WM  capacity K,  WM decay 𝜑,WM prior weight , positive learning biasparameter bias, and undirected noise 𝜀.

class Hrlwm(HDDM):
    """rlwm model that can be used to analyze data from two-armed bandit tasks.
    """
    #parameters: dobule rho (inital wm weight) (0,1), double phi (decay) (0,1), double K (1,2,3,4,5), double epsilon (0,1), double pers (0,1)

    def __init__(self, *args, **kwargs):
        self.non_centered = kwargs.pop('non_centered', False)
        self.alpha = kwargs.pop('alpha', True)
        self.rho = kwargs.pop('rho', True)
        self.phi = kwargs.pop('phi', True)
        self.epsilon = kwargs.pop('epsilon', True)
        self.pers = kwargs.pop('pers', True)
        #self.K = kwargs.pop('K', True)
        self.z = kwargs.pop('z', False)
        self.rlwm_class = RLWM

        super(Hrlwm, self).__init__(*args, **kwargs)

    def _create_stochastic_knodes(self, include):
        #v is not estimated, try to exclude it. used to be params = ['v']
        params = ['v']
        if 'p_outlier' in self.include:
            params.append('p_outlier')
        if 'z' in self.include:
            params.append('z')
        include = set(params)


        knodes = super(Hrlwm, self)._create_stochastic_knodes(include)
        if self.non_centered:
            print('setting learning rate parameter(s) to be non-centered')
            if self.alpha:
                knodes.update(self._create_family_normal_non_centered(
                    'alpha', value=0, g_mu=0.2, g_tau=3**-2, std_lower=1e-10, std_upper=10, std_value=.1))
        else:
            if self.alpha:
                knodes.update(self._create_family_normal(
                    'alpha', value=0, g_mu=0.2, g_tau=3**-2, std_lower=1e-10, std_upper=10, std_value=.1))
            if self.rho:
                knodes.update(self._create_family_invlogit(
                    'rho', value=.5, g_tau=0.5**-2, std_std=0.05))
            if self.phi:
                knodes.update(self._create_family_invlogit(
                    'phi', value=.5, g_tau=0.5**-2, std_std=0.05))
            if self.epsilon:
                knodes.update(self._create_family_invlogit(
                    'epsilon', value=.5, g_tau=0.5**-2, std_std=0.05))
            if self.pers:
                knodes.update(self._create_family_invlogit(
                    'pers', value=.5, g_tau=0.5**-2, std_std=0.05))
            #if self.K:
            #    knodes.update(self._create_family_categorical(
            #        'K', value=1, g_theta = np.array([1/6,1/6,1/6,1/6,1/6,1/6])))

        return knodes

    def _create_wfpt_parents_dict(self, knodes):
        wfpt_parents = OrderedDict()
        wfpt_parents['v'] = knodes['v_bottom']
        wfpt_parents['alpha'] = knodes['alpha_bottom']
        wfpt_parents['rho'] = knodes['rho_bottom']
        wfpt_parents['phi'] = knodes['phi_bottom']
        wfpt_parents['epsilon'] = knodes['epsilon_bottom']
        wfpt_parents['pers'] = knodes['pers_bottom']
        #wfpt_parents['K'] = knodes['K_bottom']
        wfpt_parents['z'] = knodes['z_bottom'] if 'z' in self.include else 0.5

        return wfpt_parents

    def _create_wfpt_knode(self, knodes):
        wfpt_parents = self._create_wfpt_parents_dict(knodes)
        return Knode(self.rlwm_class, 'wfpt', observed=True, col_name=['split_by', 'feedback', 'response', 'n_stim','stim'], **wfpt_parents)


def RLWM_like(x, v, alpha, rho, phi, epsilon, pers, z=0.5, p_outlier=0): #put in K here when it works with dirichlet/categorical

    wiener_params = {'err': 1e-4, 'n_st': 2, 'n_sz': 2,
                     'use_adaptive': 1,
                     'simps_err': 1e-3,
                     'w_outlier': 0.1}
    sum_logp = 0
    wp = wiener_params
    response = x['response'].values.astype(int)
    n = x['n_stim'].values
    feedback = x['feedback'].values
    split_by = x['split_by'].values
    stim = x['stim'].values
    K = 5
    #print('response: ', response, 'n: ', n, 'feedback: ', feedback, 'split_by: ', split_by, 'alpha: ', alpha, 'rho: ', rho, 'phi: ', phi, 'epsilon: ', epsilon, 'pers: ', pers, 'K: ', K)
    return wiener_like_rlwm(response, feedback, split_by, stim, n, alpha, z, rho, phi, epsilon, pers, K, p_outlier=p_outlier, **wp)
RLWM = stochastic_from_dist('RLWM', RLWM_like)