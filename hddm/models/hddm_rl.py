
"""
"""

from copy import copy
import numpy as np
import pymc
import wfpt

from kabuki.hierarchical import Knode
from kabuki.utils import stochastic_from_dist
from hddm.models import HDDM
from wfpt import wiener_like_rlddm


class HDDMrl(HDDM):
    """HDDM model that can be used for two-armed bandit tasks.

    """

    def __init__(self, *args, **kwargs):
        self.non_centered = kwargs.pop('non_centered', False)
        self.dual = kwargs.pop('dual', False)
        self.alpha = kwargs.pop('alpha', True)
        self.b_stn = kwargs.pop('b_stn',False)
        self.b_theta = kwargs.pop('b_theta',False)
        self.b_presma = kwargs.pop('b_presma',False)
        self.b_caud = kwargs.pop('b_caud',False)
        self.b_conflict = kwargs.pop('b_conflict',False)
        self.b_stn_theta = kwargs.pop('b_stn_theta',False)
        self.b_theta_conflict = kwargs.pop('b_theta_conflict',False)
        self.b_stn_conflict = kwargs.pop('b_stn_conflict',False)
        self.b_presma_conflict = kwargs.pop('b_presma_conflict',False)
        self.b_theta_stn_conflict = kwargs.pop('b_theta_stn_conflict',False)
        self.wfpt_rl_class = WienerRL

        super(HDDMrl, self).__init__(*args, **kwargs)

    def _create_stochastic_knodes(self, include):
        knodes = super(HDDMrl, self)._create_stochastic_knodes(include)
        if self.non_centered:
            print('setting learning rate parameter(s) to be non-centered')
            if self.alpha:
                knodes.update(self._create_family_normal_non_centered(
                    'alpha', value=0, g_mu=0.2, g_tau=3**-2, std_lower=1e-10, std_upper=10, std_value=.1))
            if self.dual:
               knodes.update(self._create_family_normal_non_centered(
                    'pos_alpha', value=0, g_mu=0.2, g_tau=3**-2, std_lower=1e-10, std_upper=10, std_value=.1))
        else:
            if self.alpha:
                knodes.update(self._create_family_normal(
                    'alpha', value=0, g_mu=0.2, g_tau=3**-2, std_lower=1e-10, std_upper=10, std_value=.1))
            if self.dual:
               knodes.update(self._create_family_normal(
                    'pos_alpha', value=0, g_mu=0.2, g_tau=3**-2, std_lower=1e-10, std_upper=10, std_value=.1))
            if self.b_stn:
                knodes['b_stn_bottom'] = Knode(pymc.Normal, 'b_stn', mu=0, tau=2**-2, value=0, depends=self.depends['b_stn'])
            if self.b_theta:
                knodes['b_theta_bottom'] = Knode(pymc.Normal, 'b_theta', mu=0, tau=2**-2, value=0, depends=self.depends['b_theta'])
            if self.b_presma:
                knodes['b_presma_bottom'] = Knode(pymc.Normal, 'b_presma', mu=0, tau=2**-2, value=0, depends=self.depends['b_presma'])
            if self.b_caud:
                knodes['b_caud_bottom'] = Knode(pymc.Normal, 'b_caud', mu=0, tau=2**-2, value=0, depends=self.depends['b_caud'])
            if self.b_conflict:
                knodes['b_conflict_bottom'] = Knode(pymc.Normal, 'b_conflict', mu=0, tau=2**-2, value=0, depends=self.depends['b_conflict'])
            if self.b_stn_theta:
                knodes['b_stn_theta_bottom'] = Knode(pymc.Normal, 'b_stn_theta', mu=0, tau=2**-2, value=0, depends=self.depends['b_stn_theta'])
            if self.b_theta_conflict:
                knodes['b_theta_conflict_bottom'] = Knode(pymc.Normal, 'b_theta_conflict', mu=0, tau=2**-2, value=0, depends=self.depends['b_theta_conflict'])
            if self.b_stn_conflict:
                knodes['b_stn_conflict_bottom'] = Knode(pymc.Normal, 'b_stn_conflict', mu=0, tau=2**-2, value=0, depends=self.depends['b_stn_conflict'])
            if self.b_presma_conflict:
                knodes['b_presma_conflict_bottom'] = Knode(pymc.Normal, 'b_presma_conflict', mu=0, tau=2**-2, value=0, depends=self.depends['b_presma_conflict'])
            if self.b_theta_stn_conflict:
                knodes['b_theta_stn_conflict_bottom'] = Knode(pymc.Normal, 'b_theta_stn_conflict', mu=0, tau=2**-2, value=0, depends=self.depends['b_theta_stn_conflict'])

        return knodes

    def _create_wfpt_parents_dict(self, knodes):
        wfpt_parents = super(HDDMrl, self)._create_wfpt_parents_dict(knodes)
        wfpt_parents['alpha'] = knodes['alpha_bottom']
        wfpt_parents['b_stn'] = knodes['b_stn_bottom'] if self.b_stn else 0.0
        wfpt_parents['b_theta'] = knodes['b_theta_bottom'] if self.b_theta else 0.0
        wfpt_parents['b_presma'] = knodes['b_presma_bottom'] if self.b_presma else 0.0
        wfpt_parents['b_caud'] = knodes['b_caud_bottom'] if self.b_caud else 0.0
        wfpt_parents['b_conflict'] = knodes['b_conflict_bottom'] if self.b_conflict else 0.0
        wfpt_parents['b_stn_theta'] = knodes['b_stn_theta_bottom'] if self.b_stn_theta else 0.0
        wfpt_parents['b_theta_conflict'] = knodes['b_theta_conflict_bottom'] if self.b_theta_conflict else 0.0
        wfpt_parents['b_stn_conflict'] = knodes['b_stn_conflict_bottom'] if self.b_stn_conflict else 0.0
        wfpt_parents['b_presma_conflict'] = knodes['b_presma_conflict_bottom'] if self.b_presma_conflict else 0.0
        wfpt_parents['b_theta_stn_conflict'] = knodes['b_theta_stn_conflict_bottom'] if self.b_theta_stn_conflict else 0.0
        wfpt_parents['pos_alpha'] = knodes['pos_alpha_bottom'] if self.dual else 100.00
        return wfpt_parents

    def _create_wfpt_knode(self, knodes):
        wfpt_parents = self._create_wfpt_parents_dict(knodes)
        return Knode(self.wfpt_rl_class, 'wfpt', observed=True, col_name=['split_by', 'feedback', 'response', 'rt', 'q_init','stn','theta','presma','caud'], **wfpt_parents)


def wienerRL_like(x, v, alpha, pos_alpha, b_stn, b_theta, b_presma, b_caud, b_conflict, b_stn_theta, b_theta_conflict, b_stn_conflict, b_presma_conflict, b_theta_stn_conflict, sv, a, z, sz, t, st, p_outlier=0):

    wiener_params = {'err': 1e-4, 'n_st': 2, 'n_sz': 2,
                     'use_adaptive': 1,
                     'simps_err': 1e-3,
                     'w_outlier': 0.1}
    wp = wiener_params
    response = x['response'].values.astype(int)
    q = x['q_init'].iloc[0]
    feedback = x['feedback'].values
    split_by = x['split_by'].values
    stn = x['stn'].values
    theta = x['theta'].values
    presma = x['presma'].values
    caud = x['caud'].values
    return wiener_like_rlddm(x['rt'].values, response, feedback, split_by, stn, theta, presma, caud, q, alpha, pos_alpha, b_stn, b_theta, b_presma, b_caud, b_conflict, b_stn_theta, b_theta_conflict, b_stn_conflict, b_presma_conflict, b_theta_stn_conflict, v, sv, a, z, sz, t, st, p_outlier=p_outlier, **wp)
WienerRL = stochastic_from_dist('wienerRL', wienerRL_like)
