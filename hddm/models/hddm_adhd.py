
"""
"""

from copy import copy
import numpy as np
import pymc
import wfpt

from kabuki.hierarchical import Knode
from kabuki.utils import stochastic_from_dist
from hddm.models import HDDM
from wfpt import wiener_like_adhd


class HDDMadhd(HDDM):
    """HDDM model that can be used for two-armed bandit tasks.

    """

    def __init__(self, *args, **kwargs):
        self.non_centered = kwargs.pop('non_centered', False)
        self.b_v_zpos = kwargs.pop('b_v_zpos', True)
        self.b_v_zneg = kwargs.pop('b_v_zneg', True)
        self.b_v_dmed = kwargs.pop('b_v_dmed', True)
        self.b_v_dsess = kwargs.pop('b_v_dsess', True)
        self.b_v_zpos_dmed = kwargs.pop('b_v_zpos_dmed', True)
        self.b_v_zneg_dmed = kwargs.pop('b_v_zneg_dmed', True)
        self.b_v_zpos_dsess = kwargs.pop('b_v_zpos_dsess', True)
        self.b_v_zneg_dsess = kwargs.pop('b_v_zneg_dsess', True)
        self.b_v_dmed_dsess = kwargs.pop('b_v_dmed_dsess', True)
        self.b_v_zpos_dmed_dsess = kwargs.pop('b_v_zpos_dmed_dsess', True)
        self.b_v_zneg_dmed_dsess = kwargs.pop('b_v_zneg_dmed_dsess', True)

        self.b_a_dmed = kwargs.pop('b_a_dmed', True)
        self.b_a_dsess = kwargs.pop('b_a_dsess', True)
        self.b_a_dmed_dsess = kwargs.pop('b_a_dmed_dsess', True)

        self.b_t_dmed = kwargs.pop('b_t_dmed', True)
        self.b_t_dsess = kwargs.pop('b_t_dsess', True)
        self.b_t_dmed_dsess = kwargs.pop('b_t_dmed_dsess', True)

        self.b_z_dmed = kwargs.pop('b_z_dmed', True)
        self.b_z_dsess = kwargs.pop('b_z_dsess', True)
        self.b_z_dmed_dsess = kwargs.pop('b_z_dmed_dsess', True)

        self.wfpt_adhd_class = WienerADHD

        super(HDDMadhd, self).__init__(*args, **kwargs)

    def _create_stochastic_knodes(self, include):
        knodes = super(HDDMadhd, self)._create_stochastic_knodes(include)
        if self.b_v_zpos:
            knodes.update(self._create_family_normal(
                    'b_v_zpos', value=0, g_mu=0.2, g_tau=3**-2, std_lower=1e-10, std_upper=10, std_value=.1))
        if self.b_v_zneg:
            knodes.update(self._create_family_normal(
                    'b_v_zneg', value=0, g_mu=0.2, g_tau=3**-2, std_lower=1e-10, std_upper=10, std_value=.1))
        if self.b_v_dmed:
            knodes.update(self._create_family_normal(
                    'b_v_dmed', value=0, g_mu=0.2, g_tau=3**-2, std_lower=1e-10, std_upper=10, std_value=.1))
        if self.b_v_dsess:
            knodes.update(self._create_family_normal(
                    'b_v_dsess', value=0, g_mu=0.2, g_tau=3**-2, std_lower=1e-10, std_upper=10, std_value=.1))
        if self.b_v_zpos_dmed:
            knodes.update(self._create_family_normal(
                    'b_v_zpos_dmed', value=0, g_mu=0.2, g_tau=3**-2, std_lower=1e-10, std_upper=10, std_value=.1))
        if self.b_v_zneg_dmed:
            knodes.update(self._create_family_normal(
                    'b_v_zneg_dmed', value=0, g_mu=0.2, g_tau=3**-2, std_lower=1e-10, std_upper=10, std_value=.1))
        if self.b_v_zpos_dsess:
            knodes.update(self._create_family_normal(
                    'b_v_zpos_dsess', value=0, g_mu=0.2, g_tau=3**-2, std_lower=1e-10, std_upper=10, std_value=.1))
        if self.b_v_zneg_dsess:
            knodes.update(self._create_family_normal(
                    'b_v_zneg_dsess', value=0, g_mu=0.2, g_tau=3**-2, std_lower=1e-10, std_upper=10, std_value=.1))
        if self.b_v_dmed_dsess:
            knodes['b_v_dmed_dsess_bottom'] = Knode(pymc.Normal, 'b_v_dmed_dsess', mu=0, tau=2**-2, value=0, depends=self.depends['b_v_dmed_dsess'])
        if self.b_v_zpos_dmed_dsess:
            knodes['b_v_zpos_dmed_dsess_bottom'] = Knode(pymc.Normal, 'b_v_zpos_dmed_dsess', mu=0, tau=2**-2, value=0, depends=self.depends['b_v_zpos_dmed_dsess'])
        if self.b_v_zneg_dmed_dsess:
            knodes['b_v_zneg_dmed_dsess_bottom'] = Knode(pymc.Normal, 'b_v_zneg_dmed_dsess', mu=0, tau=2**-2, value=0, depends=self.depends['b_v_zneg_dmed_dsess'])

        if self.b_a_dmed:
            knodes.update(self._create_family_normal(
                    'b_a_dmed', value=0, g_mu=0.2, g_tau=3**-2, std_lower=1e-10, std_upper=10, std_value=.1))
        if self.b_a_dsess:
            knodes.update(self._create_family_normal(
                    'b_a_dsess', value=0, g_mu=0.2, g_tau=3**-2, std_lower=1e-10, std_upper=10, std_value=.1))
        if self.b_a_dmed_dsess:
            knodes['b_a_dmed_dsess_bottom'] = Knode(pymc.Normal, 'b_a_dmed_dsess', mu=0, tau=2**-2, value=0, depends=self.depends['b_a_dmed_dsess'])


        if self.b_t_dmed:
            knodes.update(self._create_family_normal(
                    'b_t_dmed', value=0, g_mu=0.2, g_tau=3**-2, std_lower=1e-10, std_upper=10, std_value=.1))
        if self.b_t_dsess:
            knodes.update(self._create_family_normal(
                    'b_t_dsess', value=0, g_mu=0.2, g_tau=3**-2, std_lower=1e-10, std_upper=10, std_value=.1))
        if self.b_t_dmed_dsess:
            knodes['b_t_dmed_dsess_bottom'] = Knode(pymc.Normal, 'b_t_dmed_dsess', mu=0, tau=2**-2, value=0, depends=self.depends['b_t_dmed_dsess'])


        if self.b_z_dmed:
            knodes.update(self._create_family_normal(
                    'b_z_dmed', value=0, g_mu=0.2, g_tau=3**-2, std_lower=1e-10, std_upper=10, std_value=.1))
        if self.b_z_dsess:
            knodes.update(self._create_family_normal(
                    'b_z_dsess', value=0, g_mu=0.2, g_tau=3**-2, std_lower=1e-10, std_upper=10, std_value=.1))
        if self.b_z_dmed_dsess:
            knodes['b_z_dmed_dsess_bottom'] = Knode(pymc.Normal, 'b_z_dmed_dsess', mu=0, tau=2**-2, value=0, depends=self.depends['b_z_dmed_dsess'])

        return knodes

    def _create_wfpt_parents_dict(self, knodes):
        wfpt_parents = super(HDDMadhd, self)._create_wfpt_parents_dict(knodes)
        wfpt_parents['b_v_zpos'] = knodes['b_v_zpos_bottom']
        wfpt_parents['b_v_zneg'] = knodes['b_v_zneg_bottom']
        wfpt_parents['b_v_dmed'] = knodes['b_v_dmed_bottom']
        wfpt_parents['b_v_dsess'] = knodes['b_v_dsess_bottom']
        wfpt_parents['b_v_zpos_dmed'] = knodes['b_v_zpos_dmed_bottom']
        wfpt_parents['b_v_zneg_dmed'] = knodes['b_v_zpos_dmed_bottom']
        wfpt_parents['b_v_zpos_dsess'] = knodes['b_v_zpos_dsess_bottom']
        wfpt_parents['b_v_zneg_dsess'] = knodes['b_v_zpos_dsess_bottom']
        wfpt_parents['b_v_dmed_dsess'] = knodes['b_v_dmed_dsess_bottom']
        wfpt_parents['b_v_zpos_dmed_dsess'] = knodes['b_v_zpos_dmed_dsess_bottom']
        wfpt_parents['b_v_zneg_dmed_dsess'] = knodes['b_v_zneg_dmed_dsess_bottom']

        wfpt_parents['b_a_dmed'] = knodes['b_a_dmed_bottom']
        wfpt_parents['b_a_dsess'] = knodes['b_a_dsess_bottom']
        wfpt_parents['b_a_dmed_dsess'] = knodes['b_a_dmed_dsess_bottom']

        wfpt_parents['b_t_dmed'] = knodes['b_t_dmed_bottom']
        wfpt_parents['b_t_dsess'] = knodes['b_t_dsess_bottom']
        wfpt_parents['b_t_dmed_dsess'] = knodes['b_t_dmed_dsess_bottom']

        wfpt_parents['b_z_dmed'] = knodes['b_z_dmed_bottom']
        wfpt_parents['b_z_dsess'] = knodes['b_z_dsess_bottom']
        wfpt_parents['b_z_dmed_dsess'] = knodes['b_z_dmed_dsess_bottom']

        return wfpt_parents

    def _create_wfpt_knode(self, knodes):
        wfpt_parents = self._create_wfpt_parents_dict(knodes)
        return Knode(self.wfpt_adhd_class, 'wfpt', observed=True, col_name=['rt', 'zpos','zneg','dmed','dsess'], **wfpt_parents)


def wienerADHD_like(x, v, alpha, pos_alpha, sv, a, z, sz, t, st, p_outlier=0):

    wiener_params = {'err': 1e-4, 'n_st': 2, 'n_sz': 2,
                     'use_adaptive': 1,
                     'simps_err': 1e-3,
                     'w_outlier': 0.1}
    wp = wiener_params
    zpos = x['zpos'].values.astype(float)
    zneg = x['zneg'].values.astype(float)
    dmed = x['dmed'].values.astype(float)
    dsess = x['dsess'].values.astype(float)
    return wiener_like_adhd(x['rt'].values, zpos, zneg, dmed, dsess, b_v_zpos, b_v_zneg, b_v_dmed, b_v_dsess, b_v_zpos_dmed, b_v_zneg_dmed, b_v_zpos_dsess, b_v_zneg_dsess, b_v_dmed_dsess, b_v_zpos_dmed_dsess, b_v_zneg_dmed_dsess, b_a_dmed, b_a_dsess, b_a_dmed_dsess, b_t_dmed, b_t_dsess, b_t_dmed_dsess, b_z_dmed, b_z_dsess, b_z_dmed_dsess, v, sv, a, z, sz, t, st, p_outlier=p_outlier, **wp)
WienerADHD = stochastic_from_dist('wienerADHD', wienerADHD_like)
