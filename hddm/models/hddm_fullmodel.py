
"""
"""

from copy import copy
import numpy as np
import pymc
import wfpt

from kabuki.hierarchical import Knode
from kabuki.utils import stochastic_from_dist
from hddm.models import HDDM
from wfpt import wiener_like_fullmodel


class HDDMfullmodel(HDDM):
    """

    """

    def __init__(self, *args, **kwargs):
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

        self.within_med = kwargs.pop('within_med',True)

        self.wfpt_fullmodel_class = WienerFULLMODEL

        super(HDDMfullmodel, self).__init__(*args, **kwargs)

    def _create_stochastic_knodes(self, include):
        knodes = super(HDDMfullmodel, self)._create_stochastic_knodes(include)
        if self.b_v_zpos:
            knodes.update(self._create_family_normal(
                    'b_v_zpos', value=0, g_mu=0.2, g_tau=3**-2, std_lower=1e-10, std_upper=10, std_value=.1))
        if self.b_v_zneg:
            knodes.update(self._create_family_normal(
                    'b_v_zneg', value=0, g_mu=0.2, g_tau=3**-2, std_lower=1e-10, std_upper=10, std_value=.1))
        if self.b_v_dsess:
            knodes['b_v_dsess_bottom'] = Knode(pymc.Normal, 'b_v_dsess', mu=0, tau=2**-2, value=0, depends=self.depends['b_v_dsess'])
        if self.b_v_zpos_dsess:
            knodes['b_v_zpos_dsess_bottom'] = Knode(pymc.Normal, 'b_v_zpos_dsess', mu=0, tau=2**-2, value=0, depends=self.depends['b_v_zpos_dsess'])
        if self.b_v_zneg_dsess:
            knodes['b_v_zneg_dsess_bottom'] = Knode(pymc.Normal, 'b_v_zneg_dsess', mu=0, tau=2**-2, value=0, depends=self.depends['b_v_zneg_dsess'])
        if self.b_v_dmed_dsess:
            knodes['b_v_dmed_dsess_bottom'] = Knode(pymc.Normal, 'b_v_dmed_dsess', mu=0, tau=2**-2, value=0, depends=self.depends['b_v_dmed_dsess'])
        if self.b_v_zpos_dmed_dsess:
            knodes['b_v_zpos_dmed_dsess_bottom'] = Knode(pymc.Normal, 'b_v_zpos_dmed_dsess', mu=0, tau=2**-2, value=0, depends=self.depends['b_v_zpos_dmed_dsess'])
        if self.b_v_zneg_dmed_dsess:
            knodes['b_v_zneg_dmed_dsess_bottom'] = Knode(pymc.Normal, 'b_v_zneg_dmed_dsess', mu=0, tau=2**-2, value=0, depends=self.depends['b_v_zneg_dmed_dsess'])
        if self.b_a_dsess:
            knodes['b_a_dsess_bottom'] = Knode(pymc.Normal, 'b_a_dsess', mu=0, tau=2**-2, value=0, depends=self.depends['b_a_dsess'])
        if self.b_a_dmed_dsess:
            knodes['b_a_dmed_dsess_bottom'] = Knode(pymc.Normal, 'b_a_dmed_dsess', mu=0, tau=2**-2, value=0, depends=self.depends['b_a_dmed_dsess'])
        if self.b_t_dsess:
            knodes['b_t_dsess_bottom'] = Knode(pymc.Normal, 'b_t_dsess', mu=0, tau=2**-2, value=0, depends=self.depends['b_t_dsess'])
        if self.b_t_dmed_dsess:
            knodes['b_t_dmed_dsess_bottom'] = Knode(pymc.Normal, 'b_t_dmed_dsess', mu=0, tau=2**-2, value=0, depends=self.depends['b_t_dmed_dsess'])
        if self.b_z_dsess:
            knodes['b_z_dsess_bottom'] = Knode(pymc.Normal, 'b_z_dsess', mu=0, tau=2**-2, value=0, depends=self.depends['b_z_dsess'])
        if self.b_z_dmed_dsess:
            knodes['b_z_dmed_dsess_bottom'] = Knode(pymc.Normal, 'b_z_dmed_dsess', mu=0, tau=2**-2, value=0, depends=self.depends['b_z_dmed_dsess'])
        if self.within_med:
            if self.b_v_dmed:
                knodes.update(self._create_family_normal(
                        'b_v_dmed', value=0, g_mu=0.2, g_tau=3**-2, std_lower=1e-10, std_upper=10, std_value=.1))
            if self.b_v_zpos_dmed:
                knodes.update(self._create_family_normal(
                        'b_v_zpos_dmed', value=0, g_mu=0.2, g_tau=3**-2, std_lower=1e-10, std_upper=10, std_value=.1))
            if self.b_v_zneg_dmed:
                knodes.update(self._create_family_normal(
                        'b_v_zneg_dmed', value=0, g_mu=0.2, g_tau=3**-2, std_lower=1e-10, std_upper=10, std_value=.1))
            if self.b_a_dmed:
                knodes.update(self._create_family_normal(
                    'b_a_dmed', value=0, g_mu=0.2, g_tau=3**-2, std_lower=1e-10, std_upper=10, std_value=.1))
            if self.b_t_dmed:
                knodes.update(self._create_family_normal(
                        'b_t_dmed', value=0, g_mu=0.2, g_tau=3**-2, std_lower=1e-10, std_upper=10, std_value=.1))
            if self.b_z_dmed:
                knodes.update(self._create_family_normal(
                        'b_z_dmed', value=0, g_mu=0.2, g_tau=3**-2, std_lower=1e-10, std_upper=10, std_value=.1))
        else:
            if self.b_v_dmed:
                knodes['b_v_dmed_bottom'] = Knode(pymc.Normal, 'b_v_dmed', mu=0, tau=2**-2, value=0, depends=self.depends['b_v_dmed'])
            if self.b_v_zpos_dmed:
                knodes['b_v_zpos_dmed_bottom'] = Knode(pymc.Normal, 'b_v_zpos_dmed', mu=0, tau=2**-2, value=0, depends=self.depends['b_v_zpos_dmed'])
            if self.b_v_zneg_dmed:
                knodes['b_v_zneg_dmed_bottom'] = Knode(pymc.Normal, 'b_v_zneg_dmed', mu=0, tau=2**-2, value=0, depends=self.depends['b_v_zneg_dmed'])
            if self.b_a_dmed:
                knodes['b_a_dmed_bottom'] = Knode(pymc.Normal, 'b_a_dmed', mu=0, tau=2**-2, value=0, depends=self.depends['b_a_dmed'])
            if self.b_t_dmed:
                knodes['b_t_dmed_bottom'] = Knode(pymc.Normal, 'b_t_dmed', mu=0, tau=2**-2, value=0, depends=self.depends['b_t_dmed'])
            if self.b_z_dmed:
                knodes['b_z_dmed_bottom'] = Knode(pymc.Normal, 'b_z_dmed', mu=0, tau=2**-2, value=0, depends=self.depends['b_z_dmed'])


        return knodes

    def _create_wfpt_parents_dict(self, knodes):
        wfpt_parents = super(HDDMfullmodel, self)._create_wfpt_parents_dict(knodes)
        wfpt_parents['b_v_zpos'] = knodes['b_v_zpos_bottom'] if self.b_v_zpos else 0.0
        wfpt_parents['b_v_zneg'] = knodes['b_v_zneg_bottom'] if self.b_v_zneg else 0.0
        wfpt_parents['b_v_dmed'] = knodes['b_v_dmed_bottom'] if self.b_v_dmed else 0.0
        wfpt_parents['b_v_dsess'] = knodes['b_v_dsess_bottom'] if self.b_v_dsess else 0.0
        wfpt_parents['b_v_zpos_dmed'] = knodes['b_v_zpos_dmed_bottom'] if self.b_v_zpos_dmed else 0.0
        wfpt_parents['b_v_zneg_dmed'] = knodes['b_v_zneg_dmed_bottom'] if self.b_v_zneg_dmed else 0.0
        wfpt_parents['b_v_zpos_dsess'] = knodes['b_v_zpos_dsess_bottom'] if self.b_v_zpos_dsess else 0.0
        wfpt_parents['b_v_zneg_dsess'] = knodes['b_v_zneg_dsess_bottom'] if self.b_v_zneg_dsess else 0.0
        wfpt_parents['b_v_dmed_dsess'] = knodes['b_v_dmed_dsess_bottom'] if self.b_v_dmed_dsess else 0.0
        wfpt_parents['b_v_zpos_dmed_dsess'] = knodes['b_v_zpos_dmed_dsess_bottom'] if self.b_v_zpos_dmed_dsess else 0.0
        wfpt_parents['b_v_zneg_dmed_dsess'] = knodes['b_v_zneg_dmed_dsess_bottom'] if self.b_v_zneg_dmed_dsess else 0.0
        wfpt_parents['b_a_dmed'] = knodes['b_a_dmed_bottom']  if self.b_a_dmed else 0.0
        wfpt_parents['b_a_dsess'] = knodes['b_a_dsess_bottom'] if self.b_a_dsess else 0.0
        wfpt_parents['b_a_dmed_dsess'] = knodes['b_a_dmed_dsess_bottom'] if self.b_a_dmed_dsess else 0.0
        wfpt_parents['b_t_dmed'] = knodes['b_t_dmed_bottom']  if self.b_t_dmed else 0.0
        wfpt_parents['b_t_dsess'] = knodes['b_t_dsess_bottom'] if self.b_t_dsess else 0.0
        wfpt_parents['b_t_dmed_dsess'] = knodes['b_t_dmed_dsess_bottom'] if self.b_t_dmed_dsess else 0.0
        wfpt_parents['b_z_dmed'] = knodes['b_z_dmed_bottom'] if self.b_z_dmed else 0.0
        wfpt_parents['b_z_dsess'] = knodes['b_z_dsess_bottom'] if self.b_z_dsess else 0.0
        wfpt_parents['b_z_dmed_dsess'] = knodes['b_z_dmed_dsess_bottom'] if self.b_z_dmed_dsess else 0.0
        return wfpt_parents

    def _create_wfpt_knode(self, knodes):
        wfpt_parents = self._create_wfpt_parents_dict(knodes)
        return Knode(self.wfpt_fullmodel_class, 'wfpt', observed=True, col_name=['rt', 'zpos','zneg','dmed','dsess'], **wfpt_parents)


def wienerFULLMODEL_like(x, v, sv, a, z, sz, t, st,b_v_zpos, b_v_zneg, b_v_dmed, b_v_dsess, b_v_zpos_dmed, b_v_zneg_dmed, b_v_zpos_dsess, b_v_zneg_dsess, b_v_dmed_dsess, b_v_zpos_dmed_dsess, b_v_zneg_dmed_dsess, b_a_dmed, b_a_dsess, b_a_dmed_dsess, b_t_dmed, b_t_dsess, b_t_dmed_dsess, b_z_dmed, b_z_dsess, b_z_dmed_dsess, p_outlier=0):

    wiener_params = {'err': 1e-4, 'n_st': 2, 'n_sz': 2,
                     'use_adaptive': 1,
                     'simps_err': 1e-3,
                     'w_outlier': 0.1}
    wp = wiener_params
    zpos = x['zpos'].values.astype(float)
    zneg = x['zneg'].values.astype(float)
    dmed = x['dmed'].values.astype(float)
    dsess = x['dsess'].values.astype(float)
    return wiener_like_fullmodel(x['rt'].values, zpos, zneg, dmed, dsess, b_v_zpos, b_v_zneg, b_v_dmed, b_v_dsess, b_v_zpos_dmed, b_v_zneg_dmed, b_v_zpos_dsess, b_v_zneg_dsess, b_v_dmed_dsess, b_v_zpos_dmed_dsess, b_v_zneg_dmed_dsess, b_a_dmed, b_a_dsess, b_a_dmed_dsess, b_t_dmed, b_t_dsess, b_t_dmed_dsess, b_z_dmed, b_z_dsess, b_z_dmed_dsess, v, sv, a, z, sz, t, st, p_outlier=p_outlier, **wp)
WienerFULLMODEL = stochastic_from_dist('wienerFULLMODEL', wienerFULLMODEL_like)



