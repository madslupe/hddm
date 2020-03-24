
"""
"""

from copy import copy
import numpy as np
import pymc
import wfpt
import pickle

from kabuki.hierarchical import Knode
from kabuki.utils import stochastic_from_dist
from hddm.models import HDDM
from wfpt import wiener_like_nn


class HDDMnn(HDDM):
    """HDDM model that can be used for two-armed bandit tasks.

    """

    def __init__(self, *args, **kwargs):
        self.non_centered = kwargs.pop('non_centered', False)
        self.wfpt_nn_class = Wienernn

        super(HDDMnn, self).__init__(*args, **kwargs)

    def _create_stochastic_knodes(self, include):
        knodes = super(HDDMnn, self)._create_stochastic_knodes(include)
        return knodes

    def _create_wfpt_parents_dict(self, knodes):
        wfpt_parents = super(HDDMnn, self)._create_wfpt_parents_dict(knodes)
        return wfpt_parents

    def _create_wfpt_knode(self, knodes):
        wfpt_parents = self._create_wfpt_parents_dict(knodes)
        return Knode(self.wfpt_nn_class, 'wfpt', observed=True, col_name=['nn_response', 'rt'], **wfpt_parents)


def wienernn_like(x, v, sv, a, z, sz, t, st, p_outlier=0):

    wiener_params = {'err': 1e-4, 'n_st': 2, 'n_sz': 2,
                     'use_adaptive': 1,
                     'simps_err': 1e-3,
                     'w_outlier': 0.1}
    wp = wiener_params

    with open("weights.pickle", "rb") as tmp_file:
        weights = pickle.load(tmp_file)
    with open('biases.pickle', 'rb') as tmp_file:
        biases = pickle.load(tmp_file)
    with open('activations.pickle', 'rb') as tmp_file:
        activations = pickle.load(tmp_file)

    nn_response = x['nn_response'].values.astype(int)
    return wiener_like_nn(np.absolute(x['rt'].values), nn_response,activations,weights,biases, v, sv, a, z, sz, t, st, p_outlier=p_outlier, **wp)
Wienernn = stochastic_from_dist('wienernn', wienernn_like)
