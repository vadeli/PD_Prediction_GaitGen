import os
from os.path import join as pjoin

paths = {
    'pdgam' : {
        'data_root': './data/pdgam',
        'data_folder': 'HumanML3Drep',
    },

    'gaitgen' : {
        'data_root': './data/gaitgen',
        'data_folder': '',
        'file_name': '',
    },

    'tri' : {
        'data_root': pjoin('/mnt/Ndrive/AMBIENT/Vida/Unified_Motion_DB/', 'TRI'),
        'data_folder': 'HumanML3Drep_30fps',
    },

    'tri_pd' : {
        'data_root': pjoin('/mnt/Ndrive/AMBIENT/Vida/Unified_Motion_DB/', 'TRI_PD'),
        'data_folder': 'HumanML3Drep_30fps',
    },
}


def set_gaitgen_file_name(file_name):
    """
    Set the file name for the gaitgen dataset.
    """
    paths['gaitgen']['file_name'] = file_name
