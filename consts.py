import os
from os.path import join as pjoin

paths = {
    'pdgam' : {
        'data_root': pjoin('/mnt/Ndrive/AMBIENT/Vida/Unified_Motion_DB/', 'PDGaM'),
        'data_folder': 'HumanML3Drep/FROM_WHAM/25fps_score3adjusted',
    },

    'gaitgen' : {
        'data_root': '/mnt/vida/Work/AMBIENT/Code/PD_From_GaitGen_PDmodel/data/gaitgen',
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
