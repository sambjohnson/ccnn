import os
import datetime
import configparser
from dlutil.utils import make_agg_dict

# ---- GLOBAL VARIABLES ---- #

TRAINING_PLAN_SHORT = [
    dict(lr=0.00375, gamma=0.9, num_batches=int(1e5), bce_weight=0.33)
]
TRAINING_PLAN_3_STAGE = [
    dict(lr=0.00375, gamma=0.9, num_batches=int(1e4), bce_weight=0.67),
    dict(lr=0.00250, gamma=0.9, num_batches=int(1e4), bce_weight=0.33),
    dict(lr=0.00125, gamma=0.9, num_batches=int(1e4), bce_weight=0.00)
]

# ---- CLASS FUNCTIONS ---- #


class Config(object):
    """
        Training configuration for running train_run.py.
        Its only function is to consolidate different data types
        used to specify training options and parameters into a single object,
        for convenience.

        Users can change the default values as required.

        Arguments:
            config_file: (optional) the filepath to a config file that
                can be read by configparser. If supplied, values read
                from this file will overwrite default values when
                config is constructed.

        Returns:
            the config object.
    """
    def __init__(self, config_file):

        parser = configparser.ConfigParser()
        if config_file is not None:
            parser.read(config_file)



        # training options
        self.train = True
        self.reload_parameters = True
        self.reload_optimizer_state = True
        self.warmstart = True

        # training parameters
        self.batch_size = 32
        self.num_workers_in_dataloaders = 3  # number of cpu's to parallelize dl
        self.pin_memory = True  # whether to pin memory in dl

        # training plan (learning rate, number of epochs / batches...)
        self.training_plan = TRAINING_PLAN_SHORT

        # directories
        date = str(datetime.datetime.now().date())
        self.model_load_dir = './models/model_2022-10-12'
        self.model_save_dir = f'./models/model_{date}'
        os.makedirs(self.model_save_dir, exist_ok=True)

        self.base_data_dir = '/scratch/groups/' \
                             'jyeatman/samjohns-projects/data/atlas/ots-sulc'
        self.data_prefix = 'ots'
        xsubdir = 'xs'
        ysubdir = 'ys'
        xdir = os.path.join(self.base_data_dir, xsubdir)
        ydir = os.path.join(self.base_data_dir, ysubdir)
        self.xdir = xdir
        self.ydir = ydir
        self.data_prefix = 'ots'
        self.trn_csv_fpath = f'{self.base_data_dir}/' \
                             f'{self.data_prefix}_sulc_mapping_trn.csv'
        self.val_csv_fpath = f'{self.base_data_dir}/' \
                             f'{self.data_prefix}_sulc_mapping_val.csv'

        self.out_channels = 7

        frg_agg_dict = make_agg_dict(n_total=self.out_channels,
                                     n_distinct=1)  # cortex vs. background
        nil_agg_dict = make_agg_dict(n_total=self.out_channels,
                                     n_distinct=self.out_channels)
        ots_agg_dict = make_agg_dict(n_total=self.out_channels,
                                     n_distinct=2)  # OTS vs. ctx

        self.agg_dict = ots_agg_dict
