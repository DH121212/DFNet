

import os
from data import srdata


class FLIR(srdata.SRData):
    def __init__(self, args, name='FLIR_decoded', train=True, benchmark=False):
        super(FLIR, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, data_dir):
        super(FLIR, self)._set_filesystem(data_dir)
        self.dir_hr = os.path.join(self.apath, 'FLIR_train_HR')
        self.dir_lr = os.path.join(self.apath, 'FLIR_train_LR_bicubic')


