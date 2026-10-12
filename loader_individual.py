import numpy as np
from glob import glob
from torch.utils.data import Dataset

nv_sphere = [12, 42, 162, 642, 2562, 10242, 40962, 163842]


class S2D3DSegLoaderSingle(Dataset):
    """Data loader for a single subject at inference time."""

    def __init__(self, data_dir, sp_level, in_ch, hemi):
        self.in_ch = len(in_ch)
        self.nv = nv_sphere[sp_level]

        flist = []
        file_format = data_dir + '/*.' + hemi + '.*.dat'
        flist += sorted(glob(file_format))

       
        flist = [flist[_x] for _x in [[x.split('.')[-2] for x in flist].index(a) for a in in_ch]]

        # final list
        self.flist = [flist]

    def __len__(self):
        return len(self.flist)

    def __getitem__(self, idx):
        subj = self.flist[idx]
        data = np.stack([np.fromfile(feat, count=self.nv, dtype=np.double) for feat in subj])
        return data.astype(np.float32)
