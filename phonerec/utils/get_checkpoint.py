import torch
import datetime
import numpy as np
import os
from pathlib import Path


# local_dir: Model.model_path
def get_checkpoint(request, mode=None, nsml_dir='download', local_dir=None):
    import json
    from . hparams import Ops

    checkpoint = {}
    protocol, target, state = parse_protocol(request)

    if protocol == 'local':
        assert local_dir is not None
        local_dir = Path(local_dir)
        if target is not None:
            local_dir = local_dir.parent / target

        hparams_path = local_dir / 'hparams.json'
        with open(hparams_path, 'r') as f:
            checkpoint['hparams'] = Ops(json.load(f))
        path = local_dir / 'snapshots' / (state + '_model.pt')
        checkpoint['model'] = torch.load(path, map_location=device)['model']
        path = local_dir / 'snapshots' / (state + '_opt.pt')
        checkpoint['optimizer'] = torch.load(path, map_location=device)['optimizer']

    else:
        print('Invalid Protocol')

    if mode is None:
        return checkpoint
    assert mode in checkpoint
    if mode == 'hparams':
        if type(checkpoint[mode]) is str:
            return Ops(json.loads(checkpoint[mode]))
    return checkpoint[mode]
