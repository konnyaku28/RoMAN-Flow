import os
import csv
import hashlib
import tempfile
from contextlib import contextmanager
from datetime import datetime

import absl.flags as flags
import ml_collections
import wandb


class CsvLogger:
    """CSV logger for logging metrics to a CSV file."""

    def __init__(self, path):
        self.path = path
        self.header = None
        self.file = None
        self.rows = []
        self.disallowed_types = (wandb.Image, wandb.Video, wandb.Histogram)
        if os.path.isfile(self.path) and os.path.getsize(self.path) > 0:
            with open(self.path, newline='') as existing_file:
                reader = csv.DictReader(existing_file)
                if reader.fieldnames:
                    self.header = list(reader.fieldnames)
                    self.rows = [dict(row) for row in reader]

    def _open_writer(self):
        if self.file is not None:
            self.file.close()
        self.file = open(self.path, 'w', newline='')
        writer = csv.DictWriter(self.file, fieldnames=self.header)
        writer.writeheader()
        return writer

    def log(self, row, step):
        row['step'] = step
        filtered_row = {k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)}
        if self.header is None:
            self.header = list(filtered_row.keys())
            writer = self._open_writer()
        else:
            new_keys = [key for key in filtered_row.keys() if key not in self.header]
            if new_keys:
                self.header.extend(new_keys)
                writer = self._open_writer()
                for cached_row in self.rows:
                    writer.writerow(cached_row)
            elif self.file is None:
                writer = self._open_writer()
                for cached_row in self.rows:
                    writer.writerow(cached_row)
            else:
                writer = csv.DictWriter(self.file, fieldnames=self.header)

        self.rows.append(filtered_row)
        writer.writerow(filtered_row)
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()


def get_exp_name(seed):
    """Return the experiment name."""
    exp_name = ''
    exp_name += f'sd{seed:03d}_'
    if 'SLURM_JOB_ID' in os.environ:
        exp_name += f's_{os.environ["SLURM_JOB_ID"]}.'
    if 'SLURM_PROCID' in os.environ:
        exp_name += f'{os.environ["SLURM_PROCID"]}.'
    exp_name += f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    return exp_name


def get_flag_dict():
    """Return the dictionary of flags."""
    flag_dict = {k: getattr(flags.FLAGS, k) for k in flags.FLAGS if '.' not in k}
    for k in flag_dict:
        if isinstance(flag_dict[k], ml_collections.ConfigDict):
            flag_dict[k] = flag_dict[k].to_dict()
    return flag_dict


def format_wandb_tag(tag, max_length=64):
    """Keep a W&B tag within its length limit while preserving uniqueness."""
    tag = str(tag)
    if len(tag) <= max_length:
        return tag

    digest = hashlib.sha1(tag.encode('utf-8')).hexdigest()[:8]
    prefix_length = max_length - len(digest) - 1
    return f'{tag[:prefix_length]}-{digest}'


def _wandb_settings_without_package_inventory():
    """Build settings that avoid scanning installed package metadata."""
    return wandb.Settings(x_save_requirements=False)


@contextmanager
def _skip_wandb_package_inventory():
    """Prevent W&B from traversing every installed ``dist-info`` directory."""
    working_set = getattr(wandb.util, 'working_set', None)
    if working_set is None:
        yield
        return

    wandb.util.working_set = lambda: iter(())
    try:
        yield
    finally:
        wandb.util.working_set = working_set


def setup_wandb(
    entity=None,
    project='project',
    group=None,
    name=None,
    wandb_output_dir=None,
    mode='online',
    config=None,
):
    """Set up Weights & Biases for logging."""
    if wandb_output_dir is None:
        wandb_output_dir = tempfile.mkdtemp()
    tags = [format_wandb_tag(group)] if group is not None else None

    init_kwargs = dict(
        config=config if config is not None else get_flag_dict(),
        project=project,
        entity=entity,
        tags=tags,
        group=group,
        dir=wandb_output_dir,
        name=name,
        settings=_wandb_settings_without_package_inventory(),
        mode=mode,
        save_code=False,
    )

    # Even offline runs inventory every installed package by default. On DLC,
    # a replaced package directory on CPFS/NFS can make that traversal fail
    # with ESTALE and abort all DDP ranks before training starts.
    with _skip_wandb_package_inventory():
        run = wandb.init(**init_kwargs)

    return run
