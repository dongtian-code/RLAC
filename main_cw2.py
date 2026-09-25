"""cw2 entry point for RLAC, with SLURM (requeue and cancel) auto-resume support.

This mirrors the resume/preemption scheme used in `dt_rl/mprl`
(`mp_exp_multiprocessing.py`), scoped down to what RLAC needs:
  - both preemption paths. On `allgpu` (`preemption_mode: requeue`) Slurm puts
    the job back in the queue itself, so the run checkpoints and then quiesces
    at that exact boundary. On `comgpu`/`compgpu` (`preemption_mode: cancel`)
    Slurm just kills the job, so the run submits its own replacement: the reps
    packed into the Slurm job meet at a checkpoint-ready barrier and exactly one
    of them `sbatch`es the original job script again. A preemption proper
    (SIGTERM, SIGKILL ~30 s later) resubmits from the last periodic checkpoint;
    only the wall-time USR1 (300 s ahead) writes a fresh one. The cancel path is
    SimbaV2's `main_cw2.py`, ported as is, so the two repos behave identically.
  - no sampler-subprocess timeout handling -- RLAC trains in a single JAX
    process, unlike mprl's separate sampler subprocess.
  - the same config keys and directory scheme: `auto_resume_from_latest_checkpoint`,
    `resume_dir_name`, `resume_scope_name`, `preemption_mode`, `num_checkpoints`,
    `overwrite_checkpoints`, `strict_checkpoint_config`, `active_run_lock`.

Like in mprl, cw2 itself (any branch) has no restore/resume hooks at all --
`AbstractIterativeExperiment` only calls `save_state` after every `iterate()`.
All of the auto-resume behavior below (checkpoint discovery, active-run lock,
SIGTERM/SIGUSR1 handling, requeue quiescing) is application-level code that has
to run inside `initialize()`/`iterate()`, not something cw2 provides.

Known scope simplification vs. the offline part of `main.py`: on resume, the
online rollout restarts the environment with a fresh `env.reset()` rather than
restoring the exact mid-episode simulator state (qpos/qvel/...). The agent,
replay buffer, RNG streams, and step counters are restored exactly; only the
position within the *current* episode is lost, which starts a new episode.
"""

import errno
import fcntl
import glob
import hashlib
import importlib
import json
import os
import pickle
import random
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from cw2 import cluster_work, cw_error, experiment
from cw2.cw_config import cw_config as cw_config_module
from cw2.cw_data import cw_logging
from tqdm import tqdm

from agents import agents as agent_registry
from envs.env_utils import make_env_and_datasets, make_extra_train_envs
from envs.ogbench_utils import make_ogbench_env_and_datasets
from envs.robomimic_utils import is_robomimic_env
from evaluation import evaluate
from log_utils import CsvLogger, LoggingHelper, setup_wandb
from utils.datasets import MultiEnvReplayBuffer, ReplayBuffer, process_train_dataset
from utils.flax_utils import load_checkpoint, save_agent, save_checkpoint


def _agent_config_from_params(agent_params):
    """Build an agent ConfigDict the same way `config_flags.DEFINE_config_file` does:
    start from the agent module's own `get_config()` defaults (which declare
    `ml_collections.config_dict.placeholder(...)` fields for things like `ob_dims`
    that get filled in later) and merge the YAML's overrides on top. Building the
    ConfigDict directly from the raw YAML dict instead would lose the placeholder
    typing and make those later assignments raise a type error.
    """
    agent_params = dict(agent_params)
    agent_name = agent_params['agent_name']
    module = importlib.import_module(f'agents.{agent_name}')
    cfg = module.get_config()
    # YAML has no tuples, so a swept `value_hidden_dims: [1024, 1024, 1024, 1024]`
    # arrives as a list where get_config() declares a tuple. ConfigDict tolerates
    # that, but the agent keeps its config as a STATIC (non-pytree) field, so the
    # value ends up inside jit's cache key; normalising it to the declared type
    # keeps a swept run byte-identical to the same shape written in get_config().
    for key, value in agent_params.items():
        if isinstance(value, list) and isinstance(cfg.get(key, None), tuple):
            agent_params[key] = tuple(value)
    cfg.update(agent_params)
    return cfg


def _crossed(prev, cur, interval):
    """True when a multiple of `interval` lies in the half-open range (prev, cur].

    The step counters advance by `num_train_envs` at a time, so the plain
    `step % interval == 0` tests would step over their trigger. For a counter
    that advances one at a time this is exactly `cur % interval == 0`.
    """
    if not interval:
        return False
    return (prev // interval) != (cur // interval)


def _bool(value):
    if isinstance(value, str):
        return value.lower() in {'1', 'true', 'yes', 'on'}
    return bool(value)


def _safe_path_component(value):
    raw = str(value).strip()
    digest = hashlib.sha1(raw.encode('utf-8')).hexdigest()[:8]
    safe = re.sub(r'[^A-Za-z0-9._-]+', '_', raw).strip('._-') or 'default'
    return f'{safe[:80]}_{digest}'


def _wandb_run_name(cw_config):
    """A W&B run name that is unique per repetition.

    The previous `sd{seed:03d}` + `s_{SLURM_JOB_ID}` was not: with one seed per
    grid point every run is seed 0, and every repetition packed into the same
    Slurm job shares SLURM_JOB_ID -- so all reps on a node reported the one name
    ("sd000s_24668088"), with nothing in it identifying the task.

    cw2's `_experiment_name` is "<name>__<grid suffix>", i.e. unique per grid
    point, so the task goes in front. The Slurm job id is deliberately dropped:
    it changes on every requeue, which would rename a resumed run each time.
    """
    seed = int(cw_config.get('seed', 0))
    grid_tag = str(cw_config.get('_experiment_name') or '')
    prefix = f"{cw_config.get('name', '')}__"
    if grid_tag.startswith(prefix):
        grid_tag = grid_tag[len(prefix):]
    # Grid values may contain "/" ("metaworld/assembly-v2") and cw2 keeps it in
    # the name; the last segment is the readable part.
    grid_tag = re.sub(r'[^0-9A-Za-z._-]+', '_', grid_tag.rsplit('/', 1)[-1]).strip('_')
    return f'{grid_tag}_sd{seed:03d}' if grid_tag else f'sd{seed:03d}'


def _resume_scope_name(cw_config):
    scope = cw_config.get('resume_scope_name')
    if scope is None:
        scope = cw_config.get('sub_exp_name')
    if scope is None and isinstance(cw_config.get('wandb'), dict):
        scope = cw_config['wandb'].get('group')
    if scope is None:
        return None
    return _safe_path_component(scope)


def _checkpoint_config_fingerprint(cw_config):
    payload = {
        'iterations': cw_config.get('iterations'),
        'seed': cw_config.get('seed'),
        'params': cw_config.get('params'),
        # The resume scope (explicit `resume_scope_name`/`sub_exp_name`, else the
        # wandb group) is part of the identity of a checkpoint lineage. Without it
        # a run relaunched under a new group would silently adopt the old group's
        # checkpoints -- the scope only namespaces the `resume/` subtree, and the
        # `log_*` scan in `_candidate_checkpoint_dirs` reaches the old runs anyway.
        'resume_scope': _resume_scope_name(cw_config),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'), default=repr)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def _replace_path_component(path, predicate, new_component):
    parts = os.path.normpath(path).split(os.sep)
    for i in range(len(parts) - 1, -1, -1):
        if predicate(parts[i]):
            parts[i] = new_component
            break
    else:
        parts.append(new_component)
    if os.path.isabs(path):
        return os.sep + os.path.join(*[p for p in parts if p])
    return os.path.join(*parts)


# Slurm partitions differ in what preemption does. On `allgpu` a preempted job is
# requeued by Slurm; on `comgpu`/`compgpu` it is simply cancelled and nothing
# comes back unless the job submits a replacement itself.
_REQUEUE_MODES = frozenset({'requeue', 'allgpu'})
_CANCEL_MODES = frozenset({'cancel', 'comgpu', 'compgpu'})

# Keys that belong to the preemption scheme but are natural to write in the SLURM
# document (next to `partition`). `prepare_rep_configs` copies them down into each
# repetition config, which is all `initialize()`/`iterate()` ever sees.
_SLURM_PREEMPTION_KEYS = (
    'partition',
    'preemption_mode',
    'auto_resume_from_latest_checkpoint',
    'resume_dir_name',
    'disable_preemption_resubmit',
    'exclude_current_node_on_resubmit',
    'checkpoint_ready_barrier_enabled',
    'checkpoint_ready_barrier_timeout',
    'checkpoint_ready_barrier_poll_interval',
    'checkpoint_ready_barrier_submit_on_timeout',
    'checkpoint_ready_barrier_sigterm_timeout',
    'active_run_lock_wait_seconds',
)


def _get_preemption_mode(cw_config):
    mode = str(cw_config.get('preemption_mode', '') or '').lower()
    if mode:
        return mode
    partition = str(cw_config.get('partition', '') or '').lower()
    if partition == 'allgpu':
        return 'requeue'
    if partition in {'comgpu', 'compgpu'}:
        return 'cancel'
    return ''


def _slurm_marker_job_id():
    """Identify the Slurm allocation the current process belongs to.

    Array jobs need both ids: every array task of one submission shares
    SLURM_ARRAY_JOB_ID, and a marker keyed on that alone would let one task's
    replacement suppress every other task's.
    """
    array_job_id = os.environ.get('SLURM_ARRAY_JOB_ID')
    array_task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    if array_job_id is not None and array_task_id is not None:
        return f'{array_job_id}_{array_task_id}'
    return os.environ.get('SLURM_JOB_ID', 'local')


# Variables that describe the submitting *process*, not the job. `sbatch` exports
# the caller's whole environment by default, and the caller of a cancel-mode
# resubmission is a rep deep inside a running job:
#   - WANDB_SERVICE is the port of the wandb-core service this rep started on the
#     old node. Inherited by the replacement, every rep there tries to attach to
#     it on the new node and dies in wandb.init with "WandbServiceConnectionError:
#     Failed to connect to internal service" -- right after restoring its
#     checkpoint. That is how SimbaV2's 14.Sep MetaWorld comgpu chain broke.
#     wandb's own agent drops it before spawning runs (wandb/wandb_agent.py).
#   - CUDA_VISIBLE_DEVICES and the EGL device ids pin this rep to one GPU of the
#     old node; the replacement's scheduler assigns its own.
_RESUBMIT_ENV_DROP = ('WANDB_SERVICE', 'CUDA_VISIBLE_DEVICES', 'EGL_DEVICE_ID', 'MUJOCO_EGL_DEVICE_ID')


def _resubmission_env():
    env = dict(os.environ)
    for key in _RESUBMIT_ENV_DROP:
        env.pop(key, None)
    return env


def prepare_rep_configs(config_obj: cw_config_module.Config):
    """Resolve seed/resume/timestamp paths for every unfolded repetition.

    Must run after cw2 unfolds `exp_configs` (i.e. after `ClusterWork(...)` is
    constructed) and before `cw.run()` (which creates the on-disk directories
    and re-dumps the resolved YAML). This is the RLAC analogue of mprl's
    `RLExperiment._process_train_rep_config_file`.
    """
    slurm_config = getattr(config_obj, 'slurm_config', None) or {}
    for rep_config in config_obj.exp_configs:
        # cw2 keeps the SLURM document separate from the experiment documents, and
        # only the latter reach the job. Copy the preemption keys across so they
        # can be written next to `partition`, where they belong.
        for key in _SLURM_PREEMPTION_KEYS:
            if key in slurm_config and key not in rep_config:
                rep_config[key] = slurm_config[key]

        base_seed = int(rep_config.get('seed', 0))
        rep_idx = int(rep_config.get('_rep_idx', 0))
        seed = base_seed + rep_idx
        rep_config['seed'] = seed

        # cw2 unfolds repetitions as rep_00, rep_01, ... -- use the concrete
        # seed as the persistent repetition identity instead, so a later
        # supplemental seed launch cannot collide with an earlier checkpoint.
        rep_config['_rep_log_path'] = _replace_path_component(
            rep_config['_rep_log_path'], lambda p: re.fullmatch(r'rep_-?\d+', p) is not None,
            f'rep_{seed:02d}',
        )

        auto_resume = _bool(rep_config.get('auto_resume_from_latest_checkpoint', False))
        if auto_resume:
            resume_dir_name = rep_config.get('resume_dir_name', 'resume')
            basic_path = os.path.abspath(rep_config.get('path'))
            resume_root = os.path.join(basic_path, resume_dir_name)
            scope = _resume_scope_name(rep_config)
            if scope is not None:
                rep_config['resume_scope_name'] = scope
                resume_root = os.path.join(resume_root, scope)
            rep_config['resume_model_dir'] = os.path.join(resume_root, f'rep_{seed:02d}', 'model')
            # Infer from `partition` rather than defaulting to requeue: on a
            # cancel-mode partition a requeue-mode job would checkpoint, quiesce,
            # and then be killed with nothing scheduled to take its place.
            inferred = _get_preemption_mode(rep_config)
            rep_config['preemption_mode'] = inferred or 'requeue'

        # Every run/requeue segment gets its own timestamped scratch log dir.
        # The resume dir above is untouched by the timestamp -- that is what
        # makes checkpoints discoverable across restarts.
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        for key in ('log_path', '_rep_log_path'):
            rep_config[key] = os.path.abspath(_replace_path_component(
                rep_config[key], lambda p: p == 'log' or p.startswith('log_'), f'log_{timestamp}',
            ))


class RLACExperiment(experiment.AbstractIterativeExperiment):
    checkpoint_pattern = re.compile(r'^checkpoint_state_(\d+)$')
    latest_checkpoint_name = 'checkpoint_state'

    # ---------------------------------------------------------------- checkpoint discovery
    @classmethod
    def _checkpoint_metadata(cls, checkpoint_path, fallback_epoch=None):
        try:
            with open(checkpoint_path, 'rb') as f:
                state = pickle.load(f)
        except Exception as error:
            print(f'[checkpoint] Ignoring unreadable checkpoint {checkpoint_path}: {error}', flush=True)
            return None
        if not isinstance(state, dict):
            return None
        extra = state.get('extra', {}) or {}
        epoch = extra.get('num_iterations', fallback_epoch)
        if epoch is None:
            return None
        try:
            epoch = int(epoch)
        except (TypeError, ValueError):
            return None
        return epoch, extra.get('config_fingerprint')

    @classmethod
    def _latest_checkpoint_info(cls, model_dir):
        if model_dir is None or not os.path.isdir(model_dir):
            return None, None
        candidates = []
        latest_path = os.path.join(model_dir, cls.latest_checkpoint_name)
        if os.path.isfile(latest_path):
            metadata = cls._checkpoint_metadata(latest_path)
            if metadata is not None:
                epoch, fingerprint = metadata
                candidates.append((epoch, os.path.getmtime(latest_path), fingerprint))

        suffixed = []
        for name in os.listdir(model_dir):
            m = cls.checkpoint_pattern.match(name)
            if m:
                suffixed.append((int(m.group(1)), os.path.join(model_dir, name)))
        for epoch_guess, path in sorted(suffixed, reverse=True):
            metadata = cls._checkpoint_metadata(path, fallback_epoch=epoch_guess)
            if metadata is None:
                continue
            epoch, fingerprint = metadata
            candidates.append((epoch, os.path.getmtime(path), fingerprint))
            break  # file names encode the epoch; the first readable one is newest

        if not candidates:
            return None, None
        epoch, _, fingerprint = max(candidates, key=lambda item: item[:2])
        return epoch, fingerprint

    @classmethod
    def _checkpoint_mtime(cls, model_dir):
        candidates = [os.path.join(model_dir, cls.latest_checkpoint_name)]
        if os.path.isdir(model_dir):
            for name in os.listdir(model_dir):
                if cls.checkpoint_pattern.match(name):
                    candidates.append(os.path.join(model_dir, name))
        mtimes = [os.path.getmtime(p) for p in candidates if os.path.isfile(p)]
        return max(mtimes) if mtimes else 0.0

    @classmethod
    def _rep_dir_names(cls, cw_config):
        names = []
        for value in (cw_config.get('seed'), cw_config.get('_rep_idx')):
            if value is None:
                continue
            try:
                v = int(value)
            except (TypeError, ValueError):
                continue
            for cand in (f'rep_{v:02d}', f'rep_{v}'):
                if cand not in names:
                    names.append(cand)
        return names

    @classmethod
    def _group_roots_for_checkpoint_search(cls, cw_config, resume_model_dir):
        roots = []
        for key in ('path', '_basic_path'):
            p = cw_config.get(key)
            if p is not None:
                roots.append(os.path.abspath(p))
        for key in ('log_path', '_rep_log_path', 'save_model_dir', 'resume_model_dir'):
            p = cw_config.get(key)
            if p is None:
                continue
            p = os.path.abspath(p)
            parts = os.path.normpath(p).split(os.sep)
            for i in range(len(parts) - 1, -1, -1):
                if parts[i] == 'resume' or parts[i] == 'log' or parts[i].startswith('log_'):
                    roots.append(os.sep + os.path.join(*[x for x in parts[:i] if x]))
                    break
        if resume_model_dir is not None:
            d = os.path.abspath(resume_model_dir)
            for _ in range(5):
                roots.append(d)
                parent = os.path.dirname(d)
                if parent == d:
                    break
                d = parent
        seen = set()
        unique = []
        for r in roots:
            r = os.path.abspath(r)
            if r not in seen and os.path.isdir(r):
                unique.append(r)
                seen.add(r)
        return unique

    @classmethod
    def _candidate_checkpoint_dirs(cls, cw_config, resume_model_dir):
        candidates = []
        if resume_model_dir is not None:
            candidates.append(os.path.abspath(resume_model_dir))
        rep_dir_names = cls._rep_dir_names(cw_config)
        scope = _resume_scope_name(cw_config)
        for root in cls._group_roots_for_checkpoint_search(cw_config, resume_model_dir):
            resume_dir = os.path.join(root, 'resume')
            for rep_dir_name in rep_dir_names:
                if scope is not None:
                    candidates.append(os.path.join(resume_dir, scope, rep_dir_name, 'model'))
                else:
                    candidates.append(os.path.join(resume_dir, rep_dir_name, 'model'))
            # Timestamped run dirs are only searched for an unscoped lineage. They
            # predate the `resume/` scheme and carry no scope of their own, so
            # scanning them under a scope would pull in checkpoints from every
            # other group that ever ran this grid point.
            if scope is not None:
                continue
            for run_dir in [os.path.join(root, 'log'), *glob.glob(os.path.join(root, 'log_*'))]:
                for rep_dir_name in rep_dir_names:
                    candidates.append(os.path.join(run_dir, rep_dir_name, 'model'))
        seen = set()
        unique = []
        for c in candidates:
            c = os.path.abspath(c)
            if c not in seen:
                unique.append(c)
                seen.add(c)
        return unique

    @classmethod
    def _find_latest_checkpoint(cls, cw_config, resume_model_dir):
        """Return (dir, epoch, None, None) if resumable, or (None, None, dir, epoch)
        if the matching run is already complete, or (None, None, None, None)."""
        best = None
        completed = None
        max_epoch = cw_config.get('iterations')
        if max_epoch is not None:
            max_epoch = int(max_epoch)
        strict = _bool(cw_config.get('strict_checkpoint_config', True))
        expected_fp = _checkpoint_config_fingerprint(cw_config) if strict else None
        for model_dir in cls._candidate_checkpoint_dirs(cw_config, resume_model_dir):
            epoch, fingerprint = cls._latest_checkpoint_info(model_dir)
            if epoch is None:
                continue
            if strict and fingerprint is not None and fingerprint != expected_fp:
                print(f'[checkpoint] Ignoring checkpoint with mismatched config fingerprint: {model_dir}', flush=True)
                continue
            key = (epoch, cls._checkpoint_mtime(model_dir))
            if max_epoch is not None and epoch >= max_epoch:
                if completed is None or key > completed[0]:
                    completed = (key, model_dir, epoch)
                continue
            if best is None or key > best[0]:
                best = (key, model_dir, epoch)
        if completed is not None:
            _, model_dir, epoch = completed
            return None, None, model_dir, epoch
        if best is None:
            return None, None, None, None
        _, model_dir, epoch = best
        return model_dir, epoch, None, None

    # ---------------------------------------------------------------- active-run lock
    def _active_run_lock_enabled(self, cw_config):
        if _bool(cw_config.get('disable_active_run_lock', False)):
            return False
        return _bool(cw_config.get(
            'active_run_lock', cw_config.get('auto_resume_from_latest_checkpoint', False)
        ))

    def _active_run_lock_path(self, cw_config):
        if self.resume_model_dir is not None:
            return os.path.join(os.path.dirname(self.resume_model_dir), 'active.lock')
        save_model_dir = getattr(self, 'save_model_dir', None)
        if save_model_dir is not None:
            return os.path.join(os.path.dirname(os.path.abspath(save_model_dir)), 'active.lock')
        return None

    @staticmethod
    def _active_run_lock_wait_seconds(cw_config):
        """How long a held lock is waited on before this rep is skipped.

        0 (skip at once) outside cancel mode. On a cancel-mode partition the
        replacement can start on another node while the preempted rep is still
        inside Slurm's KillWait, and skipping it then would silently drop the run.
        """
        default = 120.0 if _get_preemption_mode(cw_config) in _CANCEL_MODES else 0.0
        return max(float(cw_config.get('active_run_lock_wait_seconds', default)), 0.0)

    def _acquire_active_run_lock(self, cw_config):
        lock_path = self._active_run_lock_path(cw_config)
        if lock_path is None:
            return True
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        wait_seconds = self._active_run_lock_wait_seconds(cw_config)
        deadline = time.monotonic() + wait_seconds
        waiting = False
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EAGAIN):
                    os.close(fd)
                    raise
            if time.monotonic() < deadline:
                if not waiting:
                    print(f'[checkpoint] Active run lock held at {lock_path}; waiting up to {wait_seconds:.0f}s '
                          'for the previous holder to exit.', flush=True)
                    waiting = True
                time.sleep(2.0)
                continue
            with os.fdopen(fd, 'r') as f:
                holder = f.read().strip()
            print(f'[checkpoint] Active run lock held at {lock_path}; skipping duplicate run. Holder: {holder}', flush=True)
            return False
        if waiting:
            print(f'[checkpoint] Acquired active run lock at {lock_path}.', flush=True)
        metadata = dict(
            pid=os.getpid(),
            hostname=socket.gethostname(),
            slurm_job_id=os.environ.get('SLURM_JOB_ID'),
            slurm_array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),
            slurm_array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'),
            started_at=time.strftime('%Y-%m-%d %H:%M:%S'),
        )
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps(metadata, sort_keys=True).encode('utf-8'))
        os.fsync(fd)
        self._active_run_lock_fd = fd
        return True

    def _release_active_run_lock(self):
        fd = getattr(self, '_active_run_lock_fd', None)
        if fd is None:
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
            self._active_run_lock_fd = None

    # ---------------------------------------------------------------- preemption
    def _register_preemption_handlers(self):
        def _request(signum, _frame):
            self._preemption_requested = True
            self._preemption_signal = signum
            print(f'[preemption] Received signal {signum}; will checkpoint after the current step.', flush=True)

        for sig in (signal.SIGTERM, signal.SIGUSR1):
            try:
                signal.signal(sig, _request)
            except (AttributeError, ValueError):
                pass

    def _wait_for_requeue_termination(self):
        """Sit exactly at the USR1 checkpoint boundary until Slurm SIGTERMs us."""
        while self._preemption_signal != signal.SIGTERM:
            signal.pause()
        print('[preemption] Termination received while quiesced; exiting at the saved checkpoint boundary.', flush=True)
        raise cw_error.ExperimentSurrender()

    def _existing_checkpoint_path(self):
        """The newest checkpoint this rep knows to be complete on disk, or None."""
        path = getattr(self, '_last_checkpoint_path', None)
        return path if path is not None and os.path.isfile(path) else None

    def _handle_preemption(self, cw_config):
        preemption_mode = _get_preemption_mode(cw_config) or 'requeue'
        cancel_mode = preemption_mode in _CANCEL_MODES
        # A cancel-mode partition preempts with SIGTERM and follows up with SIGKILL
        # after Slurm's KillWait -- 30 s on Maxwell, with GraceTime 0 on comgpu.
        # That does not reliably cover every rep in the job writing a fresh
        # checkpoint, meeting at the barrier and submitting, and a replacement that
        # never gets submitted loses every run in the job. So on SIGTERM keep the
        # last periodic checkpoint (checkpoint writes are atomic, so it is intact)
        # and go straight to the resubmission. USR1 is the wall-time warning, 300 s
        # ahead (`signal: B:USR1@300`), which leaves room for a fresh checkpoint.
        fast = cancel_mode and self._preemption_signal == signal.SIGTERM
        checkpoint_path = None
        if fast:
            print('[preemption] SIGTERM on a cancel-mode partition: skipping the forced checkpoint.', flush=True)
        else:
            try:
                checkpoint_path = self._save_checkpoint(cw_config, force=True)
            except Exception as error:
                # Keep going: the last good checkpoint is still on disk, and in
                # cancel mode a replacement job is worth submitting even if this
                # write failed -- it will resume from the previous checkpoint.
                print(f'[preemption] Checkpoint save failed: {error}', flush=True)
        if checkpoint_path is None:
            checkpoint_path = self._existing_checkpoint_path()
        self._skip_next_save_state = True

        if cancel_mode:
            # Slurm will not bring this job back, so announce the checkpoint,
            # wait for the other reps in this job, and submit the replacement.
            # This runs for SIGTERM too: on a cancel-mode partition SIGTERM is
            # how the preemption finishes, not a sign that the user cancelled.
            #
            # Drop the active-run lock *before* publishing readiness. The rep
            # that submits only gets past the barrier once every rep has
            # published, so this ordering guarantees no rep is still holding its
            # lock when the replacement job starts -- a rep whose lock is still
            # held would be skipped by the replacement and silently lost.
            # Training is already over at this point, and the checkpoint the
            # replacement will read (fresh, or on SIGTERM the last periodic one)
            # is complete on disk.
            self._release_active_run_lock()
            # 'skipped' when nothing is on disk yet: the replacement then starts
            # this rep from scratch, and the barrier must not wait for a
            # checkpoint that will never appear.
            print(f'[preemption] The replacement resumes this run from '
                  f'{checkpoint_path or "scratch (no checkpoint yet)"}.', flush=True)
            self._publish_checkpoint_ready(
                cw_config, checkpoint_path, status='ready' if checkpoint_path else 'skipped')
            self._resubmit_if_cancel_preemption(cw_config, fast=fast)
            print('[preemption] Cancel-mode handling complete; exiting.', flush=True)
            raise cw_error.ExperimentSurrender({'preempted': True, 'preemption_mode': preemption_mode})

        if self._preemption_signal == signal.SIGTERM:
            print('[preemption] Termination signal checkpoint complete; exiting.', flush=True)
            raise cw_error.ExperimentSurrender()

        if preemption_mode in _REQUEUE_MODES:
            print(
                '[preemption] Requeue-mode checkpoint complete; quiescing until Slurm '
                'requeues or terminates the job.', flush=True,
            )
            self._wait_for_requeue_termination()

        print('[preemption] Checkpoint complete; no preemption_mode configured, continuing.', flush=True)
        self._preemption_requested = False

    # ------------------------------------------------- cancel-mode resubmission
    # On a cancel-mode partition Slurm does not bring a preempted job back, so
    # the job submits its own replacement. Two things have to be coordinated
    # across the reps packed into one Slurm job (`reps_in_parallel`), each of
    # which is a separate process that gets its own SIGUSR1:
    #
    #   1. the barrier -- no replacement may be submitted until *every* rep in
    #      the job has a fresh checkpoint on disk. Otherwise the replacement can
    #      start while a rep is still writing, and the active-run lock would make
    #      the new job skip that rep entirely, silently dropping it.
    #   2. the claim -- exactly one rep may call sbatch, or one preemption turns
    #      into `reps_in_parallel` replacement jobs.
    #
    # Both are files under `<path>/.preemption`, keyed by the Slurm job id.
    def _marker_dir(self, cw_config):
        """A directory shared by every rep in this Slurm job.

        Not `path`: cw2 extends it per grid point, and one Slurm job packs
        several grid points. `_basic_path` is the un-extended root and is
        identical for every rep.
        """
        for key in ('_basic_path', 'path'):
            base = cw_config.get(key)
            if base:
                return os.path.join(os.path.abspath(base), '.preemption')
        return os.path.join(os.getcwd(), '.preemption')

    def _bool_cfg(self, cw_config, key, default):
        return _bool(cw_config.get(key, default))

    def _resubmit_disabled(self, cw_config):
        if os.environ.get('RLAC_DISABLE_PREEMPTION_RESUBMIT', '').lower() in {'1', 'true', 'yes', 'on'}:
            return True
        if self._bool_cfg(cw_config, 'disable_preemption_resubmit', False):
            return True
        # A drop file is the way to stop a self-perpetuating chain of jobs without
        # editing the config or waiting for the current ones to finish.
        marker_dir = self._marker_dir(cw_config)
        job_id = _slurm_marker_job_id()
        for name in ('.disable_preemption_resubmit', f'.disable_preemption_resubmit_{job_id}'):
            if os.path.exists(os.path.join(marker_dir, name)):
                print(f'[preemption] Found {name}; not submitting a replacement.', flush=True)
                return True
        return False

    # --- checkpoint-ready barrier ---
    @staticmethod
    def _barrier_task_id(value):
        return str(int(value)) if isinstance(value, (int, np.integer)) else str(value)

    def _barrier_current_task_id(self, cw_config):
        task_id = cw_config.get('_cw2_job_task_id')
        if task_id is None:
            task_id = cw_config.get('_rep_idx', cw_config.get('seed'))
        if task_id is None:
            raise RuntimeError('checkpoint barrier cannot identify the current task')
        return self._barrier_task_id(task_id)

    def _barrier_expected_task_ids(self, cw_config):
        raw = cw_config.get('_cw2_job_task_ids')
        if raw is None:
            # Repetition ids are not unique once a Slurm task packs several sweep
            # points, so only a single-task job may fall back to them.
            count = int(cw_config.get('_cw2_job_task_count', cw_config.get('reps_per_job', 1)) or 1)
            if count != 1:
                raise RuntimeError(
                    'checkpoint barrier needs _cw2_job_task_ids from cw2 for a job '
                    f'packing {count} runs; update the cw2 checkout'
                )
            return [self._barrier_current_task_id(cw_config)]
        task_ids = [self._barrier_task_id(t) for t in raw]
        if len(set(task_ids)) != len(task_ids):
            raise RuntimeError(f'checkpoint barrier got duplicate task ids: {task_ids}')
        current = self._barrier_current_task_id(cw_config)
        if current not in task_ids:
            raise RuntimeError(
                f'checkpoint barrier: current task {current} is not in the job membership {task_ids}'
            )
        return task_ids

    def _barrier_marker_path(self, cw_config, task_id):
        safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', self._barrier_task_id(task_id))
        return os.path.join(
            self._marker_dir(cw_config),
            f'.checkpoint_ready_{_slurm_marker_job_id()}',
            f'task_{safe[:48]}.json',
        )

    @staticmethod
    def _atomic_write_json(path, payload):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f'{path}.tmp.{os.getpid()}.{time.time_ns()}'
        try:
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(payload, f, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def _publish_checkpoint_ready(self, cw_config, checkpoint_path, status='ready'):
        """Announce that this rep is done with the node.

        `status='ready'` means "checkpoint written, safe to be replaced".
        `status='skipped'` means this rep will not run at all in this job -- it is
        already complete, or another process holds its lock. Publishing that is
        what keeps one skipped rep from stalling the barrier for the whole job,
        which would leave the remaining reps with no replacement submitted.
        """
        if _get_preemption_mode(cw_config) not in _CANCEL_MODES:
            return
        if not self._bool_cfg(cw_config, 'checkpoint_ready_barrier_enabled', True):
            return
        try:
            task_id = self._barrier_current_task_id(cw_config)
        except RuntimeError as error:
            print(f'[checkpoint barrier] Cannot publish readiness: {error}', flush=True)
            return
        payload = {
            'task_id': task_id,
            'job_id': _slurm_marker_job_id(),
            'status': status,
            'checkpoint_path': os.path.abspath(checkpoint_path) if checkpoint_path else None,
            'n_completed': getattr(self, 'n_completed', None),
            'online_step': getattr(self, 'online_step', None),
            'seed': cw_config.get('seed'),
            'hostname': socket.gethostname(),
            'pid': os.getpid(),
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        path = self._barrier_marker_path(cw_config, task_id)
        self._atomic_write_json(path, payload)
        print(f'[checkpoint barrier] task {task_id} {status}: {path}', flush=True)

    def _barrier_ready_task_ids(self, cw_config, task_ids):
        ready = set()
        job_id = _slurm_marker_job_id()
        for task_id in task_ids:
            try:
                with open(self._barrier_marker_path(cw_config, task_id), encoding='utf-8') as f:
                    marker = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                continue
            # Reject a marker left behind by an earlier allocation, and one whose
            # checkpoint has since disappeared.
            status = marker.get('status')
            if status not in {'ready', 'skipped'} or str(marker.get('job_id')) != job_id:
                continue
            if status == 'ready':
                checkpoint_path = marker.get('checkpoint_path')
                if checkpoint_path is None or not os.path.exists(checkpoint_path):
                    continue
            ready.add(self._barrier_task_id(marker.get('task_id')))
        return ready

    def _wait_for_checkpoint_ready_barrier(self, cw_config, fast=False):
        if not self._bool_cfg(cw_config, 'checkpoint_ready_barrier_enabled', True):
            return True
        try:
            task_ids = self._barrier_expected_task_ids(cw_config)
        except RuntimeError as error:
            print(f'[checkpoint barrier] {error}; not resubmitting.', flush=True)
            return False

        if fast:
            # SIGKILL follows SIGTERM within KillWait (30 s) whatever happens
            # here, so not submitting after a timeout would lose every rep in the
            # job. Submitting past a straggler is safe: the straggler writes
            # nothing (the forced checkpoint is skipped on SIGTERM), and the
            # replacement waits for its active-run lock instead of skipping it.
            timeout = float(cw_config.get('checkpoint_ready_barrier_sigterm_timeout', 15.0))
            submit_on_timeout = True
        else:
            timeout = float(cw_config.get('checkpoint_ready_barrier_timeout', 240.0))
            submit_on_timeout = self._bool_cfg(cw_config, 'checkpoint_ready_barrier_submit_on_timeout', False)
        poll = float(cw_config.get('checkpoint_ready_barrier_poll_interval', 0.5))
        if timeout < 0 or poll <= 0:
            print('[checkpoint barrier] Invalid timeout/poll interval; not resubmitting.', flush=True)
            return False

        deadline = time.monotonic() + timeout
        last_missing = None
        while True:
            ready = self._barrier_ready_task_ids(cw_config, task_ids)
            missing = [t for t in task_ids if t not in ready]
            if not missing:
                print(f'[checkpoint barrier] All {len(task_ids)} run(s) in this job are ready.', flush=True)
                return True
            if missing != last_missing:
                print(f'[checkpoint barrier] Waiting for task(s) {missing}; ready={sorted(ready)}.', flush=True)
                last_missing = missing
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                submit_anyway = submit_on_timeout
                print(
                    f'[checkpoint barrier] Timed out after {timeout:.1f}s; missing {missing}; '
                    f'{"submitting anyway" if submit_anyway else "not resubmitting"}.',
                    flush=True,
                )
                return submit_anyway
            time.sleep(min(poll, remaining))

    # --- the replacement command ---
    @staticmethod
    def _path_has_component(path, predicate):
        return any(predicate(part) for part in os.path.normpath(path).split(os.sep))

    def _find_original_sbatch_script(self, cw_config):
        """Locate the sbatch.sh cw2 generated for this submission.

        It sits in the code copy this job is running out of, so walking up from
        this file finds the script that produced exactly this job -- the same
        grid, the same resources, the same code snapshot.
        """
        search_dirs = []
        for base in (os.path.abspath(sys.argv[0]), os.path.abspath(__file__), os.getcwd()):
            d = base if os.path.isdir(base) else os.path.dirname(base)
            for _ in range(12):
                search_dirs.append(d)
                parent = os.path.dirname(d)
                if parent == d:
                    break
                d = parent
        for d in search_dirs:
            candidate = os.path.join(d, 'sbatch.sh')
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
        return None

    def _replacement_exclude_nodes(self, cw_config):
        if not self._bool_cfg(cw_config, 'exclude_current_node_on_resubmit', False):
            return None
        if os.environ.get('SLURM_JOB_ID') is None:
            return None
        node_list = (
            os.environ.get('SLURM_JOB_NODELIST')
            or os.environ.get('SLURM_NODELIST')
            or socket.gethostname().split('.', 1)[0]
        ).strip()
        # Goes straight onto an sbatch command line, so only accept a Slurm
        # hostlist ("max-wng023" / "max-wng[023-025]").
        if not re.fullmatch(r'[A-Za-z0-9_.\-\[\],]+', node_list):
            print(f'[resubmit] Ignoring malformed Slurm node list: {node_list!r}', flush=True)
            return None
        return node_list

    def _resubmit_command(self, cw_config):
        sbatch_script = self._find_original_sbatch_script(cw_config)
        if sbatch_script is None:
            print(
                '[resubmit] Could not find the generated sbatch.sh next to this code copy; '
                'falling back to re-running the launcher with --nocodecopy.', flush=True,
            )
            args = [os.path.abspath(sys.argv[0]), *self._strip_job_args(sys.argv[1:])]
            for flag, aliases in (('-s', ('--slurm',)), ('-o', ('--overwrite',))):
                if flag not in args and not any(a in args for a in aliases):
                    args.append(flag)
            if '--nocodecopy' not in args:
                args.append('--nocodecopy')
            return ' '.join(shlex.quote(part) for part in [sys.executable, *args])

        command = ['sbatch']
        exclude_nodes = self._replacement_exclude_nodes(cw_config)
        if exclude_nodes is not None:
            command.append(f'--exclude={exclude_nodes}')
            print(f'[resubmit] Excluding the current node(s) from the replacement: {exclude_nodes}', flush=True)
        array_task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
        if array_task_id is not None:
            # Resubmit only this array task. Without it the whole array comes
            # back, and every other task's reps would be skipped by the
            # active-run lock or, worse, restarted from their own checkpoints.
            command.append(f'--array={array_task_id}')
        command.append(sbatch_script)
        return ' '.join(shlex.quote(part) for part in command)

    @staticmethod
    def _strip_job_args(args):
        """Drop cw2's `-j/--job <idx>`, which pins the rerun to one job index."""
        stripped = []
        skip = False
        for arg in args:
            if skip:
                skip = False
                continue
            if arg in {'-j', '--job'}:
                skip = True
                continue
            if arg.startswith('--job='):
                continue
            stripped.append(arg)
        return stripped

    def _claim_resubmission(self, cw_config):
        marker_dir = self._marker_dir(cw_config)
        os.makedirs(marker_dir, exist_ok=True)
        marker_path = os.path.join(marker_dir, f'.preemption_resubmitted_{_slurm_marker_job_id()}')
        try:
            # O_EXCL is the whole mechanism: the first rep to get here wins and
            # every other rep in the job sees FileExistsError.
            fd = os.open(marker_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return None
        with os.fdopen(fd, 'w') as f:
            json.dump({'pid': os.getpid(), 'status': 'claimed',
                       'created_at': time.strftime('%Y-%m-%d %H:%M:%S')}, f, sort_keys=True)
        return marker_path

    def _resubmit_if_cancel_preemption(self, cw_config, fast=False):
        if _get_preemption_mode(cw_config) not in _CANCEL_MODES:
            return False
        if self._resubmit_disabled(cw_config):
            print('[preemption] Replacement submission disabled; checkpoint only.', flush=True)
            return False
        if not self._wait_for_checkpoint_ready_barrier(cw_config, fast=fast):
            print('[preemption] Checkpoint-ready barrier not satisfied; not resubmitting.', flush=True)
            return False

        command = self._resubmit_command(cw_config)
        marker_path = self._claim_resubmission(cw_config)
        if marker_path is None:
            print('[preemption] A replacement job was already submitted by another rep in this job.', flush=True)
            return False

        print(f'[preemption] Submitting replacement job: {command}', flush=True)
        result = subprocess.run(command, shell=True, cwd=os.getcwd(), env=_resubmission_env())
        with open(marker_path, 'w') as f:
            json.dump({'pid': os.getpid(), 'command': command, 'returncode': result.returncode,
                       'status': 'submitted' if result.returncode == 0 else 'failed',
                       'finished_at': time.strftime('%Y-%m-%d %H:%M:%S')}, f, sort_keys=True)
        if result.returncode != 0:
            print(f'[preemption] Replacement submission failed (exit {result.returncode}).', flush=True)
            # Release the claim so a later attempt -- or another rep -- can retry.
            try:
                os.remove(marker_path)
            except FileNotFoundError:
                pass
            return False
        return True

    # ---------------------------------------------------------------- checkpoint save/load
    def _checkpoint_extra_state(self, cw_config):
        return {
            'checkpoint_version': 1,
            'num_iterations': self.n_completed,
            'phase': self.phase,
            'offline_step': self.offline_step,
            'online_step': self.online_step,
            'log_step': self.log_step,
            'dataset_idx': getattr(self, 'dataset_idx', 0),
            'python_random_state': random.getstate(),
            'numpy_random_state': np.random.get_state(),
            'online_rng': np.asarray(self.online_rng) if self.online_rng is not None else None,
            'wandb_run_id': self.wandb_run.id if self.wandb_run is not None else None,
            'config_fingerprint': _checkpoint_config_fingerprint(cw_config),
            'runtime': {
                'saved_at_unix': time.time(),
                'hostname': socket.gethostname(),
                'pid': os.getpid(),
                'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
            },
        }

    @staticmethod
    def _link_or_copy(src_path, dst_path):
        if os.path.abspath(src_path) == os.path.abspath(dst_path):
            return
        tmp_dst = f'{dst_path}.tmp.{os.getpid()}'
        try:
            if os.path.exists(tmp_dst):
                os.remove(tmp_dst)
            try:
                os.link(src_path, tmp_dst)
            except OSError:
                shutil.copy2(src_path, tmp_dst)
            os.replace(tmp_dst, dst_path)
        finally:
            if os.path.exists(tmp_dst):
                os.remove(tmp_dst)

    def _periodic_checkpoint_phase(self, cw_config):
        """Offset of this rep's periodic checkpoints within the save interval.

        Every rep packed into one Slurm job would otherwise checkpoint at the same
        iteration and -- running at the same speed -- at the same moment. With the
        full replay buffer in it a checkpoint is several GB, and a rep that is
        writing one cannot act on a signal. A SIGTERM that lands while ALL of them
        are writing leaves nobody to submit the cancel-mode replacement within
        KillWait (30 s on comgpu), and the chain ends. Spread over the interval, at
        most one rep is writing at any time. Where the checkpoints land does not
        change what is trained.
        """
        interval = self.save_model_interval
        raw = cw_config.get('_cw2_job_task_ids')
        current = cw_config.get('_cw2_job_task_id')
        if interval <= 1 or not raw or current is None:
            return 0
        task_ids = [self._barrier_task_id(t) for t in raw]
        current = self._barrier_task_id(current)
        if current not in task_ids:
            return 0
        return (task_ids.index(current) * interval) // len(task_ids)

    def _save_checkpoint(self, cw_config, force=False):
        if self.save_model_dir is None:
            return None
        n = self.n_completed
        should_save = (
            force
            or (n + getattr(self, '_checkpoint_phase', 0)) % self.save_model_interval == 0
            or n >= cw_config['iterations']
        )
        if not should_save:
            return None

        os.makedirs(self.save_model_dir, exist_ok=True)
        file_epoch = None if self.overwrite_checkpoints else n
        name = self.latest_checkpoint_name if file_epoch is None else f'{self.latest_checkpoint_name}_{file_epoch}'
        checkpoint_path = os.path.join(self.save_model_dir, name)

        save_checkpoint(
            checkpoint_path,
            self.agent,
            replay_buffer=self.replay_buffer,
            extra=self._checkpoint_extra_state(cw_config),
        )
        if self.overwrite_checkpoints:
            for entry in os.listdir(self.save_model_dir):
                if self.checkpoint_pattern.match(entry):
                    os.remove(os.path.join(self.save_model_dir, entry))

        if self.resume_model_dir is not None:
            os.makedirs(self.resume_model_dir, exist_ok=True)
            self._link_or_copy(checkpoint_path, os.path.join(self.resume_model_dir, os.path.basename(checkpoint_path)))
            if self.overwrite_checkpoints:
                for entry in os.listdir(self.resume_model_dir):
                    if self.checkpoint_pattern.match(entry) and entry != os.path.basename(checkpoint_path):
                        os.remove(os.path.join(self.resume_model_dir, entry))

        # What a cancel-mode SIGTERM falls back on: the copy the next job reads.
        self._last_checkpoint_path = (
            os.path.join(self.resume_model_dir, os.path.basename(checkpoint_path))
            if self.resume_model_dir is not None else checkpoint_path)

        print(f'[checkpoint] Saved: {checkpoint_path} (n={n}, log_step={self.log_step}, phase={self.phase})', flush=True)
        return checkpoint_path

    # ---------------------------------------------------------------- experiment lifecycle
    def initialize(self, cw_config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        self._skip_next_save_state = False
        self._active_run_lock_fd = None
        self._last_checkpoint_path = None
        self._preemption_requested = False
        self._preemption_signal = None
        # NOT `self.run`: RLACExperiment inherits cw2's
        # AbstractIterativeExperiment, whose `run()` method is the driver that
        # calls `iterate()` -- binding a wandb Run to `self.run` shadows it on the
        # instance and cw2's `self.exp.run(c, r, logger)` then fails with
        # "TypeError: 'Run' object is not callable".
        self.wandb_run = None

        p = cw_config['params']
        random.seed(cw_config['seed'])
        np.random.seed(cw_config['seed'])
        # Env-side seed, kept clear of the agent's. Spaced by 100 so a run's
        # train envs (env_seed + 0..num_train_envs-1) never reach into the next
        # repetition's block; the eval env sits 10000 above, out of both.
        self.env_seed = int(cw_config['seed']) * 100

        self.resume_model_dir = cw_config.get('resume_model_dir', None)
        if self.resume_model_dir is not None:
            self.resume_model_dir = os.path.abspath(self.resume_model_dir)

        auto_resume_enabled = _bool(cw_config.get('auto_resume_from_latest_checkpoint', False))
        if auto_resume_enabled and self._active_run_lock_enabled(cw_config):
            if not self._acquire_active_run_lock(cw_config):
                self._publish_checkpoint_ready(cw_config, None, status='skipped')
                raise cw_error.ExperimentSurrender({'active_run_lock_skipped': True})

        # --- checkpoint discovery -------------------------------------------------
        resume_checkpoint_path = None
        if auto_resume_enabled:
            auto_resume_dir, latest_epoch, completed_dir, completed_epoch = self._find_latest_checkpoint(
                cw_config, self.resume_model_dir,
            )
            if completed_epoch is not None:
                print(
                    f'[checkpoint] Matching checkpoint is already complete at epoch '
                    f'{completed_epoch}/{cw_config["iterations"]}: {completed_dir}; skipping this run.',
                    flush=True,
                )
                self._publish_checkpoint_ready(cw_config, None, status='skipped')
                raise cw_error.ExperimentSurrender({'completed_checkpoint_skipped': True})
            elif latest_epoch is not None:
                resume_checkpoint_path = os.path.join(auto_resume_dir, self.latest_checkpoint_name)
                if not os.path.isfile(resume_checkpoint_path):
                    resume_checkpoint_path = os.path.join(auto_resume_dir, f'{self.latest_checkpoint_name}_{latest_epoch}')
                print(f'[checkpoint] Auto-resuming from {resume_checkpoint_path} at n={latest_epoch}.', flush=True)
            else:
                print('[checkpoint] No checkpoint found; starting from scratch.', flush=True)

        self._register_preemption_handlers()

        # --- checkpointing config ---------------------------------------------------
        if cw_config.get('save_model_dir', None) is not None:
            self.save_model_dir = os.path.abspath(cw_config['save_model_dir'])
        else:
            self.save_model_dir = os.path.join(cw_config['_rep_log_path'], 'model')
        os.makedirs(self.save_model_dir, exist_ok=True)
        self.save_model_interval = max(cw_config['iterations'] // cw_config.get('num_checkpoints', 20), 1)
        self._checkpoint_phase = self._periodic_checkpoint_phase(cw_config)
        self.overwrite_checkpoints = _bool(cw_config.get('overwrite_checkpoints', False))

        # --- env / data ---------------------------------------------------------
        self.dataset_idx = 0
        self.dataset_paths = None
        if p.get('ogbench_dataset_dir') is not None:
            assert p.get('dataset_replace_interval', 1000) != 0
            assert p.get('dataset_proportion', 1.0) == 1.0
            self.dataset_paths = [
                f for f in sorted(glob.glob(f"{p['ogbench_dataset_dir']}/*.npz")) if '-val.npz' not in f
            ]
            self.env, self.eval_env, train_dataset, _ = make_ogbench_env_and_datasets(
                p['env_name'], dataset_path=self.dataset_paths[self.dataset_idx], compact_dataset=False,
            )
        else:
            self.env, self.eval_env, train_dataset, _ = make_env_and_datasets(
                p['env_name'], seed=self.env_seed)

        self.discount = p.get('discount', 0.99)
        self.horizon_length = p['horizon_length']

        # Online data collection can run several independent copies of the train
        # env and step all of them per online step, i.e. collect `num_train_envs`
        # environment samples where the default config collects one. `env` stays
        # the first of them so everything that only needs *an* env (example
        # shapes, OGBench dataset reloading) is untouched at num_train_envs=1.
        self.num_train_envs = int(p.get('num_train_envs', 1))
        if self.num_train_envs < 1:
            raise ValueError(f'num_train_envs must be >= 1, got {self.num_train_envs}.')
        if self.num_train_envs > 1 and p.get('save_all_online_states', False):
            raise ValueError(
                'save_all_online_states records a single env stream and cannot '
                f'describe num_train_envs={self.num_train_envs} interleaved envs.'
            )
        self.train_envs = [self.env] + make_extra_train_envs(
            p['env_name'], self.num_train_envs - 1, seed=self.env_seed + 1)

        def build_train_dataset(ds):
            return process_train_dataset(
                ds,
                dataset_proportion=p.get('dataset_proportion', 1.0),
                is_robomimic=is_robomimic_env(p['env_name']),
                sparse=p.get('sparse', False),
            )

        self._build_train_dataset = build_train_dataset
        if train_dataset is None:
            # No offline dataset for this env (e.g. MetaWorld) -- this is only
            # valid with offline_steps == 0 (checked below). Get example shapes
            # straight from the env instead of from a dataset sample.
            if p.get('offline_steps', 0) != 0:
                raise ValueError(
                    f"env_name={p['env_name']!r} has no offline dataset, so "
                    "offline_steps must be 0 (pure online training)."
                )
            self.train_dataset = None
            example_batch = dict(
                observations=np.asarray(self.env.observation_space.sample(), dtype=np.float32),
                actions=np.asarray(self.env.action_space.sample(), dtype=np.float32),
            )
        else:
            self.train_dataset = build_train_dataset(train_dataset)
            example_batch = self.train_dataset.sample(())
        self.action_dim = example_batch['actions'].shape[-1]
        self._example_transition = dict(
            observations=np.zeros_like(example_batch['observations']),
            actions=np.zeros_like(example_batch['actions']),
            rewards=np.float32(0.0),
            masks=np.float32(1.0),
            terminals=np.float32(0.0),
            next_observations=np.zeros_like(example_batch['observations']),
        )

        agent_cfg = _agent_config_from_params(p['agent'])
        agent_cfg['horizon_length'] = self.horizon_length
        self.agent_cfg = agent_cfg
        agent_class = agent_registry[agent_cfg['agent_name']]
        self.agent = agent_class.create(cw_config['seed'], example_batch['observations'], example_batch['actions'], agent_cfg)

        self.online_rng = jax.random.split(jax.random.PRNGKey(cw_config['seed']), 2)[0]
        self.replay_buffer = None
        self.action_queues = [[] for _ in range(self.num_train_envs)]
        self.obs = [None] * self.num_train_envs
        self.online_extra_data = defaultdict(list)

        # --- resumable bookkeeping (overwritten below if resuming) -----------------
        self.n_completed = 0
        self.phase = 'offline'
        self.offline_step = 0
        self.online_step = 0
        self.log_step = 0
        wandb_run_id = None
        wandb_resume = None

        if resume_checkpoint_path is not None:
            # Always hand load_checkpoint a real buffer object: during the online
            # phase it gets its contents restored from the checkpoint; during the
            # offline phase the checkpoint carries no buffer, so this allocation is
            # just a placeholder that `_transition_to_online` later replaces.
            self.replay_buffer = self._make_replay_buffer(cw_config)
            self.agent, extra = load_checkpoint(resume_checkpoint_path, self.agent, replay_buffer=self.replay_buffer)
            self._last_checkpoint_path = resume_checkpoint_path
            self.n_completed = extra['num_iterations']
            self.phase = extra['phase']
            self.offline_step = extra['offline_step']
            self.online_step = extra['online_step']
            self.log_step = extra['log_step']
            self.dataset_idx = extra.get('dataset_idx', 0)
            random.setstate(extra['python_random_state'])
            np.random.set_state(extra['numpy_random_state'])
            if extra.get('online_rng') is not None:
                self.online_rng = jnp.asarray(extra['online_rng'])
            wandb_run_id = extra.get('wandb_run_id')
            wandb_resume = 'allow'

            if self.phase == 'online':
                self._reset_train_envs()
            else:
                self.replay_buffer = None

        exp_name = _wandb_run_name(cw_config)
        wandb_cfg = cw_config.get('wandb', {}) if isinstance(cw_config.get('wandb'), dict) else {}
        # A checkpoint carries the id of the W&B run it was logging to, so a
        # requeued job continues one curve instead of fragmenting per allocation.
        # But if that run has since been deleted in the W&B UI, wandb.init fails
        # with "CommError: ... 410 ... previously created and deleted; try a new
        # run id" -- which used to kill the job inside initialize() and throw away
        # a perfectly good checkpoint over a logging-only problem. Fall back to a
        # fresh W&B run instead; training still resumes from the checkpoint.
        # Only the resume path is retried, and only once, so a real outage or auth
        # failure still surfaces rather than being swallowed.
        wandb_kwargs = dict(
            project=wandb_cfg.get('project', 'qc'),
            group=wandb_cfg.get('group', cw_config.get('_experiment_name')),
            entity=wandb_cfg.get('entity'),
            name=exp_name,
            config=p,
        )
        try:
            self.wandb_run = setup_wandb(
                run_id=wandb_run_id, resume=wandb_resume, **wandb_kwargs)
        except Exception as exc:
            if wandb_run_id is None:
                raise
            print(
                f'[wandb] Could not resume run {wandb_run_id!r} '
                f'({type(exc).__name__}: {exc}). Starting a fresh W&B run; '
                'training still resumes from the checkpoint.',
                flush=True,
            )
            self.wandb_run = setup_wandb(run_id=None, resume=None, **wandb_kwargs)

        prefixes = ['eval', 'env']
        if p.get('offline_steps', 0) > 0:
            prefixes.append('offline_agent')
        if p.get('online_steps', 0) > 0:
            prefixes.append('online_agent')
        self.logger = LoggingHelper(
            csv_loggers={prefix: CsvLogger(os.path.join(cw_config['_rep_log_path'], f'{prefix}.csv')) for prefix in prefixes},
            wandb_logger=wandb,
        )

        self.progress_bar = tqdm(total=cw_config['iterations'], initial=self.n_completed)

    def _reset_train_envs(self):
        """Reset every train env and drop its pending action chunk.

        Also the resume path: a checkpoint carries the replay buffer but not the
        envs' simulator state, so a requeued run restarts the episodes it was in
        the middle of. Queued actions belong to the episode that is being
        abandoned, so they go too.
        """
        self.obs = [env.reset()[0] for env in self.train_envs]
        self.action_queues = [[] for _ in self.train_envs]

    def _make_replay_buffer(self, cw_config):
        """Build the online replay buffer, split per env when there are several.

        `sample_sequence` reads action chunks out of consecutive buffer slots, so
        each env needs its own contiguous region -- see
        `utils/datasets.py::MultiEnvReplayBuffer`.
        """
        p = cw_config['params']
        buffer_size = p.get('buffer_size', 2000000)
        if self.train_dataset is None:
            # No prior data (e.g. MetaWorld, BoxPushing) -- start from an empty buffer.
            if self.num_train_envs > 1:
                return MultiEnvReplayBuffer.create(
                    self._example_transition, size=buffer_size, num_envs=self.num_train_envs)
            return ReplayBuffer.create(self._example_transition, size=buffer_size)
        if self.num_train_envs > 1:
            raise NotImplementedError(
                'num_train_envs > 1 with an offline dataset is not supported: the '
                'dataset would have to be split across the per-env buffers.'
            )
        return ReplayBuffer.create_from_initial_dataset(
            dict(self.train_dataset), size=max(buffer_size, self.train_dataset.size + 1),
        )

    def _transition_to_online(self, cw_config):
        self.replay_buffer = self._make_replay_buffer(cw_config)
        self._reset_train_envs()
        self.phase = 'online'
        self.online_step = 0

    def _offline_step(self, cw_config):
        p = cw_config['params']
        self.offline_step += 1
        self.log_step += 1
        i = self.offline_step

        if (
            self.dataset_paths is not None
            and p.get('dataset_replace_interval', 1000) != 0
            and i % p['dataset_replace_interval'] == 0
        ):
            self.dataset_idx = (self.dataset_idx + 1) % len(self.dataset_paths)
            print(f'Using new dataset: {self.dataset_paths[self.dataset_idx]}', flush=True)
            train_dataset, _ = make_ogbench_env_and_datasets(
                p['env_name'], dataset_path=self.dataset_paths[self.dataset_idx],
                compact_dataset=False, dataset_only=True, cur_env=self.env,
            )
            self.train_dataset = self._build_train_dataset(train_dataset)

        batch = self.train_dataset.sample_sequence(
            self.agent_cfg['batch_size'], sequence_length=self.horizon_length, discount=self.discount,
        )
        self.agent, offline_info = self.agent.update(batch)

        if i % p['log_interval'] == 0:
            self.logger.log(offline_info, 'offline_agent', step=self.log_step)

        if p.get('save_interval', -1) > 0 and i % p['save_interval'] == 0:
            save_agent(self.agent, cw_config['_rep_log_path'], self.log_step)

        offline_steps = p['offline_steps']
        if i == offline_steps - 1 or (p.get('eval_interval', 0) != 0 and i % p['eval_interval'] == 0):
            eval_info, _, _ = evaluate(
                agent=self.agent, env=self.eval_env, action_dim=self.action_dim,
                num_eval_episodes=p.get('eval_episodes', 50), num_video_episodes=p.get('video_episodes', 0),
                video_frame_skip=p.get('video_frame_skip', 3),
            )
            self.logger.log(eval_info, 'eval', step=self.log_step)

    def _collect_from_env(self, cw_config, env_idx):
        """Step one train env once and store the transition. Returns its `info`."""
        p = cw_config['params']
        env = self.train_envs[env_idx]
        queue = self.action_queues[env_idx]
        ob = self.obs[env_idx]

        self.online_rng, key = jax.random.split(self.online_rng)
        if len(queue) == 0:
            # One agent call per env rather than one batched call over the envs
            # that need a chunk: `sample_actions` is jitted on the observation
            # shape, and a per-step-varying batch size would compile a variant per
            # size. Envs desync after the first episode ends, so a batched call
            # would rarely be full anyway.
            action = self.agent.sample_actions(observations=ob, rng=key)
            for a in np.array(action).reshape(-1, self.action_dim):
                queue.append(a)
        action = queue.pop(0)

        next_ob, int_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if p.get('save_all_online_states', False):
            # Rejected at init for num_train_envs > 1, so this is env 0 alone.
            state = env.get_state()
            self.online_extra_data['steps'].append(self.online_step)
            self.online_extra_data['obs'].append(np.copy(next_ob))
            self.online_extra_data['qpos'].append(np.copy(state['qpos']))
            self.online_extra_data['qvel'].append(np.copy(state['qvel']))
            if 'button_states' in state:
                self.online_extra_data['button_states'].append(np.copy(state['button_states']))

        env_name = p['env_name']
        if 'antmaze' in env_name and ('diverse' in env_name or 'play' in env_name or 'umaze' in env_name):
            int_reward = int_reward - 1.0
        elif is_robomimic_env(env_name):
            int_reward = int_reward - 1.0
        if p.get('sparse', False):
            assert int_reward <= 0.0
            int_reward = (int_reward != 0.0) * -1.0

        transition = dict(
            observations=ob, actions=action, rewards=int_reward, terminals=float(done),
            masks=1.0 - terminated, next_observations=next_ob,
        )
        if self.num_train_envs > 1:
            self.replay_buffer.add_transition(transition, env_idx)
        else:
            self.replay_buffer.add_transition(transition)

        if done:
            self.obs[env_idx], _ = env.reset()
            self.action_queues[env_idx] = []
        else:
            self.obs[env_idx] = next_ob
        return info

    def _online_step(self, cw_config):
        """Collect `num_train_envs` environment samples and train on them.

        Returns the number of environment samples collected, which is what
        `online_step`, `chunk_size`, `online_steps` and every `*_interval` are
        counted in -- so those knobs keep meaning the same thing whether a run
        collects from one env or four, and stay comparable with SimbaV2, which
        also logs against environment steps.
        """
        p = cw_config['params']
        collected = self.num_train_envs
        prev_step = self.online_step
        self.online_step += collected
        self.log_step += collected
        i = self.online_step

        for env_idx in range(self.num_train_envs):
            info = self._collect_from_env(cw_config, env_idx)
            if env_idx == 0:
                env_info = {k: v for k, v in info.items() if k.startswith('distance')}
                self.logger.log(env_info, 'env', step=self.log_step)

        update_info = {}
        if i >= p.get('start_training', 5000):
            utd_ratio = p.get('utd_ratio', 1)
            batch = self.replay_buffer.sample_sequence(
                self.agent_cfg['batch_size'] * utd_ratio, sequence_length=self.horizon_length, discount=self.discount,
            )
            batch = jax.tree.map(lambda x: x.reshape((utd_ratio, self.agent_cfg['batch_size']) + x.shape[1:]), batch)
            self.agent, update_info = self.agent.batch_update(batch)

        # `_crossed` instead of `i % interval == 0`: `i` advances by
        # num_train_envs, so a modulo test would step over its trigger. At
        # num_train_envs=1 the two are the same test.
        if _crossed(prev_step, i, p['log_interval']) and update_info:
            self.logger.log(update_info, 'online_agent', step=self.log_step)

        online_steps = p['online_steps']
        last_step = prev_step < online_steps - 1 <= i
        if last_step or _crossed(prev_step, i, p.get('eval_interval', 0)):
            eval_info, _, _ = evaluate(
                agent=self.agent, env=self.eval_env, action_dim=self.action_dim,
                num_eval_episodes=p.get('eval_episodes', 50), num_video_episodes=p.get('video_episodes', 0),
                video_frame_skip=p.get('video_frame_skip', 3),
            )
            self.logger.log(eval_info, 'eval', step=self.log_step)

        if p.get('save_interval', -1) > 0 and _crossed(prev_step, i, p['save_interval']):
            save_agent(self.agent, cw_config['_rep_log_path'], self.log_step)

        return collected

    def _finish_run(self, cw_config):
        p = cw_config['params']
        for csv_logger in self.logger.csv_loggers.values():
            csv_logger.close()
        if p.get('save_all_online_states', False) and self.online_extra_data['steps']:
            d = self.online_extra_data
            c_data = {
                'steps': np.array(d['steps']),
                'qpos': np.stack(d['qpos'], axis=0),
                'qvel': np.stack(d['qvel'], axis=0),
                'obs': np.stack(d['obs'], axis=0),
            }
            if len(d['button_states']) != 0:
                c_data['button_states'] = np.stack(d['button_states'], axis=0)
            np.savez(os.path.join(cw_config['_rep_log_path'], 'data.npz'), **c_data)
        if self.wandb_run is not None and self.wandb_run.url:
            with open(os.path.join(cw_config['_rep_log_path'], 'token.tk'), 'w') as f:
                f.write(self.wandb_run.url)

    def iterate(self, cw_config: dict, rep: int, n: int) -> dict:
        p = cw_config['params']
        chunk_size = p.get('chunk_size', p.get('log_interval', 5000))

        steps_done = 0
        while steps_done < chunk_size and self.phase != 'done':
            if self.phase == 'offline':
                if self.offline_step >= p.get('offline_steps', 0):
                    # Matches main.py: the offline for-loop is a no-op when
                    # offline_steps == 0, so transition without counting a step.
                    self._transition_to_online(cw_config)
                    continue
                self._offline_step(cw_config)
                steps_done += 1
                # Check right away (not just at the top of the next pass) so a
                # boundary landing exactly on the last step of a chunk doesn't
                # need a whole extra `iterate()` call to be noticed.
                if self.offline_step >= p.get('offline_steps', 0):
                    self._transition_to_online(cw_config)
            elif self.phase == 'online':
                if self.online_step >= p.get('online_steps', 0):
                    self.phase = 'done'
                    self._finish_run(cw_config)
                    continue
                steps_done += self._online_step(cw_config)
                if self.online_step >= p.get('online_steps', 0):
                    self.phase = 'done'
                    self._finish_run(cw_config)

        # `n` is cw2's per-process loop index: AbstractIterativeExperiment.run()
        # always iterates `range(cw_config["iterations"])` from 0, so after a
        # requeue it restarts at 0. Assigning `n + 1` here would throw away the
        # value restored from the checkpoint (and make checkpoint_state_<n> names
        # go backwards); count cumulatively instead.
        self.n_completed += 1
        self.progress_bar.update(1)

        if self._preemption_requested:
            self._handle_preemption(cw_config)

        return {}

    def save_state(self, cw_config: dict, rep: int, n: int) -> None:
        if self._skip_next_save_state:
            self._skip_next_save_state = False
            return
        self._save_checkpoint(cw_config)

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        self._release_active_run_lock()


if __name__ == '__main__':
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
        os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']

    cw = cluster_work.ClusterWork(RLACExperiment)
    prepare_rep_configs(cw.config)
    cw.run()
