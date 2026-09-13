"""cw2 entry point for RLAC, with SLURM (allgpu/requeue) auto-resume support.

This mirrors the resume/preemption scheme used in `dt_rl/mprl`
(`mp_exp_multiprocessing.py`), scoped down to what RLAC needs:
  - only the `requeue` preemption path (no `cancel`-mode multi-task barrier,
    no sampler-subprocess timeout handling -- RLAC trains in a single JAX
    process, unlike mprl's separate sampler subprocess).
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
import shutil
import signal
import socket
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
from envs.env_utils import make_env_and_datasets
from envs.ogbench_utils import make_ogbench_env_and_datasets
from envs.robomimic_utils import is_robomimic_env
from evaluation import evaluate
from log_utils import CsvLogger, LoggingHelper, setup_wandb
from utils.datasets import ReplayBuffer, process_train_dataset
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
    cfg.update(agent_params)
    return cfg


def _bool(value):
    if isinstance(value, str):
        return value.lower() in {'1', 'true', 'yes', 'on'}
    return bool(value)


def _safe_path_component(value):
    raw = str(value).strip()
    digest = hashlib.sha1(raw.encode('utf-8')).hexdigest()[:8]
    safe = re.sub(r'[^A-Za-z0-9._-]+', '_', raw).strip('._-') or 'default'
    return f'{safe[:80]}_{digest}'


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


def prepare_rep_configs(config_obj: cw_config_module.Config):
    """Resolve seed/resume/timestamp paths for every unfolded repetition.

    Must run after cw2 unfolds `exp_configs` (i.e. after `ClusterWork(...)` is
    constructed) and before `cw.run()` (which creates the on-disk directories
    and re-dumps the resolved YAML). This is the RLAC analogue of mprl's
    `RLExperiment._process_train_rep_config_file`.
    """
    for rep_config in config_obj.exp_configs:
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
            rep_config.setdefault('preemption_mode', 'requeue')

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

    def _acquire_active_run_lock(self, cw_config):
        lock_path = self._active_run_lock_path(cw_config)
        if lock_path is None:
            return True
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno not in (errno.EACCES, errno.EAGAIN):
                os.close(fd)
                raise
            with os.fdopen(fd, 'r') as f:
                holder = f.read().strip()
            print(f'[checkpoint] Active run lock held at {lock_path}; skipping duplicate run. Holder: {holder}', flush=True)
            return False
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

    # ---------------------------------------------------------------- preemption (requeue only)
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

    def _handle_preemption(self, cw_config):
        self._save_checkpoint(cw_config, force=True)
        self._skip_next_save_state = True
        preemption_mode = str(cw_config.get('preemption_mode', 'requeue')).lower()
        if self._preemption_signal == signal.SIGTERM:
            print('[preemption] Termination signal checkpoint complete; exiting.', flush=True)
            raise cw_error.ExperimentSurrender()
        if preemption_mode in {'requeue', 'allgpu'}:
            print(
                '[preemption] Requeue-mode checkpoint complete; quiescing until Slurm '
                'requeues or terminates the job.', flush=True,
            )
            self._wait_for_requeue_termination()
        print('[preemption] Checkpoint complete; no preemption_mode configured, continuing.', flush=True)
        self._preemption_requested = False

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

    def _save_checkpoint(self, cw_config, force=False):
        if self.save_model_dir is None:
            return None
        n = self.n_completed
        should_save = (
            force
            or n % self.save_model_interval == 0
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

        print(f'[checkpoint] Saved: {checkpoint_path} (n={n}, log_step={self.log_step}, phase={self.phase})', flush=True)
        return checkpoint_path

    # ---------------------------------------------------------------- experiment lifecycle
    def initialize(self, cw_config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        self._skip_next_save_state = False
        self._active_run_lock_fd = None
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

        self.resume_model_dir = cw_config.get('resume_model_dir', None)
        if self.resume_model_dir is not None:
            self.resume_model_dir = os.path.abspath(self.resume_model_dir)

        auto_resume_enabled = _bool(cw_config.get('auto_resume_from_latest_checkpoint', False))
        if auto_resume_enabled and self._active_run_lock_enabled(cw_config):
            if not self._acquire_active_run_lock(cw_config):
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
            self.env, self.eval_env, train_dataset, _ = make_env_and_datasets(p['env_name'])

        self.discount = p.get('discount', 0.99)
        self.horizon_length = p['horizon_length']

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
        self.action_queue = []
        self.ob = None
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
            buffer_size = p.get('buffer_size', 2000000)
            if self.train_dataset is None:
                self.replay_buffer = ReplayBuffer.create(self._example_transition, size=buffer_size)
            else:
                self.replay_buffer = ReplayBuffer.create_from_initial_dataset(
                    dict(self.train_dataset), size=max(buffer_size, self.train_dataset.size + 1),
                )
            self.agent, extra = load_checkpoint(resume_checkpoint_path, self.agent, replay_buffer=self.replay_buffer)
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
                self.ob, _ = self.env.reset()
            else:
                self.replay_buffer = None

        exp_name = f"sd{cw_config['seed']:03d}"
        if os.environ.get('SLURM_JOB_ID'):
            exp_name += f"s_{os.environ['SLURM_JOB_ID']}"
        wandb_cfg = cw_config.get('wandb', {}) if isinstance(cw_config.get('wandb'), dict) else {}
        self.wandb_run = setup_wandb(
            project=wandb_cfg.get('project', 'qc'),
            group=wandb_cfg.get('group', cw_config.get('_experiment_name')),
            entity=wandb_cfg.get('entity'),
            name=exp_name,
            run_id=wandb_run_id,
            resume=wandb_resume,
            config=p,
        )

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

    def _transition_to_online(self, cw_config):
        p = cw_config['params']
        buffer_size = p.get('buffer_size', 2000000)
        if self.train_dataset is None:
            # No prior data (e.g. MetaWorld) -- start from an empty buffer.
            self.replay_buffer = ReplayBuffer.create(self._example_transition, size=buffer_size)
        else:
            self.replay_buffer = ReplayBuffer.create_from_initial_dataset(
                dict(self.train_dataset), size=max(buffer_size, self.train_dataset.size + 1),
            )
        self.ob, _ = self.env.reset()
        self.action_queue = []
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

    def _online_step(self, cw_config):
        p = cw_config['params']
        self.online_step += 1
        self.log_step += 1
        i = self.online_step

        self.online_rng, key = jax.random.split(self.online_rng)
        if len(self.action_queue) == 0:
            action = self.agent.sample_actions(observations=self.ob, rng=key)
            for a in np.array(action).reshape(-1, self.action_dim):
                self.action_queue.append(a)
        action = self.action_queue.pop(0)

        next_ob, int_reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        if p.get('save_all_online_states', False):
            state = self.env.get_state()
            self.online_extra_data['steps'].append(i)
            self.online_extra_data['obs'].append(np.copy(next_ob))
            self.online_extra_data['qpos'].append(np.copy(state['qpos']))
            self.online_extra_data['qvel'].append(np.copy(state['qvel']))
            if 'button_states' in state:
                self.online_extra_data['button_states'].append(np.copy(state['button_states']))

        env_info = {k: v for k, v in info.items() if k.startswith('distance')}
        self.logger.log(env_info, 'env', step=self.log_step)

        env_name = p['env_name']
        if 'antmaze' in env_name and ('diverse' in env_name or 'play' in env_name or 'umaze' in env_name):
            int_reward = int_reward - 1.0
        elif is_robomimic_env(env_name):
            int_reward = int_reward - 1.0
        if p.get('sparse', False):
            assert int_reward <= 0.0
            int_reward = (int_reward != 0.0) * -1.0

        transition = dict(
            observations=self.ob, actions=action, rewards=int_reward, terminals=float(done),
            masks=1.0 - terminated, next_observations=next_ob,
        )
        self.replay_buffer.add_transition(transition)

        if done:
            self.ob, _ = self.env.reset()
            self.action_queue = []
        else:
            self.ob = next_ob

        update_info = {}
        if i >= p.get('start_training', 5000):
            utd_ratio = p.get('utd_ratio', 1)
            batch = self.replay_buffer.sample_sequence(
                self.agent_cfg['batch_size'] * utd_ratio, sequence_length=self.horizon_length, discount=self.discount,
            )
            batch = jax.tree.map(lambda x: x.reshape((utd_ratio, self.agent_cfg['batch_size']) + x.shape[1:]), batch)
            self.agent, update_info = self.agent.batch_update(batch)

        if i % p['log_interval'] == 0 and update_info:
            self.logger.log(update_info, 'online_agent', step=self.log_step)

        online_steps = p['online_steps']
        if i == online_steps - 1 or (p.get('eval_interval', 0) != 0 and i % p['eval_interval'] == 0):
            eval_info, _, _ = evaluate(
                agent=self.agent, env=self.eval_env, action_dim=self.action_dim,
                num_eval_episodes=p.get('eval_episodes', 50), num_video_episodes=p.get('video_episodes', 0),
                video_frame_skip=p.get('video_frame_skip', 3),
            )
            self.logger.log(eval_info, 'eval', step=self.log_step)

        if p.get('save_interval', -1) > 0 and i % p['save_interval'] == 0:
            save_agent(self.agent, cw_config['_rep_log_path'], self.log_step)

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
                self._online_step(cw_config)
                steps_done += 1
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
