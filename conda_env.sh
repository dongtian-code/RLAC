#!/usr/bin/env bash
set -Eeuo pipefail

# Sets up a user-owned Conda environment for RLAC. Everything below lives
# inside that env (conda create/activate, then pip installs), so nothing
# needs system-wide / sudo write access -- this is what avoids the permission
# issues a global `pip install` can hit on a shared machine or cluster.

ENV_NAME="${ENV_NAME:-rlac}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
RLAC_CUDA="${RLAC_CUDA:-auto}"
RLAC_SKIP_EXTERNAL_DEPS="${RLAC_SKIP_EXTERNAL_DEPS:-0}"
RLAC_WITH_METAWORLD="${RLAC_WITH_METAWORLD:-0}"
RLAC_WITH_D4RL="${RLAC_WITH_D4RL:-0}"
RLAC_WITH_BOXPUSHING="${RLAC_WITH_BOXPUSHING:-0}"
RLAC_DEPS_DIR="${RLAC_DEPS_DIR:-}"
RLAC_SKIP_VERIFY="${RLAC_SKIP_VERIFY:-0}"

read -r -a CONDA_CHANNEL_ARGS <<< "${CONDA_CHANNEL_ARGS:---override-channels -c conda-forge}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
REPO_PARENT="$(cd -- "$REPO_ROOT/.." && pwd)"
DEFAULT_DEPS_DIR="$REPO_PARENT/.deps"
DEPS_DIR="${RLAC_DEPS_DIR:-$DEFAULT_DEPS_DIR}"

log() {
    printf '\n[rlac setup] %s\n' "$*"
}

warn() {
    printf '\n[rlac setup warning] %s\n' "$*" >&2
}

die() {
    printf '\n[rlac setup error] %s\n' "$*" >&2
    exit 1
}

run() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    "$@"
}

on_error() {
    local line="$1"
    die "Setup failed near line ${line}. Fix the error above and rerun bash conda_env.sh."
}
trap 'on_error "$LINENO"' ERR

require_command() {
    local command_name="$1"
    local install_hint="$2"

    if ! command -v "$command_name" >/dev/null 2>&1; then
        die "$command_name is required but was not found. ${install_hint}"
    fi
}

init_conda() {
    require_command conda "Install Miniconda or Anaconda, then open a new terminal so conda is on PATH."

    local conda_base
    conda_base="$(conda info --base 2>/dev/null)" || die "Could not locate the Conda base installation."

    if [[ -f "$conda_base/etc/profile.d/conda.sh" ]]; then
        # shellcheck source=/dev/null
        source "$conda_base/etc/profile.d/conda.sh"
    else
        eval "$(conda shell.bash hook)"
    fi
}

conda_env_exists() {
    conda env list | awk -v env="$ENV_NAME" '$1 == env { found = 1 } END { exit !found }'
}

create_or_update_env() {
    if conda_env_exists; then
        log "Conda environment '$ENV_NAME' already exists; reusing it."
    else
        log "Creating Conda environment '$ENV_NAME' with Python $PYTHON_VERSION."
        run conda create "${CONDA_CHANNEL_ARGS[@]}" -y -n "$ENV_NAME" "python=$PYTHON_VERSION" pip setuptools wheel
    fi

    log "Activating Conda environment '$ENV_NAME' for this setup run."
    run conda activate "$ENV_NAME"

    if [[ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]]; then
        die "Expected Conda environment '$ENV_NAME', but active environment is '${CONDA_DEFAULT_ENV:-none}'."
    fi
}

# JAX's CUDA support ships as plain pip wheels (jax-cuda12-plugin /
# jax-cuda12-pjrt, already pinned in requirements.txt) -- unlike PyTorch there
# is no separate --index-url/conda channel dance. A CUDA 12-capable NVIDIA
# driver (>=525) on the host is still required; the wheels bundle their own
# CUDA runtime, so no system/conda CUDA toolkit install is needed either way.
resolve_use_cuda() {
    case "$RLAC_CUDA" in
        1|true|TRUE|cuda|CUDA|gpu|GPU)
            echo 1 ;;
        0|false|FALSE|cpu|CPU)
            echo 0 ;;
        auto|AUTO|"")
            if [[ "$(uname -s)" == "Linux" ]] && command -v nvidia-smi >/dev/null 2>&1; then
                echo 1
            else
                echo 0
            fi
            ;;
        *)
            die "Unsupported RLAC_CUDA value '$RLAC_CUDA'. Use auto, 1, or 0." ;;
    esac
}

install_base_python_deps() {
    log "Installing base Python packaging tools."
    run python -m pip install --upgrade pip setuptools wheel

    local use_cuda
    use_cuda="$(resolve_use_cuda)"

    if [[ "$use_cuda" == "1" ]]; then
        log "Installing Python dependencies from requirements.txt (CUDA 12 JAX included)."
        run python -m pip install -r "$REPO_ROOT/requirements.txt"
    else
        log "Installing Python dependencies from requirements.txt (CPU-only JAX; skipping jax-cuda12-plugin/pjrt)."
        local cpu_requirements
        cpu_requirements="$(mktemp)"
        grep -Ev '^(jax-cuda12-plugin|jax-cuda12-pjrt)==' "$REPO_ROOT/requirements.txt" > "$cpu_requirements"
        run python -m pip install -r "$cpu_requirements"
        rm -f "$cpu_requirements"
    fi

    # Only imported lazily by envs/robomimic_utils.py's own top-level `import
    # robomimic...` lines, but main.py/main_cw2.py import that module
    # unconditionally (for is_robomimic_env), so this is required just to
    # start the codebase, not only to actually train on RoboMimic envs.
    # Note: robomimic's own setup.py pulls in a fair amount (h5py, imageio,
    # and historically torch) -- this can take a while.
    log "Installing robomimic (required at import time by main.py/main_cw2.py, even if you don't train on RoboMimic envs)."
    run python -m pip install robomimic h5py imageio
}

clone_or_update_repo() {
    local name="$1"
    local url="$2"
    local branch="$3"
    local target="$DEPS_DIR/$name"
    local current_url

    if [[ -e "$target" && ! -d "$target/.git" ]]; then
        die "Dependency path exists but is not a Git checkout: $target"
    fi

    if [[ -d "$target/.git" ]]; then
        current_url="$(git -C "$target" remote get-url origin 2>/dev/null || true)"
        if [[ -z "$current_url" ]]; then
            log "Adding origin '$url' for dependency '$name'."
            run git -C "$target" remote add origin "$url"
        elif [[ "$current_url" != "$url" ]]; then
            log "Updating dependency '$name' origin from '$current_url' to '$url'."
            run git -C "$target" remote set-url origin "$url"
        fi
        log "Updating dependency '$name'."
        run git -C "$target" fetch --depth 1 origin "$branch"
        run git -C "$target" checkout "$branch"
        run git -C "$target" pull --ff-only origin "$branch"
    else
        log "Cloning dependency '$name'."
        run git clone --branch "$branch" --single-branch --depth 1 "$url" "$target"
    fi

    log "Installing dependency '$name' in editable mode."
    run python -m pip install -e "$target"
}

install_external_repos() {
    if [[ "$RLAC_SKIP_EXTERNAL_DEPS" == "1" ]]; then
        warn "Skipping external GitHub dependencies (cw2, and MetaWorld/D4RL/fancy_gym if requested) because RLAC_SKIP_EXTERNAL_DEPS=1."
        return
    fi

    require_command git "Install Git, then rerun this script."
    mkdir -p "$DEPS_DIR"

    # cw2 (dt_branch): the SLURM/allgpu-requeue auto-resume workflow in
    # main_cw2.py depends on this specific fork/branch -- the upstream cw2
    # package has no such resume support.
    clone_or_update_repo "cw2" "git@github.com:DongTian95/cw2.git" "dt_branch"

    if [[ "$RLAC_WITH_METAWORLD" == "1" ]]; then
        # MetaWorld 3.x registers its own gymnasium ids on import, so this is
        # the only MetaWorld dependency -- envs/metaworld_utils.py goes through
        # "Meta-World/goal_observable" directly. Do NOT add fancy_gymnasium back:
        # it targets gymnasium 0.29 (subclasses the EnvCompatibility wrapper
        # gymnasium 1.0 removed) and pins mujoco==2.3.3, which downgrades mujoco
        # out from under metaworld, dm_control and ogbench.
        clone_or_update_repo "Metaworld" "git@github.com:dongtian-code/Metaworld.git" "dt_branch"
        warn "MetaWorld pulls numpy back below 2 and pins mujoco==3.3.0, while requirements.txt asks for numpy==2.2.5 / mujoco==3.3.1. jax, dm_control and ogbench all work with numpy 1.26.4 + mujoco 3.3.0, but check those versions first if anything fails at import."
    fi

    if [[ "$RLAC_WITH_BOXPUSHING" == "1" ]]; then
        # fancy_gym, for the BoxPushing envs (envs/box_pushing_utils.py).
        #
        # --no-deps is MANDATORY: fancy_gym pins mujoco==2.3.3 and
        # gymnasium>=0.26, and resolving those downgrades mujoco out from under
        # metaworld, dm_control and ogbench. Nothing of fancy_gym is imported
        # either -- its package __init__ subclasses
        # gymnasium.wrappers.EnvCompatibility, which gymnasium 1.0 removed, so
        # `import fancy_gym` cannot run here at all. box_pushing_utils.py loads
        # the single env module out of the installed tree in isolation, which is
        # why the package still has to be *installed* (or at least importable)
        # even though it is never imported as a whole.
        # Installed NON-editable into this env's site-packages rather than
        # through $DEPS_DIR: that checkout is shared with dt_rl, whose jobs run
        # against it live on mujoco 2.3.3, and clone_or_update_repo would
        # fetch/checkout/pull it out from under them.
        log "Installing fancy_gym (BoxPushing) into this environment only."
        run python -m pip install --no-cache-dir --no-deps \
            "git+https://github.com/DongTian95/fancy_gymnasium.git@dt_branch"
    fi

    if [[ "$RLAC_WITH_D4RL" == "1" ]]; then
        # Needed only for D4RL AntMaze/Adroit env names (envs/d4rl_utils.py).
        # d4rl's own legacy mujoco-py dependency chain is known to be finicky
        # to install; this is why it's opt-in rather than installed by default.
        log "Installing d4rl (AntMaze/Adroit env support)."
        run python -m pip install git+https://github.com/Farama-Foundation/D4RL.git
    fi
}

install_local_project() {
    if [[ -f "$REPO_ROOT/pyproject.toml" || -f "$REPO_ROOT/setup.py" || -f "$REPO_ROOT/setup.cfg" ]]; then
        log "Installing this repository in editable mode."
        run python -m pip install -e "$REPO_ROOT"
    else
        warn "No pyproject.toml, setup.py, or setup.cfg found; adding repo root with conda develop."
        run conda install "${CONDA_CHANNEL_ARGS[@]}" -y -n "$ENV_NAME" conda-build
        run conda develop "$REPO_ROOT"
    fi
}

verify_module() {
    local module="$1"
    run python -c "import importlib; importlib.import_module('$module')"
}

verify_install() {
    if [[ "$RLAC_SKIP_VERIFY" == "1" ]]; then
        warn "Skipping import verification because RLAC_SKIP_VERIFY=1."
        return
    fi

    log "Verifying core imports."
    verify_module jax
    verify_module flax
    verify_module optax
    verify_module numpy
    verify_module gymnasium
    verify_module mujoco
    verify_module dm_control
    verify_module ogbench
    verify_module distrax
    verify_module wandb
    verify_module robomimic

    if [[ "$RLAC_SKIP_EXTERNAL_DEPS" != "1" ]]; then
        verify_module cw2
        if [[ "$RLAC_WITH_METAWORLD" == "1" ]]; then
            verify_module metaworld
        fi
        if [[ "$RLAC_WITH_BOXPUSHING" == "1" ]]; then
            # NOT verify_module fancy_gym -- importing it is exactly what does
            # not work. Check the env this repo actually builds instead.
            log "Verifying the BoxPushing env builds."
            run python -c "from envs.box_pushing_utils import make_box_pushing_env; e = make_box_pushing_env('fancy/BoxPushingRandomInitDense-v0'); e.reset(seed=0); e.step(e.action_space.sample()); print('  box_pushing  ok')" \
                || die "The BoxPushing env failed to build -- see the traceback above."
        fi
        if [[ "$RLAC_WITH_D4RL" == "1" ]]; then
            verify_module d4rl
        fi
    fi

    log "Verifying main.py / main_cw2.py import cleanly."
    run python -c "import main" || die "main.py failed to import -- see the traceback above."
}

verify_cuda() {
    if [[ "$RLAC_SKIP_VERIFY" == "1" ]]; then
        warn "Skipping GPU verification because RLAC_SKIP_VERIFY=1."
        return
    fi
    if [[ "$(resolve_use_cuda)" == "0" ]]; then
        return
    fi

    log "Verifying CUDA-enabled JAX build."
    run python - <<'PY'
import jax

print("jax:", jax.__version__)
print("devices:", jax.devices())

gpu_devices = [d for d in jax.devices() if d.platform == "gpu"]
if not gpu_devices:
    raise RuntimeError(
        "No GPU device visible to JAX. Check `nvidia-smi` and the NVIDIA driver "
        "version (CUDA 12 wheels need driver >=525)."
    )

import jax.numpy as jnp
x = jnp.ones((1024, 1024))
print("cuda matmul ok:", float((x @ x).mean()))
PY
}

main() {
    cd "$REPO_ROOT"

    init_conda
    create_or_update_env
    install_base_python_deps
    install_external_repos
    install_local_project
    verify_install
    verify_cuda

    log "Configuration completed successfully."
    cat <<EOF

The environment was activated while this script ran. To use it in your shell:

  conda activate $ENV_NAME
  export MUJOCO_GL=egl   # headless rendering, matches the README examples

Quick verification:

  python -c "import jax; print(jax.devices())"

Useful options:

  ENV_NAME=my_env bash conda_env.sh              # choose a different env name
  PYTHON_VERSION=3.10 bash conda_env.sh          # choose a different Python version
  RLAC_CUDA=1 bash conda_env.sh                  # force CUDA 12 JAX
  RLAC_CUDA=0 bash conda_env.sh                  # force CPU-only JAX
  RLAC_WITH_METAWORLD=1 bash conda_env.sh        # also install MetaWorld
  RLAC_WITH_D4RL=1 bash conda_env.sh             # also install D4RL (AntMaze/Adroit)
  RLAC_WITH_BOXPUSHING=1 bash conda_env.sh       # also install fancy_gym (BoxPushing), --no-deps
  RLAC_SKIP_EXTERNAL_DEPS=1 bash conda_env.sh    # skip cw2/MetaWorld/D4RL/fancy_gym git installs
  RLAC_DEPS_DIR=/path/to/deps bash conda_env.sh  # where external repos get cloned

EOF
}

main "$@"
