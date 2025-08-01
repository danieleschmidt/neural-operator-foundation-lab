# Neural Operator Foundation Lab Development Container
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    vim \
    nano \
    htop \
    tree \
    jq \
    unzip \
    build-essential \
    pkg-config \
    cmake \
    ninja-build \
    libhdf5-dev \
    libopenmpi-dev \
    libblas-dev \
    liblapack-dev \
    libfftw3-dev \
    libgsl-dev \
    ssh \
    sudo \
    zsh \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Install Oh My Zsh for better terminal experience
RUN su - $USERNAME -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" || true

# Install additional Python packages for development
RUN pip install --no-cache-dir \
    # Core ML/DL packages
    numpy \
    scipy \
    scikit-learn \
    pandas \
    matplotlib \
    seaborn \
    plotly \
    jupyter \
    jupyterlab \
    ipywidgets \
    # Development tools
    black \
    isort \
    ruff \
    mypy \
    pytest \
    pytest-cov \
    pytest-xdist \
    pytest-benchmark \
    pre-commit \
    # ML experiment tracking
    wandb \
    mlflow \
    tensorboard \
    # Distributed computing
    ray \
    dask \
    # Scientific computing
    h5py \
    netcdf4 \
    xarray \
    # Visualization
    mayavi \
    pyvista \
    # Performance profiling
    line_profiler \
    memory_profiler \
    py-spy \
    # Documentation
    sphinx \
    sphinx-rtd-theme \
    myst-parser \
    # Utilities
    tqdm \
    rich \
    typer \
    pydantic \
    # Security
    bandit \
    safety

# Install additional tools
RUN curl -sSL https://install.python-poetry.org | python3 -
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt update \
    && apt install gh -y

# Switch to non-root user
USER $USERNAME
WORKDIR /workspace

# Set up Git configuration template
RUN git config --global init.defaultBranch main \
    && git config --global pull.rebase false \
    && git config --global core.editor "code --wait"

# Add user binaries to PATH
ENV PATH="/home/$USERNAME/.local/bin:$PATH"

# Set up shell
RUN echo 'export PATH="/home/$USERNAME/.local/bin:$PATH"' >> /home/$USERNAME/.zshrc \
    && echo 'export PYTHONPATH="/workspace/src:$PYTHONPATH"' >> /home/$USERNAME/.zshrc \
    && echo 'alias ll="ls -alF"' >> /home/$USERNAME/.zshrc \
    && echo 'alias la="ls -A"' >> /home/$USERNAME/.zshrc \
    && echo 'alias l="ls -CF"' >> /home/$USERNAME/.zshrc

# Create workspace structure
RUN mkdir -p /workspace/{data,models,checkpoints,logs,experiments,outputs}

SHELL ["/bin/zsh", "-c"]