FROM nvcr.io/nvidia/pytorch:24.04-py3

ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

USER root

# Create user and prepare environment
RUN groupadd -g 1000 user && \
    useradd -g user -u 1000 -m user && \
    mkdir -p /tmp/.jupyter_data /tmp/.jupyter /home/user && \
    chown -R user:user /tmp/.jupyter_data /tmp/.jupyter /home/user

COPY InternImage /InternImage
RUN chown -R user:user /InternImage

ENV DEBIAN_FRONTEND=noninteractive

# Combine update, package installation, Nvtop installation, and cleanup in one command for optimization
RUN apt-get update && \
    apt-get install -y --no-install-recommends libaio-dev tzdata tmux libncurses5-dev libncursesw5-dev git cmake build-essential libudev-dev libsystemd-dev libdrm-dev curl && \
    apt-get upgrade -qy && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    git clone https://github.com/Syllo/nvtop.git && \
    mkdir -p nvtop/build && cd nvtop/build && \
    cmake .. && \
    make && \
    make install && \
    cd ../.. && rm -rf nvtop

# Create and activate virtual environment
RUN python -m venv /home/user/.venv
ENV VIRTUAL_ENV=/home/user/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.in ./requirements.in

# Install Python packages
RUN pip install -U pip setuptools wheel && \
    pip install -r requirements.in && \
    pip install jupyterlab notebook tqdm wandb ipywidgets nvitop sacrebleu bert_score rouge_score sentence_transformers ruff packaging ninja flash-attn transformers[deepspeed-testing] && \
    git clone https://github.com/EleutherAI/lm-evaluation-harness && cd lm-evaluation-harness && \
    pip install -e . && \
    pip install -e ".[multilingual]"

ENV PYTHONUSERBASE=/home/user/.local
ENV PYTHONUSERPATH="$PYTHONUSERBASE/bin"
ENV PATH=$PYTHONUSERPATH:$PATH

USER user
CMD ["bash"]
