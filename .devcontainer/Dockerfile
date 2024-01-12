FROM ubuntu:20.04 as devcontainer

SHELL ["/bin/bash", "-c"]

# Install prerequisites, Set locale & timezone
RUN set -x \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        tzdata \
        ca-certificates \
        software-properties-common \
        language-pack-ja \
    && apt-get clean && rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/*

# ロケールとタイムゾーンの設定
RUN update-locale LANG=ja_JP.UTF-8 LANGUAGE=ja_JP:ja \
    && ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime \
    && echo "Asia/Tokyo" > /etc/timezone
ENV LANG ja_JP.UTF-8
ENV LC_ALL ja_JP.UTF-8
ENV LC_CTYPE ja_JP.UTF-8

# common tools
RUN set -x \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        software-properties-common \
        bash-completion \
        curl \
        git \
        gnupg2 \
        iputils-ping \
        jq \
        less \
        make \
        net-tools \
        openssh-client \
        tar \
        time \
        unzip \
        xz-utils \
        gosu \
        zip \
        vim \
        wget \
        unzip \
        gnupg \
        gcc \
        lsb-release \
    && apt-get clean && rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/*

# create user
RUN set -x \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        sudo \
    && apt-get clean && rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/* \
    && echo "user ALL=(ALL) NOPASSWD:ALL" >>/etc/sudoers.d/ALL \
    && groupadd \
        --gid 5000 \
        vscode \
    && useradd \
        --uid 5000 \
        --gid 5000 \
        --home-dir /home/vscode \
        --create-home \
        --shell /bin/bash \
        vscode \
    && usermod -aG video vscode

# python3.8
RUN set -x \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.8-dev \
        python3-pip \
        python3-distutils \
        python3-venv \
    && apt-get clean && rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/* \
    && ln -s $(which python3.8) /usr/local/bin/python
    
# Python の設定
ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_NO_CACHE_DIR=on \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PATH=${PATH}:/home/vscode/.local/bin

# pip のインストール
RUN set -x \
    && curl -sSL https://bootstrap.pypa.io/get-pip.py | python3

USER vscode

# poetry のインストール
RUN set -x \
    && POETRY_VERSION=1.5.1 \
      curl -sSL https://install.python-poetry.org | python3 - \
    && poetry config virtualenvs.in-project true \
    && poetry self add "poetry-dynamic-versioning[plugin]"

ADD entrypoint.sh /entrypoint.sh
ENTRYPOINT [ "bash", "-eux", "/entrypoint.sh" ]
