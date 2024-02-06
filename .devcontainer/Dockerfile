FROM nvidia/cuda:12.3.1-devel-ubuntu20.04 as devcontainer

# 最低限必要なライブラリのインストール
RUN set -x \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        gnupg2 \
        software-properties-common \
        language-pack-ja \
        tzdata \
        curl \
        lsb-release \
    && apt-get -y clean \
    && rm -rf /var/lib/apt/lists/*

# ロケールとタイムゾーンの設定
RUN update-locale LANG=ja_JP.UTF-8 LANGUAGE=ja_JP:ja \
    && ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime \
    && echo "Asia/Tokyo" > /etc/timezone
ENV LANG ja_JP.UTF-8
ENV LC_ALL ja_JP.UTF-8
ENV LC_CTYPE ja_JP.UTF-8

# 開発に必要なパッケージのインストール
RUN set -x \
    # Python をインストールするために、deadsnakes PPA を追加
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        # 便利ツール
        bash-completion \
        build-essential \
        git \
        iputils-ping \
        jq \
        less \
        net-tools \
        openssh-client \
        shellcheck \
        sudo \
        tar \
        time \
        unzip \
        vim \
        wget \
        xz-utils \
        zip \
        gem \
        libx11-xcb1 \
    && apt-get -y clean \
    && rm -rf /var/lib/apt/lists/*

# Python3.10 のインストール
RUN set -x \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3.10-distutils \
    && apt-get -y clean \
    && rm -rf /var/lib/apt/lists/*

# Python3.10 の設定
RUN set -x \
    && ln -s /usr/bin/python3.10 /usr/local/bin/python3 \
    && ln -s /usr/bin/python3.10 /usr/local/bin/python 

# Python の設定
ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_NO_CACHE_DIR=on \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PATH=${PATH}:/home/vscode/.local/bin

# pip のインストール
RUN set -x \
    && curl -sSL https://bootstrap.pypa.io/get-pip.py | python3

# vscode ユーザーの追加と sudo 権限の付与
RUN useradd -m -s /bin/bash -u 5000 -U vscode \
    && echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/ALL

USER vscode

# poetry のインストール
RUN set -x \
    && POETRY_VERSION=1.5.1 \
      curl -sSL https://install.python-poetry.org | python3 - \
    && poetry config virtualenvs.in-project true \
    && poetry self add "poetry-dynamic-versioning[plugin]"