#!/bin/bash

CONT_UID=$(id -u user)
CONT_GID=$(id -g user)
RUN_UID=${LOCAL_UID:=$CONT_UID}
RUN_GID=${LOCAL_GID:=$CONT_GID}

if [ "${RUN_UID}" != "${CONT_UID}" ] || [ "${RUN_GID}" != "${CONT_GID}" ]; then
  groupmod -g "${RUN_GID}" user
  usermod -u "${RUN_UID}" -g "${RUN_GID}" user
fi

exec /usr/sbin/gosu user "$@"