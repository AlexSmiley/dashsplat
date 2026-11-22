#!/bin/bash
# Installed in the create request, independent of the benchmark SSH connection.
stopself() {
  if [[ -n "$RUNPOD_API_KEY" && -n "$RUNPOD_POD_ID" ]]; then
    if ! curl --fail --silent --max-time 15 -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" >/dev/null; then
      python -c 'import json,os; print(json.dumps({"query": "mutation { podTerminate(input: {podId: %s}) }" % json.dumps(os.environ["RUNPOD_POD_ID"])}))' |
        curl --fail --silent --max-time 15 -H 'Content-Type: application/json' --data-binary @- "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" >/dev/null
    fi
  fi
  kill -TERM 1
}
(sleep 3900; stopself) &
(
  sleep 840; idle=0
  while (( idle < 150 )); do
    if pgrep -f '^bash pipeline/cuda/setup.sh|^tar xf - -C /root/rs' >/dev/null; then idle=0; else idle=$((idle+15)); fi
    sleep 15
  done
  stopself
) &
set -e; umask 077; mkdir -p /root/.ssh /run/sshd
printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
for name in PATH LD_LIBRARY_PATH LIBRARY_PATH CPATH CUDA_HOME; do printf 'export %s=%q\n' "$name" "${!name}"; done >/root/rs-ngc-paths
if ! test -x /usr/sbin/sshd; then export DEBIAN_FRONTEND=noninteractive; apt-get update -qq && apt-get install -y --no-install-recommends openssh-server; fi
ssh-keygen -A; exec /usr/sbin/sshd -D -e -o PermitRootLogin=prohibit-password -o PasswordAuthentication=no -o KbdInteractiveAuthentication=no
