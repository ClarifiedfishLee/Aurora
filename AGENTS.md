# Aurora Agent Notes

## Merlin A100 Worker

Run Worker commands from Merlin Devbox `56720`. Disable colored/interactive
output for every `mlx worker` command so logs remain readable.

Check the Arnold quota before launching:

```bash
NO_COLOR=1 TERM=dumb mlx worker quota \
  --resourcetype arnold \
  --usergroup TilTok_LBS_location
```

Launch the verified one-card Aurora Worker:

```bash
cd /mlx_devbox/users/jieyu.li/Aurora
NO_COLOR=1 TERM=dumb mlx worker launch \
  --resourcetype arnold \
  --usergroup TilTok_LBS_location \
  --cluster cloudnative-maliva \
  --queuename compute-815-aliyun.va-cloudnative-ai-tiltok.lbs.location-guarantee \
  --gpu 1 \
  --type A100-SXM-80GB \
  --alias aurora-a100 \
  --workdir /mlx_devbox/users/jieyu.li/Aurora \
  -- bash
```

The long `compute-815-aliyun...location-guarantee` value is the queue name;
the `--cluster` value is `cloudnative-maliva`. The verified allocation is one
A100-SXM4-80GB with 15 CPU cores and 247 GiB memory.

Inspect, reconnect to, and release a Worker:

```bash
NO_COLOR=1 TERM=dumb mlx worker list
NO_COLOR=1 TERM=dumb mlx worker login <worker-id>
NO_COLOR=1 TERM=dumb mlx worker kill <worker-id>
```

Workers run for at most 96 hours and Arnold GPU Workers are subject to the
platform utilization policy. Release the Worker as soon as GPU work is done.

## Persistent Paths

- Source and virtual environment: `/mlx_devbox/users/jieyu.li/Aurora`
- Run Aurora from the persistent source directory above.
- Do not assume files under the Devbox master's `/tmp` are visible on a Worker.
- Store code, metadata, logs, and checkpoints under `/mlx_devbox` or another
  explicitly persistent mount.
