# Detached Train Launch Instability

Status: resolved

## Symptom

Attempts to start long training runs through `nohup` or `setsid` returned a PID but the process exited immediately and produced no stable log file.

## Root Cause

The failure was not in the training code itself. The environment-specific detached launch path was unreliable for this command pattern.

## Resolution

Training was validated in direct execution first, then launched through a persistent exec session instead of relying on the broken detached pattern.

## Outcome

The following runs completed successfully:

- Stage A main
- Stage B main
- Stage C main
- Stage B refresh after teacher outputs increased

## Follow-up

If true detached training is required later, use a session manager that is known to be stable in this environment.
