python evaluate_traj.py hydra/job_logging=none hydra/hydra_logging=none \
                task=pick \
                task.environment.sim_gui=false \
                task.environment.viz=true \
                task.environment.viz_time=2 \