
    export VIVARIUM_LOGGING_DIRECTORY=/mnt/share/homes/abie/vivarium_results/sodium_usa/sodium_usa/2024_09_03_19_49_02/logs/2024_09_03_19_49_02_run/worker_logs
    export PYTHONPATH=/mnt/share/homes/abie/vivarium_results/sodium_usa/sodium_usa/2024_09_03_19_49_02:$PYTHONPATH

    /homes/abie/.conda/envs/vivarium_nih_us_cvd/bin/rq worker -c settings         --name ${SLURM_ARRAY_JOB_ID}.${SLURM_ARRAY_TASK_ID}         --burst         -w "vivarium_cluster_tools.psimulate.worker.core._ResilientWorker"         --exception-handler "vivarium_cluster_tools.psimulate.worker.core._retry_handler" vivarium

    