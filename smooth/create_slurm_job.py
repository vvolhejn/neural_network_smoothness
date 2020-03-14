import subprocess
import os
import shutil

from colorama import Fore, Back, Style
import click
import yaml

import smooth.util
import smooth.config

template = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output_file}
#SBATCH --time={time}
#SBATCH --mem={memory}G
#SBATCH --cpus-per-task={cpus}
#SBATCH --mail-user=vaclav.volhejn@gmail.com
#SBATCH --mail-type={mail_type}
#SBATCH --no-requeue
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
# for single-CPU jobs make sure that they use a single thread
export OMP_NUM_THREADS=1

module load cuda/10.0 cudnn/7.5 nccl/2.4.2
module load tensorflow/python3/2.1.0
export PYTHONPATH={pythonpath}
export SMOOTH_CONFIG={config_path}
srun --cpu_bind=verbose {command}
"""


def preview(format_dict):
    t = template.replace("{", Style.BRIGHT + Fore.CYAN + "{").replace("}", "}" + Style.RESET_ALL)
    print(t.format(**format_dict))

@click.command()
@click.option('--config-path', required=True)
def main(config_path):
    config = smooth.config.Config(config_path)

    log_dir = smooth.util.get_logdir_name(debug=config.debug)

    format_dict = dict(
        job_name=config.name,
        output_file="slurm_output.txt",
        time=config.max_time,
        memory=config.memory_gb,
        cpus=config.cpus,
        mail_type="NONE" if config.debug else config.mail_type,
        pythonpath=os.environ.get("PYTHONPATH") or "",
        command="python3 -m smooth.train_models_general with log_dir=.",
        config_path="run_config.yaml",
    )

    # print(template.replace("{", "\{{"))
    preview(format_dict)
    if input("Run job? (y/n): ") != "y":
        print("Aborted.")
        exit(1)

    print("Running! Log dir: {}".format(log_dir))

    os.makedirs(log_dir)
    shutil.copyfile(config_path, os.path.join(log_dir, "run_config.yaml"))

    os.chdir(log_dir)

    script = template.format(**format_dict)
    with open("slurm_job.sh", "w") as f:
        f.write(script)

    with open("run_config.yaml", "w") as f:
        config.raw_config["confirm"] = False
        yaml.safe_dump(config.raw_config, f)

    subprocess.run(["bash", "-c", "sbatch ./slurm_job.sh"])


if __name__ == '__main__':
    main()
