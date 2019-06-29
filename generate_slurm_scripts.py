import os
import subprocess

# reset slurm scripts directory
subprocess.check_output(['rm','-r', "slurm_scripts"])
subprocess.check_output(['mkdir','slurm_scripts'])

def create_global_scripts(task, sbatch, eval_sbatch):
    # create script to run all training slurm scripts
    script = "train_all_{}_models.sh".format(task)
    f = open(script, "w+")
    f.write(sbatch)
    f.close()
    subprocess.check_output(['chmod','+x', script])     

    # create script to run all evaluation slurm scripts
    script = "eval_all_{}_models.sh".format(task)
    f = open(script, "w+")
    f.write(eval_sbatch)
    f.close()
    subprocess.check_output(['chmod','+x', script]) 

## mlc.py

task = 'attr_mlc'

# prepare global scripts
sbatch = "#!/bin/bash\n"
eval_sbatch = "#!/bin/bash\n"

script = os.path.join("slurm_scripts", "{0}_train".format(task))
f = open(script, "w+")
f.write(
'''#!/bin/bash
#SBATCH --job-name={0}_train       # the name of the job
#SBATCH --output=logging/{0}_train # where stdout and stderr will write to
#
#SBATCH --gres=gpu:4                # number of GPUs your job requests
#SBATCH --mem=64G                   # amount of memory needed
#SBATCH --time=48:00:00             # limit on total runtime
#
# send mail during process execution
#SBATCH --mail-type=all
#SBATCH --mail-user=ick@princeton.edu
#
srun -A visualai python main.py
'''.format(task))
f.close()
subprocess.check_output(['chmod','+x', script])
sbatch += "sbatch {}\n".format(script)

# create slurm script for evaluation
script = os.path.join("slurm_scripts", "{0}_eval".format(task))
f = open(script, "w+")
f.write(
'''#!/bin/bash
#SBATCH --job-name={0}_eval        # the name of the job
#SBATCH --output=logging/{0}_eval  # where stdout and stderr will write to
#
#SBATCH --gres=gpu:1               # number of GPUs your job requests
#SBATCH --mem=64G                  # amount of memory needed
#SBATCH --time=2:00:00             # limit on total runtime
#
# send mail during process execution
#SBATCH --mail-type=all
#SBATCH --mail-user=ick@princeton.edu
#
srun -A visualai python main.py --model {0}
'''.format(task))
f.close()
subprocess.check_output(['chmod','+x', script])
eval_sbatch += "sbatch {}\n".format(script)

# create_global_scripts(task, sbatch, eval_sbatch)
