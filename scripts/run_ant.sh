python train.py alg=shac env=ant general.seed=42 general.logdir=AntIncreasingStiffness
python train.py alg=shac env=ant env.config.contact.ke=4e4 env.config.contact.kd=1e4 general.logdir=AntIncreasingStiffness
python train.py alg=shac env=ant env.config.contact.ke=8e4 env.config.contact.kd=2e4 general.logdir=AntIncreasingStiffness
python train.py alg=shac env=ant env.config.contact.ke=1.2e5 env.config.contact.kd=3e4 general.logdir=AntIncreasingStiffness
python train.py alg=shac env=ant env.config.contact.ke=1.6e5 env.config.contact.kd=4e4 general.logdir=AntIncreasingStiffness
python train.py alg=shac env=ant env.config.contact.ke=4e5 env.config.contact.kd=1e5 general.logdir=AntIncreasingStiffness
python train.py alg=shac env=ant env.config.contact.ke=6e5 env.config.contact.kd=1.5e5 general.logdir=AntIncreasingStiffness
python train.py alg=shac env=ant env.config.contact.ke=8e5 env.config.contact.kd=2e5 general.logdir=AntIncreasingStiffness
python train.py alg=shac env=ant env.config.contact.ke=1e6 env.config.contact.kd=0.25e6 general.seed=42 general.logdir=AntIncreasingStiffness

## Loop for the first command
#for seed in {43..46}; do
#  python train.py alg=shac env=hopper general.seed=$seed
#done
#
## Loop for the second command (with contact config)
#for seed in {43..46}; do
#  python train.py \
#    alg=shac \
#    env=hopper \
#    env.config.contact.ke=1e6 \
#    env.config.contact.kd=1e5 \
#    general.seed=$seed
#done


## Loop for the first command
#for seed in {42..46}; do
#  python train.py alg=ahac env=hopper general.seed=$seed
#done
#
## Loop for the second command (with contact config)
#for seed in {42..46}; do
#  python train.py \
#    alg=ahac \
#    env=hopper \
#    env.config.contact.ke=1e6 \
#    env.config.contact.kd=1e5 \
#    general.seed=$seed
#done