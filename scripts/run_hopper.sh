#python train.py alg=shac env=hopper general.seed=42
##python train.py alg=shac env=hopper env.config.contact.ke=4e4 env.config.contact.kd=4e3
##python train.py alg=shac env=hopper env.config.contact.ke=5e4 env.config.contact.kd=5e3
##python train.py alg=shac env=hopper env.config.contact.ke=7e4 env.config.contact.kd=7e3
##python train.py alg=shac env=hopper env.config.contact.ke=9e4 env.config.contact.kd=9e3
##python train.py alg=shac env=hopper env.config.contact.ke=1e5 env.config.contact.kd=1e4
##python train.py alg=shac env=hopper env.config.contact.ke=5e5 env.config.contact.kd=5e4
##python train.py alg=shac env=hopper env.config.contact.ke=7e5 env.config.contact.kd=7e4
#python train.py alg=shac env=hopper env.config.contact.ke=1e6 env.config.contact.kd=1e5 general.seed=42
#
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


# Loop for the first command
for seed in {42..46}; do
  python train.py alg=ahac env=hopper general.seed=$seed
done

# Loop for the second command (with contact config)
for seed in {42..46}; do
  python train.py \
    alg=ahac \
    env=hopper \
    env.config.contact.ke=1e6 \
    env.config.contact.kd=1e5 \
    general.seed=$seed
done