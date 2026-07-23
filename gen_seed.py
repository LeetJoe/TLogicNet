
import sys
import random


def gen_env_seed(data_dir):
    seed_mln = random.randint(0, 10000)
    seed_java = random.randint(0, 10000)
    seed_default = random.randint(0, 10000)
    seed_python = random.randint(0, 10000)
    seed_numpy = random.randint(0, 10000)
    seed_torch = random.randint(0, 10000)
    seed_cuda = random.randint(0, 10000)
    seed_rng = random.randint(0, 10000)

    seed_file = data_dir + '/set_seed.sh'
    with open(seed_file, 'w') as fo:
        fo.write('#!/bin/bash\n')
        fo.write('export MLN_SEED_PYTHON={}\n'.format(seed_mln))
        fo.write('export MLN_SEED_JAVA={}\n'.format(seed_java))
        fo.write('export TKGE_SEED={}\n'.format(seed_default))
        fo.write('export TKGE_SEED_PYTHON={}\n'.format(seed_python))
        fo.write('export TKGE_SEED_NUMPY={}\n'.format(seed_numpy))
        fo.write('export TKGE_SEED_TORCH={}\n'.format(seed_torch))
        fo.write('export TKGE_SEED_CUDA={}\n'.format(seed_cuda))
        fo.write('export TKGE_SEED_NUMPY_RNG={}\n'.format(seed_rng))

        fo.close()


data_dir = sys.argv[1]
print("[GEN RANDOM SEED] - Generating seed shell script...")
gen_env_seed(data_dir)
print("[GEN RANDOM SEED] - Done.")
