import subprocess
from concurrent.futures import ThreadPoolExecutor
import argparse
from torch.cuda import device_count

def run(command):
    subprocess.run(command, shell=True, check=True)

def allocate_gpu_fn(i, n_gpus):
    return i % n_gpus


def main(args):
    n_gpus = device_count()
    n_procs = args.end_at - args.start_at
    print(f'Running on {n_gpus} GPUs')
    commands = [
        f"export CUDA_VISIBLE_DEVICES={allocate_gpu_fn(i, n_gpus)}; \
        python data/create_structures.py --data_cfg {args.data_cfg} {'--test' if args.test else ''}\
            --start_at {i} --end_at {i+1} --n_jobs {args.n_jobs} \
            --n_structures {args.n_structures} --save_every {args.save_every} \
            {'--overwrite' if args.overwrite else ''} --device {args.device} \
            {'--optimized' if args.optimized else ''} \
            {'--prism' if args.prism else ''} \
            " 
        for i in range(args.start_at, args.end_at)
    ]

    with ThreadPoolExecutor() as executor:
        executor.map(run, commands)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg", type=str)
    parser.add_argument("--test", action='store_true', default=False)
    parser.add_argument("--start_at", type=int, default=0)
    parser.add_argument("--end_at", type=int, default=6)
    parser.add_argument("--n_jobs", type=int, default=10)
    parser.add_argument("--n_structures", type=int, default=10000)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--overwrite", action='store_true', default=False)
    parser.add_argument("--device", type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument("--optimized", action='store_true', default=False)
    parser.add_argument("--prism", action='store_true', default=False)

    args = parser.parse_args()
    main(args)
    print('Done.')
    # scripts_to_run = ['script1.py', 'script2.py', 'script3.py']  # Add your script names here

    
