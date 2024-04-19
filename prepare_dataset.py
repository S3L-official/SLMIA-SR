
import argparse
import os
from concurrent.futures import as_completed, ProcessPoolExecutor

def parse_args(input_args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='vox2', choices=['vox2', 'ls'], help='dataset name vox2: VoxCeleb2, ls: LibriSpeech')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory')
    parser.add_argument('--n_jobs', type=int, default=2, help='number of jobs to run in parallel')
    args = parser.parse_args() if input_args is None else parser.parse_args(input_args)

    return args


def _worker(job_id, local_infos):
    print('enter worker', job_id, len(local_infos))
    for index, (src_path, des_path) in enumerate(local_infos):
        if os.path.exists(des_path):
            print(des_path, 'exists')
            continue
        try:
            os.system("ffmpeg -i '{}' '{}' </dev/null > /dev/null 2>&1 &".format(src_path, des_path))
            print(job_id, len(local_infos), index, src_path, des_path)
        except:
            print(des_path, 'ffmpeg convert error')
            continue


def main(args):

    dataset_root = os.path.join(args.data_dir, args.dataset)
    assert os.path.exists(dataset_root), f'{dataset_root} does not exist! please download first.'

    infos = []
    ext = 'm4a' if args.dataset == 'vox2' else 'flac'
    for root, _, files in os.walk(dataset_root, followlinks=True):
        for file in files: 
            if file.split('.')[-1] != ext:
                continue
            src_path = os.path.join(root, file)
            des_path = src_path.replace(args.dataset, f'{args.dataset}-wav').replace(ext, 'wav').replace('/aac', '')
            if not os.path.exists(os.path.dirname(des_path)):
                os.makedirs(os.path.dirname(des_path))
            infos.append((src_path, des_path))
    print('total voices:', len(infos))


    n_audios = len(infos)
    n_jobs = args.n_jobs
    n_jobs = n_jobs if n_jobs <= n_audios else n_audios
    n_audios_per_job = n_audios // n_jobs
    process_index = []
    for ii in range(n_jobs):
        process_index.append([ii*n_audios_per_job, (ii+1)*n_audios_per_job])
    if n_jobs * n_audios_per_job != n_audios:
        process_index[-1][-1] = n_audios
    print(process_index)
    futures = set()
    with ProcessPoolExecutor() as executor:
        for job_id in range(n_jobs):
            future = executor.submit(_worker, job_id, infos[process_index[job_id][0]:process_index[job_id][1]])
            futures.add(future)
            print('submit {} job, {} {}'.format(job_id, process_index[job_id][0], process_index[job_id][1]))
        for future in as_completed(futures):
            pass


if __name__ == '__main__':
    
    args = parse_args()
    main(args)