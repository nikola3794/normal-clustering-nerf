import tarfile
import os


def make_tarfile(output_filename, directory_paths):
    with tarfile.open(output_filename, "w:gz") as tar:
        for dir_pth in directory_paths:
            tar.add(dir_pth, arcname=os.path.basename(dir_pth))

if __name__ == '__main__':
    hypersim_root_dir = 'HYPERSIM_ROOT_DIR'
    output_root_dir = 'OUTPUT_ROOT_DIR'
    hypersim_scene_dirs = [x.path for x in os.scandir(hypersim_root_dir)]
    hypersim_scene_dirs.sort()
    all_generated_targz_files = []
    for i in range(1, 55+1):
        partition_name = f'ai_{i:03d}_'
        print(f'Zipping part {partition_name}....')
        scenes_to_be_zipped = [x for x in hypersim_scene_dirs if partition_name in x ]
        if scenes_to_be_zipped:
            output_filename = os.path.join(output_root_dir, f'{partition_name}.tar.gz')
            make_tarfile(output_filename, scenes_to_be_zipped)
            all_generated_targz_files.append(output_filename)

    txt_summary_path = os.path.join(output_root_dir, 'all_targz_files.txt')
    with open(txt_summary_path, 'w') as f:
        for line in all_generated_targz_files:
            f.write(f"{os.path.basename(line)}\n")
    print(f'Created summary in {txt_summary_path}')
