import subprocess

if __name__ == '__main__':
    with open("../misc/gdownload_links.txt") as f:
        lines = f.readlines()
        for line in lines:
            number, id = line.replace('\n', '').split(' ')
            print(number, id)
            subprocess.call(['/bin/bash', '-i', '-c',
                             f'gdrive_download {id} /nfs/students/summer-term-2018/project_2/data/CelebAHQ/deltas{number}000.zip'])
