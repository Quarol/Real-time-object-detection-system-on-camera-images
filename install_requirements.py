import subprocess
import sys

default_packages = [
    'opencv-python==4.10.0.84',
    'ultralytics==8.3.7',
    'playsound==1.2.2',
    'pillow==10.4.0',
]

extended_packages = [
    ('torch==2.4.1', 'https://download.pytorch.org/whl/cu118'),
    ('torchaudio==2.4.1', 'https://download.pytorch.org/whl/cu118'),
    ('torchvision==0.19.1', 'https://download.pytorch.org/whl/cu118'),
]

def install_package(package: str, extra_index: str = None):
    command = [sys.executable, '-m', 'pip', 'install']
    
    if extra_index:
        command.extend(['--extra-index-url', extra_index])
    
    command.append(package)
    
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as e:
        print(f'Error installing package {package}: {e}')


def main():
    for package, extra_index in extended_packages:
        install_package(package, extra_index)

    for package in default_packages:
        install_package(package)


if __name__ == "__main__":
    main()