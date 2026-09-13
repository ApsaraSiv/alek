import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'erc_perception'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rava',
    maintainer_email='sharavi.9596@gmail.com',
    description='Perception nodes for ERC 2026: shelf column and book detection',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_viewer = erc_perception.camera_viewer:main',
            'book_color_detector = erc_perception.book_color_detector:main',
            'shelf_number_detector = erc_perception.shelf_number_detector:main',
            'bin_detector = erc_perception.bin_detector:main',
        ],
    },
)
