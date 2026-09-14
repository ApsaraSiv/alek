from setuptools import find_packages, setup

package_name = 'erc_solution'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/solution.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='team',
    maintainer_email='apsfamily2020@gmail.com',
    description='ERC 2026 Phase 1 team solution package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'state_machine_node = erc_solution.state_machine_node:main',
            'shelf_column_detector = erc_solution.shelf_column_detector:main',
            'book_detector = erc_solution.book_detector:main',
            'navigation_node = erc_solution.navigation_node:main',
            'manipulation_node = erc_solution.manipulation_node:main',
        ],
    },
)
