from setuptools import setup, find_packages
from glob import glob

package_name = 'drone_ocp_py'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),  # Include tutti i pacchetti e sottopacchetti
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Modelli e mondi ora in altro pacchetto
        # ('share/' + package_name + '/models/mrsim', glob('models/mrsim/*')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='acocchia',
    maintainer_email='acocchia@example.com',
    description='Drone OCP planner with acados and ROS2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'openLoop_test= drone_ocp_py.openLoop_test:main',   #test senza controllore
            'peg_planner_node = drone_ocp_py.peg_planner_node:main', # peg planner
            'ocp_planner_node = drone_ocp_py.ocp_planner_node:main', # OCP
            'MPC_planner_node = drone_ocp_py.MPC_planner_node:main', # MPC
            'planner_prova=drone_ocp_py.planner_prova:main', # planner test
            'ocp_offline_loader = drone_ocp_py.ocp_offline_loader:main', #traj offline
            'logger = drone_ocp_py.logger:main', # nodo logger
            'PID_controller = drone_ocp_py.PID_controller:main', # PID/hierarchical controller
            'geometric_controller= drone_ocp_py.geometric_controller:main', # geometric controller,
            'human_goal_node= drone_ocp_py.human_goal_node:main'
        ],
    },
)
