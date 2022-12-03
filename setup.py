from setuptools import setup

package_name = 'comprobo_road_navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='melody',
    maintainer_email='cchiu@olin.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'road_sign_detector = comprobo_road_navigation.road_sign_detector:main',
            'neato_car = comprobo_road_navigation.neato_car:main'
        ],
    },
)
