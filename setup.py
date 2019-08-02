from setuptools import setup

setup(name='timemachine-train',
    packages=['system'],
    entry_points={
        'console_scripts':['timemachine-train = system.training:main']
    },
    )
