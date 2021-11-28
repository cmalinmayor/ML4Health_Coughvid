from setuptools import setup

setup(
        name='coughvid',
        version='0.0',
        url='https://github.com/cmalinmayor/ML4Heatlh_Coughvid',
        packages=[
            'coughvid',
            'coughvid.audio_processing',
            'coughvid.pytorch',
        ],
)
