from setuptools import setup, Extension, find_packages
import re
import numpy
import os


def get_property(prop, project):
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop), open(project + '/__init__.py').read())
    return result.group(1)

project_name = 'PhonemeRecognition'

include_dirs = [numpy.get_include()]
setup_requires = []
install_requires = ['numpy>=1.13.0',
                    'scipy>=1.0.0'
                   ]
setup(
    name=project_name,
    version=get_property('__version__', 'phonerec'),
    description='Phoneme Recognition using PyTorch',
    author='Sangeon Yong',
    author_email='koragon2@kaist.ac.kr',
    include_dirs=include_dirs,
    install_requires=install_requires,
    packages=find_packages(),
    package_data={
        'weight': weight_files
    },
)