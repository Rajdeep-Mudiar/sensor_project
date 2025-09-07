# help to manage sensor data

# setuptool is a package development and distribution that provides tools for packaging Python projects
from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT='-e.'

# input --> file_path of str datatype
# output --> strings inside list
def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
    
    # To remove the dot
    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name='Fault detection',
    version='0.0.1',
    author='rajdeep',
    author_email='rajdeepmudiar01@gmail.com',
    install_requirements=get_requirements('requirements.txt'),
    packages=find_packages()
)

