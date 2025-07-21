from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str) -> List[str]:
    """
    This function reads a requirements file and returns a list of dependencies.
    """

    requirements = []
    with open(file_path)as file:
        requirements = file.readlines()
        requirements = [req.replace('\n', "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

        return requirements


setup(
    name='end-to-end-mlops',
    version='0.0.1',
    author='Faraz Ahmad',
    author_email='farazahmad1234@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)