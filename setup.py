from setuptools import find_packages, setup
from typing import List

def get_requirements() -> List[str] :
    """
    This function returns the list of requirements to be downloaded
    """
    requirements_lst:List[str] = []
    try:
        with open("requirements.txt", "r") as file:
            lines = file.readlines()
            for line in lines:
                requirement = line.strip()
                if requirement and requirement != "-e .":
                    requirements_lst.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found")
    return requirements_lst

print(get_requirements())

setup(
    name="FindMyNextAnime",
    version= "0.0.1",
    author= "Prasoon Parashar",
    author_email= "prasoonparashar21@gmail.com",
    packages= find_packages(),
    install_requires = get_requirements()
)