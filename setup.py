from setuptools import setup, find_packages
import sys

with open('README.md') as f:
    readme = f.read()

# with open('LICENSE') as f:
#     license = f.read()

with open('requirements.txt') as f:
    reqs = f.read()

setup(
    name='gbcmodel',
    version='1.0.0',
  #  packages=['models', 'models.tests'],
    url='',
    license='',
    author='Serhii Korsunenko',
    author_email='serge@korsunenko.com',
    description=' Gradient Boosting Classifier with hyperparameters tunning ',
    long_description=readme,
    python_requires='==3.6',
    packages=find_packages(exclude=('examples')),
    install_requires=reqs.strip().split('\n'),
)
