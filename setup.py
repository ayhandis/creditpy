from setuptools import setup, find_packages

setup(
    name='creditpy',
    version='1.8',
    packages=find_packages(),
    author='Ayhan Dis',
    author_email='disayhan@gmail.com',
    description='A package for credit scoring analysis',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ayhandis/creditpy',
    license='MIT',
    install_requires=[
        'pandas==2.2.0',
        'numpy==1.26.4',
        'scikit-learn==1.4.0',
        'statsmodels==0.14.1',
        'scipy==1.12.0',
        'gap_stat==2.0.3',
        'scikit-learn-extra==0.3.0'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
