from setuptools import setup, find_packages

setup(
    name='creditpy',
    version='1.1',
    packages=find_packages(),
    author='Ayhan Dis',
    author_email='disayhan@gmail.com',
    description='A package for credit scoring analysis',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ayhandis/creditpy',
    license='MIT',
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'statsmodels',
        'scipy',
        'gap_stat',
        'scikit-learn-extra'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
