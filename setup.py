import setuptools

package_data = {
    '': ['aux_files/*','aux_files/WA/*','aux_files/VS_ADR/*'],
    }
setuptools.setup(
    name="pyCRISM",
    version="1.0.0",
    author='Binlong Ye',
    author_email='binlongy@connect.hku.hk',
    description='A python implementation of CAT to process CRISM data',
    long_description_content_type='text/markdown',
    url='https://github.com/justinbl/pyCRISM/',
    project_urls={
        'Source': 'https://github.com/justinbl/pyCRISM/',
    },
    packages=setuptools.find_packages(),
    package_data=package_data,
    py_modules=["pyCRISM"],
    python_requires='>=3.11',
    setup_requires=['wheel'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy'
        ]
    )