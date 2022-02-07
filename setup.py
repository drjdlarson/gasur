import setuptools


def readme():
    with open('README.md') as f:
        return f.read()

setuptools.setup(name='gasur',
                 version='0.0.0',
                 description='A package for Guidance, Navigation, and Control (GNC) for Autonomous Swarms Using Random finite sets (RFS).',
                 # long_description=readme(),
                 url='https://github.com/drjdlarson/gasur',
                 author='Jordan D. Larson, and Ryan W. Thomas, and Vaughn Weirens',
                 author_email='',
                 license='MIT',
                 packages=setuptools.find_packages('src'),
                 package_dir={"": "src"},
                 install_requires=['numpy', 'scipy', 'matplotlib'],
                 tests_require=['pytest', 'numpy'],
                 include_package_data=True,
                 zip_safe=False)
