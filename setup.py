import setuptools


def readme():
    with open('README.md') as f:
        return f.read()
        
setuptools.setup(name='gasur',
      version='0.0.0',
      description='A package for Guidance, Navigation, and Control (GNC) for Autonomous Swarms Using Random finite sets (RFS).',
      long_description=readme(),
      url='https://github.com/drjdlarson/gasur',
      author='Laboratory for Autonomy GNC and Estimation Research (LAGER)',
      author_email='',
      license='MIT',
      packages=setuptools.find_packages(),
      install_requires=[
          'numpy',
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False)