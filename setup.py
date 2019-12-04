from setuptools import find_packages, setup

# The hyphens in release candidates (RCs) will automatically be normalized.
# But we normalize below manually anyway.
_VERSION = '1.0-rc0'

# TODO: Add version numbers.
REQUIRED_PACKAGES = [
  'scipy',
  'tensorflow >= 1.0.0',
  'scikit-learn',
  'librosa', # audio preprocessing
  'h5py'
]

setup(name='Fathom-Workloads', # "fathom" is already taken on PyPI
      description='Reference workloads for modern deep learning',
      url='http://github.com/rdadolf/fathom',

      version=_VERSION.replace('-', ''),

      # Authors: Robert Adolf, Saketh Rama, and Brandon Reagen
      # PyPI does not have an easy way to specify multiple authors.
      author="Saketh Rama",
      author_email="rama@seas.harvard.edu",
      
      # We don't use __file__, but mark False to be safe.
      zip_safe=False,

      python_requires='>3.5',

      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',

        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: System :: Hardware',
      ],

      packages=find_packages(), # find packages in subdirectories

      package_data={'fathom': [
        'fathom.png',

        'Dockerfile',
        'pylintrc',

        'README.md',
        'mkdocs.yml',

        'runtest.sh',

        'setup.cfg',
      ]},
      include_package_data=True,
)

