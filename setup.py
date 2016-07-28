from setuptools import setup

setup(name='learningml',
      version='0.3',
      description='This repository demonstrates how to make a project pip installable, write a Python module in C and use scikit-learn, keras and spearmint.',
      url='https://github.com/weissercn/learningml',
      author='Constantin Weisser',
      author_email='weissercn@gmail.com',
      license='MIT',
      packages=['learningml'],
      install_requires=[
          'numpy',
          'scipy',
          'scikit-learn',
          'keras',
      ],
      zip_safe=False)
