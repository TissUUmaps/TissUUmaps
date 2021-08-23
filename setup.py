import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
     name='TissUUmaps',  
     version='2.4.3',
     author="Leslie Solorzano, Christophe Avenel, Fredrik Nysjö",
     author_email="christophe.avenel@it.uu.se",
     description="TissUUmaps is a lightweight viewer that uses basic web tools to visualize gene expression data or any kind of point data on top of whole slide images.",
     long_description=long_description,
     long_description_content_type="text/markdown",
     license='GNU General Public License v3.0',
     url="https://tissuumaps.research.it.uu.se/",
     packages=["tissuumaps"],
     include_package_data=True,
     zip_safe=False,
     install_requires=[
        'Flask>=2.0.0',
        'openslide-python>=1.1.2',
        'Pillow>=8.2.0',
        'pyvips>=2.1.14',
        'PyQt5>=5.15.4',
        'PyQtWebEngine>=5.15.4'
    ],
     extra_requires=[
        'PyQt5>=5.15.4',
        'PyQtWebEngine>=5.15.4'
    ],
     classifiers=[
         "Programming Language :: Python :: 3",
         'Intended Audience :: Science/Research',
         "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
         "Operating System :: OS Independent",
     ],
     entry_points={
        'console_scripts': [
            'tissuumaps_server = tissuumaps.__main__:main',
            'tissuumaps = tissuumaps.gui:main'
        ]
    }
 )