from setuptools import setup
import os

twinning_cpp = os.path.join(os.getcwd(), "twinning_cpp")

setup(name="twinning",
      version="1.0",
      description="An efficient algorithm for data twinning.",
      author="Akhil Vakayil",
      author_email="akhilv@gatech.edu",
      packages=['twinning'],
      install_requires=['numpy', f"twinning_cpp @ file://localhost/{twinning_cpp}#egg=twinning_cpp"]
     )