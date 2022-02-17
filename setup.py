from setuptools import setup, find_packages

setup(
    name='pig',
    package='pig',
    version='0.0.1',
    author='AliYounes',
    author_email="ali@robot-learning.de",
    description="PIG: PIG loss for Keypoint Detection",
    url="https://github.com/Alonso94/pig_kp",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.7",
)