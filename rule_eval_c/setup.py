"""
Build the rule_eval_c extension (pure C rule evaluation kernel).

Recommended: build in place (no pip install -e). From this directory run:

  python setup.py build_ext --inplace

Then the .so/.pyd is created inside rule_eval_c/ and "import rule_eval_c"
from the repo root will use it. Re-run after git pull if the C sources change.
"""
from setuptools import setup, Extension

# Build as rule_eval_c._rule_eval_c so "import rule_eval_c" loads the package (__init__.py)
# and the C extension is available as rule_eval_c.eval_rules_c.
rule_eval = Extension(
    #"rule_eval_c"
    "rule_eval_c._rule_eval_c",
    sources=[
        "rule_eval_core.c",
        "rule_eval_module.c",
    ],
    include_dirs=["."],
    extra_compile_args=["-O2"] if __import__("sys").platform != "win32" else ["/O2"],
)

setup(
    name="rule_eval_c",
    version="0.1.0",
    description="Pure C rule evaluation kernel (bitset AND + popcount)",
    package_dir={"rule_eval_c": "."},
    packages=["rule_eval_c"],
    ext_modules=[rule_eval],
    python_requires=">=3.7",
)
