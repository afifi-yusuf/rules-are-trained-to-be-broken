"""Stand-in for OpenCV on hosts with kernel FIPS mode on (Databricks Serverless GPU).

opencv-python-headless, which vLLM depends on, bundles a Red Hat build of OpenSSL
1.1.1k. When the kernel is in FIPS mode, loading that library runs the FIPS self-test,
which fails because the bundled copy has no integrity file, and the whole process
aborts ("FATAL FIPS SELFTEST FAILURE"). Our runs are text-only and never call OpenCV,
so scripts/run_rtb.sh and the notebooks put this directory first on PYTHONPATH on FIPS
hosts: `import cv2` succeeds, and any real use raises an error instead of aborting.
"""

__version__ = "4.11.0+fipsstub"


def __getattr__(name):
    raise AttributeError(f"cv2.{name}: OpenCV is stubbed out on this FIPS-mode host (see scripts/fips_stubs/cv2)")
