"""Minimal Colab runtime setup. No cloning, checkpoint downloads or model loading."""

from pathlib import Path
import subprocess
import sys

REQUIREMENTS = (
    "SimpleITK==2.5.3",
    "monai==1.5.1",
    "monai-generative==0.2.3",
    "transformers==4.49.0",
    "einops>=0.8.1",
    "nibabel>=5.3.2",
    "scipy>=1.10",
    "matplotlib>=3.7",
    "psutil>=5",
    "tqdm>=4.67",
)


def install_dependencies(constraints_path):
    """Install missing packages while retaining Colab's existing torch/CUDA and NumPy.

    Run before importing encoders. Restart if replacing a package already imported
    in the notebook kernel. Export uses the legacy setup separately, once only.
    """
    from importlib.metadata import PackageNotFoundError, version
    from packaging.requirements import Requirement

    needed = []
    for specification in REQUIREMENTS:
        requirement = Requirement(specification)
        try:
            if not requirement.specifier.contains(
                version(requirement.name), prereleases=True
            ):
                needed.append(specification)
        except PackageNotFoundError:
            needed.append(specification)
    constraints = Path(constraints_path)
    constraints.write_text(
        "\n".join(f"{name}=={version(name)}" for name in ("torch", "numpy")) + "\n",
        encoding="utf-8",
    )
    if needed:
        print("Installing compact inference dependencies:", needed, flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--prefer-binary",
                "--upgrade-strategy",
                "only-if-needed",
                "-c",
                str(constraints),
                *needed,
            ],
            check=True,
        )
    else:
        print("Compact inference dependencies ready; pip skipped.")
    return needed
