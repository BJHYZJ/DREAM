import setuptools

__version__ = None
with open("dream/version.py") as f:
    exec(f.read())  # overrides __version__

with open("../README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dream",
    version=__version__,
    author="Zhijie Yan",
    author_email="Yanzhijie@buaa.edu.cn",
    description="dream package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BJHYZJ/dream",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={"dream": ["config/**/*.yaml", "perception/*.tsv"]},
    install_requires=[
        # Machine learning code
        "torch<2.4",
        "torchvision",
        # General utilities
        "pyyaml",
        "pyzmq",
        "numpy",
        "transforms3d==0.3.1",
        "numba",
        "opencv-python",
        "scipy",
        "matplotlib",
        "trimesh>=3.10.0",
        "yacs",
        "scikit-image>=0.21.0",
        "sophuspy",
        "pin",  # Pinocchio IK solver
        "pynput",
        "pyusb",
        "schema",
        "overrides",
        "wget",
        # From openai
        "openai",
        "openai-clip",
        # For Yolo
        # "ultralytics",
        "pyrealsense2",
        "urchin",
        # Visualization
        "rerun-sdk==0.26.1",
        # For siglip encoder
        "sentencepiece",
        # For git tools
        "gitpython",
        # Configuration tools and neural networks
        "hydra-core",
        "timm>1.0.0",
        "huggingface_hub[cli]",
        "transformers>=4.39.2",
        "accelerate",
        "einops",
        # Meta neural nets
        "segment-anything",
        # Compression tools
        "pyliblzfse",
        "webp>=0.3.0",
        # UI tools
        "termcolor",
        # Audio
        "google-cloud-texttospeech",  # online TTS engine, requiring credentials.
        "gtts",  # online TTS engine, not requiring credentials.
        "librosa",  # audio analysis (e.g., spectral similarity)
        "PyAudio>=0.2.14",  # the version specification is necessary because apt has 0.2.12 which is incompatible with recent numpy
        "openai-whisper",
        "overrides",  # better inheritance of docstrings
        "pydub",  # playback audio
        "pyttsx3",  # offline TTS engine. TODO: There are better options, such as "tts_models/en/ljspeech/fast_pitch" from https://github.com/coqui-ai/TTS
        "simpleaudio",  # playback audio
        "sounddevice",  # Suppresses ALSA warnings when launching PyAudio
        # "wave",
        # These are not supported in python 3.12
        "scikit-fmm",
        "open3d",
    ],
    extras_require={
        "dev": [
            "pre-commit",
            "pytest",
            "flake8",
            "black",
            "mypy",
            "lark",
        ],
        "discord": [
            "discord.py",
            "python-dotenv",
        ],
        "hand_tracker": [
            "mediapipe",
            "webcam",
        ],
    },
)
