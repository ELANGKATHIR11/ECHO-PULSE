# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from pathlib import Path

block_cipher = None

BASE_DIR = Path(r"f:\echopulsenet---marine-sonar-intelligence-platform (1)")

added_files = [
    (str(BASE_DIR / "dist"), "dist"),
    (str(BASE_DIR / "backend"), "backend"),
    (str(BASE_DIR / "models_checkpoints"), "models_checkpoints"),
    (str(BASE_DIR / "configs"), "configs"),
    (str(BASE_DIR / "assets"), "assets"),
    (str(BASE_DIR / "data" / "manifests"), "data/manifests"),
]

hidden_imports = [
    "uvicorn",
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "fastapi",
    "fastapi.staticfiles",
    "fastapi.responses",
    "fastapi.middleware.cors",
    "starlette",
    "pydantic",
    "cv2",
    "PIL",
    "PIL.Image",
    "numpy",
    "torch",
    "torchvision",
    "ultralytics",
    "webview",
    "sqlalchemy",
    "sqlite3",
]

a = Analysis(
    ['desktop_app.py'],
    pathex=[str(BASE_DIR), str(BASE_DIR / "backend")],
    binaries=[],
    datas=added_files,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter', 'matplotlib', 'tensorboard', 'mlflow', 'dvc', 'jupyter',
        'IPython', 'spacy', 'transformers', 'ray', 'tornado', 'notebook',
        'pytest', 'sphinx', 'torchaudio', 'tensorrt', 'tensorrt_libs',
        'playwright', 'nltk', 'langchain', 'langchain_classic', 'fastai',
        'dask', 'xarray', 'botocore', 'openpyxl', 'statsmodels', 'pywt',
        'reportlab', 'pypdfium2', 'pypdfium2_raw', 'sentry_sdk', 'plotly',
        'nbformat', 'bitsandbytes'
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='EchoPulseNet_Standalone',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(BASE_DIR / "assets" / "app_icon.ico")
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='EchoPulseNet_Standalone'
)
