# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for GestureFlow Canvas."""

import sys
import os

block_cipher = None

ROOT = os.path.abspath(os.path.join(os.path.dirname(SPEC), '..'))

# Collect mediapipe data files (model files needed at runtime)
import importlib
_mp_spec = importlib.util.find_spec('mediapipe')
_mp_datas = []
if _mp_spec and _mp_spec.submodule_search_locations:
    _mp_root = _mp_spec.submodule_search_locations[0]
    for _dirpath, _dirnames, _filenames in os.walk(_mp_root):
        for _fn in _filenames:
            if _fn.endswith(('.tflite', '.binarypb', '.pbtxt', '.txt')):
                _src = os.path.join(_dirpath, _fn)
                _dst = os.path.join('mediapipe', os.path.relpath(_dirpath, _mp_root))
                _mp_datas.append((_src, _dst))

a = Analysis(
    [os.path.join(ROOT, 'apps', 'canvas', 'main.py')],
    pathex=[ROOT],
    binaries=[],
    datas=[
        (os.path.join(ROOT, 'assets', 'icons', '*.ico'), 'assets/icons'),
    ] + _mp_datas,
    hiddenimports=[
        'mediapipe',
        'mediapipe.python',
        'mediapipe.python.solutions',
        'mediapipe.python.solutions.hands',
        'mediapipe.python.solutions.drawing_utils',
        'mediapipe.python.solutions.drawing_styles',
        'cv2',
        'numpy',
        'PyQt6',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'PyQt6.sip',
        # Windows-specific system modules often missed
        'pythoncom',
        'pywintypes',
        'win32api',
        'win32con',
        'win32gui',
        'win32ui',
        'win32clipboard',
        # matplotlib + dependencies (needed by mediapipe)
        'matplotlib',
        'matplotlib.backends.backend_agg',
        'matplotlib.figure',
        'matplotlib.pyplot',
        'PIL',
        'PIL.Image',
        'PIL.ImageDraw',
        'PIL.ImageFont',
        'PIL.ImageColor',
        'kiwisolver',
        'cycler',
        'dateutil',
        'pyparsing',
        'pyparsing.testing',
        'packaging',
        'packaging.version',
        'contourpy',
        'fontTools',
        # mediapipe dependencies
        'google.protobuf',
        'google.protobuf.descriptor',
        'google.protobuf.json_format',
        'google.protobuf.text_format',
        'absl',
        'absl.logging',
        'flatbuffers',
        'attr',
        'attrs',
        'sounddevice',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'test',
        'torch',
        'torch._C',
        'torch.nn',
        'torch.cuda',
        'torch.utils',
        'torch.autograd',
        'torch.jit',
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
    name='GestureFlowCanvas',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(ROOT, 'assets', 'icons', 'app_icon.ico'),
)

# Filter out torch binaries and data (not needed at runtime)
_torch_exclude = ('torch', 'caffe2')
a.binaries = [b for b in a.binaries if not any(b[0].startswith(prefix) or ('\\torch\\' in b[1] if len(b) > 1 else False) for prefix in _torch_exclude)]
a.datas = [d for d in a.datas if not any(d[0].startswith(prefix) or ('\\torch\\' in d[1] if len(d) > 1 else False) for prefix in _torch_exclude)]

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='GestureFlowCanvas',
)
