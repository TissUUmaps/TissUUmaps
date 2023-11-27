# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

import distutils.util

COMPILING_PLATFORM = distutils.util.get_platform()
STRIP = False

if COMPILING_PLATFORM == 'win-amd64':
    platform = 'win'
elif COMPILING_PLATFORM == 'linux-x86_64':
    platform = 'nix64'
elif "macosx" in COMPILING_PLATFORM:
    platform = 'mac'

a = Analysis(['../tissuumaps/gui.py'],
             pathex=['./'],
             binaries=[],
             datas=[('../tissuumaps/VERSION', './'), ('../tissuumaps/templates', 'templates'), ('../tissuumaps/static', 'static'), ('../tissuumaps/plugins/__init__.py','plugins')],
             hiddenimports=["pyvips","matplotlib","mpl_toolkits","scipy.sparse", "tissuumaps_schema"],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='TissUUmaps',
          debug=False,
          bootloader_ignore_signals=False,
          strip=STRIP,
          upx=True,
          console=False,
          icon='../tissuumaps/static/misc/favicon.ico')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=STRIP,
               upx=True,
               upx_exclude=[],
               name='TissUUmaps')

if platform == 'mac':
    app = BUNDLE(coll,
                 name='TissUUmaps.app',
                 icon='../tissuumaps/static/misc/design/logo.png',
                 bundle_identifier=None)
