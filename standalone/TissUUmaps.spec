# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

from PyInstaller.utils.hooks import copy_metadata
metadata = copy_metadata('anndata')

a = Analysis(['../tissuumaps/gui.py'],
             pathex=['./'],
             binaries=[],
             datas=[('../tissuumaps/VERSION', './'), ('../tissuumaps/web.zip', './'), ('../tissuumaps/templates', 'templates'), ('../tissuumaps/flask_filetree', 'flask_filetree'), ('../tissuumaps/static', 'static'), ('../tissuumaps/plugins/__init__.py','plugins')]+metadata,
             hiddenimports=["pyyaml","pyvips","matplotlib","mpl_toolkits","anndata"],
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
          strip=False,
          upx=True,
          console=False,
          icon='../tissuumaps/static/misc/favicon.ico')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='TissUUmaps')
