# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['../src/TissUUmaps_standalone.py'],
             pathex=['./'],
             binaries=[],
             datas=[('../src/templates_standalone', 'templates_standalone'), ('../src/static', 'static'), ('../src/plugins/__init__.py','plugins')],
             hiddenimports=["pyvips","matplotlib","mpl_toolkits"],
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
          console=True,
          icon='../src/static/misc/favicon.ico')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='TissUUmaps')
