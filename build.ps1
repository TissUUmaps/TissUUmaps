pyinstaller.exe --noconsole .\TissUUmaps.spec --noconfirm
Rename-Item -Path dist\TissUUmaps\PyQt5\Qt -NewName Qt5
ISCC.exe build_installer.iss
