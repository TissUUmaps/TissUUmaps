pyinstaller.exe .\TissUUmaps.spec --noconfirm
Copy-Item -Path "dep\vips-dev-8.10\bin\*" -Destination "dist\TissUUmaps\" -Recurse
ISCC.exe build_installer.iss
