pyinstaller.exe --noconsole --add-data "templates;templates" --add-data "static;static"  --icon="static/misc/favicon.ico" .\flasktissuumaps.py --noconfirm
Rename-Item -Path dist\flasktissuumaps\PyQt5\Qt -NewName Qt5
