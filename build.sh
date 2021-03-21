pyinstaller --noconsole --add-data "templates:templates" --add-data "static:static"  --icon="static/misc/favicon.ico" ./flasktissuumaps.py --noconfirm
mv dist/flasktissuumaps/PyQt5/Qt dist/flasktissuumaps/PyQt5/Qt5
mkdir -p ~/.local/share/applications/
cp ./TissUUmaps.desktop ~/.local/share/applications/

