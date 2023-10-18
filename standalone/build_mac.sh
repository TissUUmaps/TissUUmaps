# install dependencies
brew install vips

rm -Rf build dist

pyinstaller ./TissUUmaps.spec --noconfirm
export VIPS_PATH=$(brew --prefix vips)
cp -rf $VIPS_PATH/lib/libvips.42*.dylib dist/TissUUmaps.app/Contents/MacOS/

# Install Mac DMG Creator
Brew install build-dmg

# Copy app to separate dmg folder
mkdir -p dist/dmg && cp -r dist/TissUUmaps.app dist/dmg
# Create disk image with the app
create-dmg \
  --volname "TissUUmaps" \
  --volicon "../tissuumaps/static/misc/design/logo.png" \
  --window-pos 200 120 \
  --window-size 600 300 \
  --icon-size 100 \
  --hide-extension "TissUUmaps.app" \
  --app-drop-link 0 0 \
  "dist/TissUUmaps_arm64.dmg" \
  "dist/dmg/"

brew remove vips
