from optparse import OptionParser

from . import views

def main ():
    parser = OptionParser(usage='Usage: %prog [options] [slide-directory]')
    parser.add_option('-B', '--ignore-bounds', dest='DEEPZOOM_LIMIT_BOUNDS',
                default=False, action='store_false',
                help='display entire scan area')
    parser.add_option('-c', '--config', metavar='FILE', dest='config',
                help='config file')
    parser.add_option('-d', '--debug', dest='DEBUG', action='store_true',
                help='run in debugging mode (insecure)')
    parser.add_option('-e', '--overlap', metavar='PIXELS',
                dest='DEEPZOOM_OVERLAP', type='int',
                help='overlap of adjacent tiles [1]')
    parser.add_option('-f', '--format', metavar='{jpeg|png}',
                dest='DEEPZOOM_FORMAT',
                help='image format for tiles [jpeg]')
    parser.add_option('-l', '--listen', metavar='ADDRESS', dest='host',
                default='127.0.0.1',
                help='address to listen on [127.0.0.1]')
    parser.add_option('-p', '--port', metavar='PORT', dest='port',
                type='int', default=5000,
                help='port to listen on [5000]')
    parser.add_option('-Q', '--quality', metavar='QUALITY',
                dest='DEEPZOOM_TILE_QUALITY', type='int',
                help='JPEG compression quality [75]')
    parser.add_option('-s', '--size', metavar='PIXELS',
                dest='DEEPZOOM_TILE_SIZE', type='int',
                help='tile size [254]')
    parser.add_option('-D', '--depth', metavar='LEVELS',
                dest='FOLDER_DEPTH', type='int',
                help='folder depth search for opening files [4]')

    (opts, args) = parser.parse_args()
    # Load config file if specified
    if opts.config is not None:
        views.app.config.from_pyfile(opts.config)
        
    # Overwrite only those settings specified on the command line
    for k in dir(opts):
        if not k.startswith('_') and getattr(opts, k) is None:
            delattr(opts, k)
    views.app.config.from_object(opts)

    try:
        views.app.config['SLIDE_DIR'] = args[0]
    except IndexError:
        pass

    views.app.run(host=opts.host, port=opts.port, threaded=True, debug=False)

if __name__ == '__main__':
    main ()