
(function( $ ){
    /**
     * @class GeoTIFFTileSource
     * @classdesc The GeoTIFFTileSource uses a the GeoTIFF.js library to serve tiles from local file or remote URL. Requires GeoTIFF.js.
     *
     * @memberof OpenSeadragon
     * @extends OpenSeadragon.TileSource
     * @param {File|String|Object} input A File object, url string, or object with fields for pre-loaded GeoTIFF and GeoTIFFImages objects
     * @param {Object} opts Options object. To do: how to document options fields?
     *                 opts.logLatency: print latency to fetch and process each tile to console.log or the provided function
     *                 opts.tileWidth: tileWidth to request at each level. Defaults to tileWidth specified by TIFF file or 256 if unspecified by the file
     *                 opts.tileHeight:tileWidth to request at each level. Defaults to tileWidth specified by TIFF file or 256 if unspecified by the file
     *                 
     * @property {Object} GeoTIFF The GeoTIFF.js representation of the underlying file. Undefined until the file is opened successfully
     * @property {Array}  GeoTIFFImages Array of GeoTIFFImage objects, each representing one layer. Undefined until the file is opened successfully
     * @property {Bool}   ready set to true once all promises have resolved
     * @property {Object} promises
     * @property {Number} dimensions
     * @property {Number} aspectRatio
     * @property {Number} tileOverlap
     * @property {Number} tileSize
     * @property {Array}  levels
     */
    $.GeoTIFFTileSource=function( input, opts={logLatency:false} ){
        let self=this;
        this.options = opts;
        // create random unique id
        this.id = Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
        
        // $.TileSource.apply( this, [ {width:1,height:1} ] );
        $.TileSource.apply( this );
        this._ready=false;
        this._pool = new GeoTIFF.Pool();

        this._setupComplete=function(){
            this._ready=true;
            self.promises.ready.resolve();
            self.raiseEvent( 'ready', { tileSource: self } );
        }
        
        if(input.GeoTIFF && input.GeoTIFFImages){
            this.promises={
                GeoTIFF: Promise.resolve(input.GeoTIFF),
                GeoTIFFImages:Promise.resolve(input.GeoTIFFImages),
                ready:DeferredPromise(),
            }
            this.GeoTIFF = input.GeoTIFF;
            // $.TileSource.apply( this, [ {url:'dummy'} ] );
            // $.TileSource.apply( this, [ {width:1,height:1} ] );
            this.imageCount = input.GeoTIFFImages.length;
            this.GeoTIFFImages=input.GeoTIFFImages;
            setupLevels.call(this);
        } else{
            this.promises={
                GeoTIFF: input instanceof File ? GeoTIFF.fromBlob(input) : GeoTIFF.fromUrl(input),
                GeoTIFFImages:DeferredPromise(),
                ready:DeferredPromise(),
            }
            this.promises.GeoTIFF.then(tiff=>{
                self.GeoTIFF = tiff;
                // $.TileSource.apply( this, [{url:'dummy'}] );
                return tiff.getImageCount();
            }).then(count=>{
                self.imageCount = count;
                let promises=[...Array(count).keys()].map(index=>self.GeoTIFF.getImage(index));
                return Promise.all(promises);
            }).then(images=>{
                self.GeoTIFFImages = images;
                self.promises.GeoTIFFImages.resolve(images);
                setupLevels.call(self);
            }).catch(error=>{
                console.error('Re-throwing error with GeoTIFF:',error);
                throw(error);
            });
        }
        
    }

    //Static functions

    //To do: add documentation about what this does (i.e. separates likely subimages into separate GeoTIFFTileSource objects)
    $.GeoTIFFTileSource.getAllTileSources = async function(input, opts){
        let tiff= input instanceof File ? GeoTIFF.fromBlob(input) : GeoTIFF.fromUrl(input);
        return tiff.then(t=>{tiff=t; return t.getImageCount()})
                   .then(c=>Promise.all([...Array(c).keys()].map(index=>tiff.getImage(index))))
                   .then(images=>{
                        // Filter out images with photometricInterpretation.TransparencyMask
                        images = images.filter(image=>image.fileDirectory.photometricInterpretation!==GeoTIFF.globals.photometricInterpretations.TransparencyMask)
                        // Sort by width (largest first), then detect pyramids
                        images.sort((a,b)=>b.getWidth() - a.getWidth());
                        // find unique aspect ratios (with tolerance to account for rounding) 
                        const tolerance = 0.015
                        let aspectRatioSets = images.reduce((accumulator, image)=>{
                            let r = image.getWidth() / image.getHeight();
                            let exists = accumulator.filter(set=>Math.abs(1-set.aspectRatio/r) < tolerance);
                            if(exists.length == 0){
                                let set = {
                                    aspectRatio: r,
                                    images: [image]
                                };
                                accumulator.push(set);
                            } else {
                                exists[0].images.push(image);
                            }
                            return accumulator;
                        }, []);

                        let imagesets = aspectRatioSets.map(set=>set.images);
                        let tilesources = imagesets.map(images=> new $.GeoTIFFTileSource({GeoTIFF:tiff, GeoTIFFImages:images},opts));
                        return tilesources;

                    })
    }

    // Extend OpenSeadragon.TileSource, and override/add prototype functions
    Object.defineProperty($.GeoTIFFTileSource.prototype, "ready", {
        set: function ready(r) {
            //ignore
        },
        get: function ready(){
            return this._ready;
        }
    });
    $.extend( $.GeoTIFFTileSource.prototype, $.TileSource.prototype, /** @lends OpenSeadragon.GeoTIFFTileSource.prototype */{
        
        /**
         * Return the tileWidth for a given level.
         * @function
         * @param {Number} level
         */
         getTileWidth: function (level) {
            if (this.levels.length > level) {
                return this.levels[level].tileWidth;
            }
        },
    
        /**
         * Return the tileHeight for a given level.
         * @function
         * @param {Number} level
         */
        getTileHeight: function (level) {
            if (this.levels.length > level) {
                return this.levels[level].tileHeight;
            }
        },
    
        /**
         * @function
         * @param {Number} level
         */
        getLevelScale: function ( level ) {
            // console.log('getLevelScale')
            var levelScale = NaN;
            if ( this.levels.length > 0 && level >= this.minLevel && level <= this.maxLevel ) {
                levelScale =
                    this.levels[ level ].width /
                    this.levels[ this.maxLevel ].width;
            }
            return levelScale;
        },
        
        /**
         * Implement function here instead of as custom tile source in client code
         * @function
         * @param {Number} levelnum
         * @param {Number} x
         * @param {Number} y
         */
        getTileUrl: function ( levelnum, x, y ) {
            // return dataURL from reading tile data from the GeoTIFF object as String object (for cache key) with attached promise 
            let level = this.levels[levelnum];
            // add id to url so that it can be used as a cache key
            let url = new String(`${levelnum}/${x}_${y}_${this.id}`); // use new String() so that custom fields can be set (see url.fetch below)

            url.fetch = ( (ts,level,x,y,src)=> ()=>regionToDataUrl.call(ts, level, x, y, src))(this, level, x, y, url);

            return url;
        },

        //To do: documentation necessary? Kind of an internal function...
        downloadTileStart:function(context){
            context.src.fetch().then(dataURL=>{
                let image = new Image();
                let request=''+context.src;
                image.onload=function(){
                    context.finish(image);
                }
                image.onerror = image.onabort = function(){
                    context.finish(null,request,'Request aborted');
                }
                image.src = dataURL;
            })
        },
        downloadTileAbort:function(context){
            context.src.abortController && context.src.abortController.abort();
        },
        getTileHashKey: function(level, x, y) {
            return this.id + "/" + level + '/' + x + '_' + y;
        },
        

    })

    //private functions

    function regionToDataUrl(level, x, y, src){

        let startTime = this.options.logLatency && Date.now();
        
        let abortController = src.abortController = new AbortController(); // add abortController to the src object so OpenSeadragon can abort the request
        let abortSignal = abortController.signal;
        let w = level.tileWidth;
        let h = level.tileHeight;
        let window = [x*w, y*h, (x+1)*w, (y+1)*h].map(v=>Math.round(v * level.scalefactor));//scale the requested tile to layer image coordinates

        // Use getTileOrStrip followed by converters because it is noticably more efficient than readRGB
        return level.image.readRGB(
                {window: window,
                interleave:true,
                pool:this._pool,
                width:level.tileWidth,
                height:level.tileHeight,
                signal:abortSignal
                }).then(raster=>{
                let data = new Uint8ClampedArray(raster);
                // convert RGB data to RGBA
                

                let canvas = document.createElement('canvas');
                canvas.width = level.tileWidth;
                canvas.height = level.tileHeight;
                let ctx = canvas.getContext('2d');

                let photometricInterpretation = level.image.fileDirectory.PhotometricInterpretation;
                let arr = new Uint8ClampedArray(4*canvas.width*canvas.height);
                let rgb = new Uint8ClampedArray(data);
                let i, a;
                for(i=0, a=0;i<rgb.length; i+=3, a+=4){
                    arr[a]=rgb[i];
                    arr[a+1]=rgb[i+1];
                    arr[a+2]=rgb[i+2];
                    arr[a+3]=255;
                }
                
                ctx.putImageData(new ImageData(arr,canvas.width,canvas.height), 0, 0);
                
                let dataURL = canvas.toDataURL('image/jpeg',0.8);
                this.options.logLatency && (typeof this.options.logLatency=='function' ? this.logLatency : console.log)('Tile latency (ms):', Date.now() - startTime)
                return dataURL;
            })
    }

    // Adapted from https://github.com/geotiffjs/geotiff.js
    class Converters{
        static RGBAfromYCbCr(input) {
            const rgbaRaster = new Uint8ClampedArray(input.length * 4 / 3);
            let i, j;
            for (i = 0, j = 0; i < input.length; i += 3, j += 4) {
                const y = input[i];
                const cb = input[i + 1];
                const cr = input[i + 2];
            
                rgbaRaster[j] = (y + (1.40200 * (cr - 0x80)));
                rgbaRaster[j + 1] = (y - (0.34414 * (cb - 0x80)) - (0.71414 * (cr - 0x80)));
                rgbaRaster[j + 2] = (y + (1.77200 * (cb - 0x80)));
                rgbaRaster[j + 3] = 255;
            }
            return rgbaRaster;
        }
        static RGBAfromRGB(input) {
            const rgbaRaster = new Uint8ClampedArray(input.length * 4 / 3);
            let i, j;
            for (i = 0, j = 0; i < input.length; i += 3, j += 4) {
                rgbaRaster[j] = input[i];
                rgbaRaster[j + 1] = input[i+1];
                rgbaRaster[j + 2] = input[i+2];
                rgbaRaster[j + 3] = 255;
            }
            return rgbaRaster;
        }

        static RGBAfromWhiteIsZero(input, max) {
            const rgbaRaster = new Uint8ClampedArray(input.length * 4);
            let value;
            for (let i = 0, j = 0; i < input.length; ++i, j += 3) {
                value = 256 - (input[i] / max * 256);
                rgbaRaster[j] = value;
                rgbaRaster[j + 1] = value;
                rgbaRaster[j + 2] = value;
                rgbaRaster[j + 3] = 255;
            }
            return rgbaRaster;
        }
          
        static RGBAfromBlackIsZero(input, max) {
            const rgbaRaster = new Uint8ClampedArray(input.length * 4);
            let value;
            for (let i = 0, j = 0; i < input.length; ++i, j += 3) {
                value = input[i] / max * 256;
                rgbaRaster[j] = value;
                rgbaRaster[j + 1] = value;
                rgbaRaster[j + 2] = value;
                rgbaRaster[j + 3] = 255;
            }
            return rgbaRaster;
        }
          
        static RGBAfromPalette(input, colorMap) {
            const rgbaRaster = new Uint8ClampedArray(input.length * 4);
            const greenOffset = colorMap.length / 3;
            const blueOffset = colorMap.length / 3 * 2;
            for (let i = 0, j = 0; i < input.length; ++i, j += 3) {
                const mapIndex = input[i];
                rgbaRaster[j] = colorMap[mapIndex] / 65536 * 256;
                rgbaRaster[j + 1] = colorMap[mapIndex + greenOffset] / 65536 * 256;
                rgbaRaster[j + 2] = colorMap[mapIndex + blueOffset] / 65536 * 256;
                rgbaRaster[j + 3] = 255;
            }
            return rgbaRaster;
        }
          
        static RGBAfromCMYK(input) {
            const rgbaRaster = new Uint8ClampedArray(input.length);
            for (let i = 0, j = 0; i < input.length; i += 4, j += 4) {
                const c = input[i];
                const m = input[i + 1];
                const y = input[i + 2];
                const k = input[i + 3];
            
                rgbaRaster[j] = 255 * ((255 - c) / 256) * ((255 - k) / 256);
                rgbaRaster[j + 1] = 255 * ((255 - m) / 256) * ((255 - k) / 256);
                rgbaRaster[j + 2] = 255 * ((255 - y) / 256) * ((255 - k) / 256);
                rgbaRaster[j + 3] = 255;
            }
            return rgbaRaster;
        }
          
        static RGBAfromCIELab(input) {
            // from https://github.com/antimatter15/rgb-lab/blob/master/color.js
            const Xn = 0.95047;
            const Yn = 1.00000;
            const Zn = 1.08883;
            const rgbaRaster = new Uint8ClampedArray(input.length * 4 / 3);
          
            for (let i = 0, j = 0; i < input.length; i += 3, j += 4) {
                const L = input[i + 0];
                const a_ = input[i + 1] << 24 >> 24; // conversion from uint8 to int8
                const b_ = input[i + 2] << 24 >> 24; // same
            
                let y = (L + 16) / 116;
                let x = (a_ / 500) + y;
                let z = y - (b_ / 200);
                let r;
                let g;
                let b;
            
                x = Xn * ((x * x * x > 0.008856) ? x * x * x : (x - (16 / 116)) / 7.787);
                y = Yn * ((y * y * y > 0.008856) ? y * y * y : (y - (16 / 116)) / 7.787);
                z = Zn * ((z * z * z > 0.008856) ? z * z * z : (z - (16 / 116)) / 7.787);
            
                r = (x * 3.2406) + (y * -1.5372) + (z * -0.4986);
                g = (x * -0.9689) + (y * 1.8758) + (z * 0.0415);
                b = (x * 0.0557) + (y * -0.2040) + (z * 1.0570);
            
                r = (r > 0.0031308) ? ((1.055 * (r ** (1 / 2.4))) - 0.055) : 12.92 * r;
                g = (g > 0.0031308) ? ((1.055 * (g ** (1 / 2.4))) - 0.055) : 12.92 * g;
                b = (b > 0.0031308) ? ((1.055 * (b ** (1 / 2.4))) - 0.055) : 12.92 * b;
            
                rgbaRaster[j] = Math.max(0, Math.min(1, r)) * 255;
                rgbaRaster[j + 1] = Math.max(0, Math.min(1, g)) * 255;
                rgbaRaster[j + 2] = Math.max(0, Math.min(1, b)) * 255;
                rgbaRaster[j + 3] = 255;
            }
            return rgbaRaster;
        }
    }
    


    function setupLevels(){
        if(this.ready){
            return;
        }

        let images = this.GeoTIFFImages.sort((a,b)=>b.getWidth() - a.getWidth());
        
        //default to 256x256 tiles, but defer to options passed in
        let defaultTileWidth=defaultTileHeight=256;

        //the first image is the highest-resolution view (at least, with the largest width)
        let fullWidth = this.width = images[0].getWidth();
        let fullHeight= this.height = images[0].getHeight();
        this.tileOverlap = 0;
        this.minLevel = 0;
        this.aspectRatio = this.width/this.height;
        this.dimensions  = new $.Point( this.width, this.height );

        //a valid tiled pyramid has strictly monotonic size for levels
        let pyramid=images.reduce((acc,im)=>{
            if(acc.width!==-1){
                acc.valid = acc.valid && im.getWidth()<acc.width;//ensure width monotonically decreases
            }
            acc.width=im.getWidth();
            return acc;
        },{valid:true, width:-1});

        if(pyramid.valid){
            this.levels = images.map((image)=>{
                let w = image.getWidth();
                let h = image.getHeight();
                return {
                    width:w,
                    height:h,
                    tileWidth:this.options.tileWidth || image.getTileWidth() || defaultTileWidth,
                    tileHeight:this.options.tileHeight || image.getTileHeight() || defaultTileHeight,
                    image:image,
                    scalefactor:1,
                }
            })
            this.maxLevel = this.levels.length - 1;
        }
        else{
            let numPowersOfTwo= Math.ceil(Math.log2(Math.max(fullWidth/defaultTileWidth, fullHeight/defaultTileHeight)));
            let levelsToUse = [...Array(numPowersOfTwo).keys()].filter(v=>v%2==0);//use every other power of two for scales in the "pyramid" 

            this.levels = levelsToUse.map(levelnum=>{
                let scale = Math.pow(2,levelnum)
                let image = images.filter(im=>im.getWidth()*scale >= fullWidth).slice(-1)[0];//smallest image with sufficient resolution
                return {
                    width:fullWidth/scale,
                    height:fullHeight/scale,
                    tileWidth:this.options.tileWidth || image.getTileWidth() || defaultTileWidth,
                    tileHeight:this.options.tileHeight || image.getTileHeight() || defaultTileHeight,
                    image:image,
                    scalefactor:scale*image.getWidth()/fullWidth,
                }
            })
            this.maxLevel = this.levels.length - 1;
        }
        this.levels = this.levels.sort((a,b)=>a.width - b.width);   
        
        
        this._tileWidth  = this.levels[0].tileWidth;
        this._tileHeight = this.levels[0].tileHeight;

        this._setupComplete();
    }

    function DeferredPromise(){
        let self=this;
        let promise=new Promise((resolve,reject)=>{
            self.resolve=resolve;
            self.reject=reject;
        })
        promise.resolve=self.resolve;
        promise.reject=self.reject;
        return promise;
    }
    

})(OpenSeadragon)


