/**
 * @file OSDViewerUtils.js Interface to ask information to OSD
 * @author Leslie Solorzano
 * @see {@link OSDViewerUtils}
 */

/**
 * @namespace OSDViewerUtils
 */
OSDViewerUtils={
  _currentZoom:0,
  _currentPan:0,
}

/** 
 * Get viewport maximum zoom
 * @param {string} overlay Object prefix identifying the desired viewport in case there is more than one.
 * Established at {@link tmapp} but can be called directly, for example @example OSDViewerUtils.getMaxZoom("ISS");  */
OSDViewerUtils.getMaxZoom=function(overlay){
  return tmapp[overlay+"_viewer"].viewport.getMaxZoom();

}

/** 
 * Get current viewport zoom
 * @param {string} overlay Object prefix identifying the desired viewport in case there is more than one.
 * Established at {@link tmapp} but can be called directly, for example @example OSDViewerUtils.getMaxZoom("ISS");  */
OSDViewerUtils.getZoom=function(overlay){
  return tmapp[overlay+"_viewer"].viewport.getZoom();
}

/** 
 * Get image width. For now it only brings the size of the main image */
OSDViewerUtils.getImageWidth=function(){
  var op=tmapp["object_prefix"];
  return tmapp[op+"_viewer"].world.getItemAt(0).getContentSize().x;
}

/** 
 * Get image width. For now it only brings the size of the main image */
OSDViewerUtils.getImageHeight=function(){
  var op=tmapp["object_prefix"];
  return tmapp[op+"_viewer"].world.getItemAt(0).getContentSize().y;
}

/** 
 * Add a new image on top of the main viewer. It is mandatory to have the same tile size for this to work. Currently only in main viewer */
OSDViewerUtils.addTiledImage=function(options){
  if(!options){var options={}};
  var replace= options.replace || false;
  var tileSource = options.tileSource || false;
  var op=tmapp["object_prefix"];
  //get zoom
  OSDViewerUtils._currentZoom=tmapp[op+"_viewer"].viewport.getZoom();
  //get center
  OSDViewerUtils._currentPan=tmapp[op+"_viewer"].viewport.getCenter();
  options.success=function(){
      tmapp[op+"_viewer"].viewport.zoomTo(OSDViewerUtils._currentZoom,null, true);
      tmapp[op+"_viewer"].viewport.panTo(OSDViewerUtils._currentPan, true);
  };

  tmapp[op+"_viewer"].addTiledImage(options);
}
