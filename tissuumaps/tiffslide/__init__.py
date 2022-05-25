"""tiffslide

a somewhat drop-in replacement for openslide-python using tifffile and zarr

"""
from typing import AnyStr
from warnings import warn

from tissuumaps.tiffslide._types import PathOrFileOrBufferLike
from tissuumaps.tiffslide.tiffslide import (
    PROPERTY_NAME_BACKGROUND_COLOR,
    PROPERTY_NAME_BOUNDS_HEIGHT,
    PROPERTY_NAME_BOUNDS_WIDTH,
    PROPERTY_NAME_BOUNDS_X,
    PROPERTY_NAME_BOUNDS_Y,
    PROPERTY_NAME_COMMENT,
    PROPERTY_NAME_MPP_X,
    PROPERTY_NAME_MPP_Y,
    PROPERTY_NAME_OBJECTIVE_POWER,
    PROPERTY_NAME_QUICKHASH1,
    PROPERTY_NAME_VENDOR,
    NotTiffSlide,
    TiffFileError,
    TiffSlide,
)

try:
    from tissuumaps.tiffslide._version import version as __version__
except ImportError:  # pragma: no cover
    __version__ = "not-installed"

__all__ = ["TiffSlide", "TiffFileError"]


def __getattr__(name):  # type: ignore
    """support some drop-in behavior"""
    # alias the most important bits
    if name in {"OpenSlideUnsupportedFormatError", "OpenSlideError"}:
        warn(
            f"compatibility: aliasing tiffslide.TiffFileError to {name!r}", stacklevel=2
        )
        return TiffFileError
    elif name == "OpenSlide":
        # warn(f"compatibility: aliasing tiffslide.TiffSlide to {name!r}", stacklevel=2)
        return TiffSlide
    elif name == "ImageSlide":
        warn(
            f"compatibility: aliasing tiffslide.NotTiffSlide to {name!r}", stacklevel=2
        )
        return NotTiffSlide
    # warn if internals are imported that we dont support
    if name in {"AbstractSlide", "__library_version__"}:
        warn(f"{name!r} is not provided by tiffslide", stacklevel=2)
    raise AttributeError(name)


def open_slide(filename: PathOrFileOrBufferLike[AnyStr]) -> TiffSlide:
    """drop-in helper function

    Note: this will always try to fallback to the NotTiffSlide class
      in case creating a TiffSlide instance fails. If this is undesired
      use `tiffslide.TiffSlide` directly instead of `tiffslide.open_slide`
    """
    try:
        return TiffSlide(filename)
    except TiffFileError as original_err:
        try:
            return NotTiffSlide(filename)
        except Exception as fallback_err:
            # I want this fallback to raise the original error
            # so that a user code can catch TiffFileError.
            # this is bending the __cause__ rules a bit...
            raise original_err from fallback_err