import os
import sys
from types import TracebackType
from typing import IO, Any, AnyStr, Optional, Type, Union

if sys.version_info >= (3, 8):
    from typing import Protocol, runtime_checkable
else:
    from typing_extensions import Protocol
    from typing_extensions import runtime_checkable

from fsspec import AbstractFileSystem

__all__ = [
    "PathOrFileOrBufferLike",
    "OpenFileLike",
    "TiffFileIO",
]

if sys.version_info >= (3, 9):
    PathLikeAnyStr = os.PathLike[AnyStr]
else:
    PathLikeAnyStr = os.PathLike


@runtime_checkable
class OpenFileLike(Protocol):
    """minimal fsspec open file type"""

    fs: AbstractFileSystem
    path: str

    def __enter__(self) -> IO[AnyStr]:
        ...

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        ...


@runtime_checkable
class TiffFileIO(Protocol):
    """minimal stream io for use in tifffile.FileHandle"""

    def seek(self, offset: int, whence: int = 0) -> int:
        ...

    def tell(self) -> int:
        ...

    def read(self, n: int = -1) -> AnyStr:
        ...

    def readinto(self, __buffer: Any) -> Optional[int]:
        ...


PathOrFileOrBufferLike = Union[AnyStr, PathLikeAnyStr, OpenFileLike, TiffFileIO]
