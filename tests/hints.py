from typing import Protocol, TypeVar

from typing_extensions import Self

from locus._core.hints import Scalar as _Scalar


class Scalar(_Scalar, Protocol):
    def __abs__(self, /) -> Self: ...


ScalarT = TypeVar('ScalarT', bound=Scalar)
