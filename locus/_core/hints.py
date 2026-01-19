from abc import abstractmethod
from typing import Protocol, TypeVar

from ground.hints import Scalar as _Scalar


class HasCustomRepr(Protocol):
    @abstractmethod
    def __repr__(self, /) -> str:
        raise NotImplementedError


class Scalar(_Scalar, Protocol):
    def __floor__(self, /) -> int: ...


ScalarT = TypeVar('ScalarT', bound=Scalar)
