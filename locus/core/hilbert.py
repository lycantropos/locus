SQUARE_SIZE = 2 ** 16
MAX_COORDINATE = SQUARE_SIZE - 1


def index(x: int, y: int) -> int:
    # based on https://github.com/rawrunprotected/hilbert_curves
    a = x ^ y
    b = MAX_COORDINATE ^ a
    c, d = (MAX_COORDINATE ^ (x | y),
            x & (y ^ MAX_COORDINATE))
    a, b, c, d = (a | (b >> 1),
                  (a >> 1) ^ a,
                  ((c >> 1) ^ (b & (d >> 1))) ^ c,
                  ((a & (c >> 1)) ^ (d >> 1)) ^ d)
    a, b, c, d = (((a & (a >> 2)) ^ (b & (b >> 2))),
                  ((a & (b >> 2)) ^ (b & ((a ^ b) >> 2))),
                  c ^ ((a & (c >> 2)) ^ (b & (d >> 2))),
                  d ^ ((b & (c >> 2)) ^ ((a ^ b) & (d >> 2))))
    a, b, c, d = (((a & (a >> 4)) ^ (b & (b >> 4))),
                  ((a & (b >> 4)) ^ (b & ((a ^ b) >> 4))),
                  c ^ ((a & (c >> 4)) ^ (b & (d >> 4))),
                  d ^ ((b & (c >> 4)) ^ ((a ^ b) & (d >> 4))))
    c ^= ((a & (c >> 8)) ^ (b & (d >> 8)))
    d ^= ((b & (c >> 8)) ^ ((a ^ b) & (d >> 8)))
    a, b = c ^ (c >> 1), d ^ (d >> 1)
    i0 = x ^ y
    i1 = b | (MAX_COORDINATE ^ (i0 | a))
    return (interleave(i1) << 1) | interleave(i0)


def interleave(value: int) -> int:
    value = (value | (value << 8)) & 0x00FF00FF
    value = (value | (value << 4)) & 0x0F0F0F0F
    value = (value | (value << 2)) & 0x33333333
    return (value | (value << 1)) & 0x55555555
