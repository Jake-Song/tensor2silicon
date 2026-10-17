"""Independently check BF16 shared layouts and block rasterization on CPU."""


def a_offset(row, col, bk):
    shift = ((row >> 1) if bk == 32 else row) & (bk // 8 - 1)
    return row * bk + (col ^ (shift * 8))


def b_offset(row, col, bn):
    shift = ((row >> 1) if bn == 32 else row) & min(7, bn // 8 - 1)
    return row * bn + (col ^ (shift * 8))


def check():
    cases = 0
    for bm, bn in [(32, 32), (32, 64), (64, 32), (64, 64), (64, 128), (128, 64), (128, 128), (128, 256), (256, 128)]:
        for bk in [32, 64]:
            for rows, cols, offset in [(bm, bk, lambda r, c: a_offset(r, c, bk)),
                                       (bk, bn, lambda r, c: b_offset(r, c, bn))]:
                addresses = [offset(r, c) for r in range(rows) for c in range(cols)]
                assert sorted(addresses) == list(range(rows * cols))
                # Each global 16-byte vector remains an aligned contiguous vector.
                for row in range(rows):
                    for col in range(0, cols, 8):
                        start = offset(row, col)
                        assert start % 8 == 0
                        assert [offset(row, col + i) for i in range(8)] == list(range(start, start + 8))
                # An LDMATRIX 8x8 matrix reads eight 16-byte rows. Their four-bank
                # groups must cover all 32 banks once for this 128-byte wavefront.
                for row in range(0, rows, 8):
                    for col in range(0, cols, 8):
                        banks = []
                        for r in range(row, row + 8):
                            first_bank = (offset(r, col) * 2 // 4) % 32
                            banks.extend((first_bank + j) % 32 for j in range(4))
                        assert sorted(banks) == list(range(32))
                cases += 1
    for grid_y in [1, 3, 8, 9, 17, 32]:
        for grid_x in [1, 3, 7, 32]:
            for group in [1, 2, 4, 8, 16]:
                actual = []
                for linear in range(grid_x * grid_y):
                    per = group * grid_x
                    first = linear // per * group
                    rows = min(grid_y - first, group)
                    by = first + (linear % per) % rows
                    bx = (linear % per) // rows
                    actual.append((by, bx))
                assert sorted(actual) == [(y, x) for y in range(grid_y) for x in range(grid_x)]
                cases += 1
    # The hoisted-address path XORs a byte address for A and adds a row stride
    # for B. Check every lane and K step, including nonzero shared base offsets.
    for bk in [32, 64]:
        for bn in [64, 128, 256]:
            for base in [0, 128, 1024, 65536]:
                for stage in range(4):
                    for lane in range(32):
                        for m in range(4):
                            row = (lane & 15) + 16 * m
                            col = lane // 16 * 8
                            addr = base + 128 * bk * 2 * stage + 2 * a_offset(row, col, bk)
                            for step in range(bk // 16):
                                assert addr ^ (step * 32) == base + 128 * bk * 2 * stage + 2 * a_offset(row, col + 16 * step, bk)
                        for n in range(4):
                            row, col = lane & 15, n * 16 + lane // 16 * 8
                            addr = base + stage * bk * bn * 2 + 2 * b_offset(row, col, bn)
                            for step in range(bk // 16):
                                assert addr + step * 16 * bn * 2 == base + stage * bk * bn * 2 + 2 * b_offset(row + step * 16, col, bn)
                cases += 1
    return cases


if __name__ == '__main__':
    print(f'{check()} shared-layout/rasterization cases PASS')
