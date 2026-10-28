CTRL_TOKENS = {
    (0, 0): 0b1101010100,
    (0, 1): 0b0010101011,
    (1, 0): 0b0101010100,
    (1, 1): 0b1010101011,
}

def _ones(x: int, nbits: int) -> int:
    return bin(x & ((1 << nbits) - 1)).count("1")

class TmdsEncoder:
    def __init__(self):
        self.cnt = 0

    def encode(self, de: int, d: int, c0: int, c1: int) -> int:
        if not de:
            self.cnt = 0
            return CTRL_TOKENS[(c1, c0)]

        n1_d = _ones(d, 8)
        use_xnor = n1_d > 4 or (n1_d == 4 and (d & 1) == 0)
        qm = [d & 1]
        for i in range(1, 8):
            bit = qm[i - 1] ^ ((d >> i) & 1)
            qm.append(1 - bit if use_xnor else bit)
        qm8 = 0 if use_xnor else 1
        qm_byte = sum(b << i for i, b in enumerate(qm))

        n1 = _ones(qm_byte, 8)
        n0 = 8 - n1
        inv = (~qm_byte) & 0xFF
        if self.cnt == 0 or n1 == n0:
            q9 = 1 - qm8
            low = qm_byte if qm8 else inv
            self.cnt += (n1 - n0) if qm8 else (n0 - n1)
        elif (self.cnt > 0 and n1 > n0) or (self.cnt < 0 and n0 > n1):
            q9 = 1
            low = inv
            self.cnt += 2 * qm8 + (n0 - n1)
        else:
            q9 = 0
            low = qm_byte
            self.cnt += -2 * (1 - qm8) + (n1 - n0)
        return (q9 << 9) | (qm8 << 8) | low

def tmds_decode(q: int):
    for key, tok in CTRL_TOKENS.items():
        if q == tok:
            return "ctrl", key
    low = q & 0xFF
    if (q >> 9) & 1:
        low = (~low) & 0xFF
    qm8 = (q >> 8) & 1
    d = low & 1
    for i in range(1, 8):
        bit = ((low >> i) & 1) ^ ((low >> (i - 1)) & 1)
        if not qm8:
            bit ^= 1
        d |= bit << i
    return "data", d
