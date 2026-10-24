"""Illustrate selected versus changed bits on a synthetic FP16 value."""

import struct

from fi_campaign.words import mutate_word


def main():
    value = 1.0
    word = int.from_bytes(struct.pack("<e", value), "little")
    bits = (9, 10, 14)
    print(f"Input: {value}; FP16 encoding: 0x{word:04x}; selected positions: {bits}")
    for mode in ("flip", "stuck_at_0", "stuck_at_1"):
        updated, changed = mutate_word(word, bits, mode)
        result = struct.unpack("<e", updated.to_bytes(2, "little"))[0]
        print(f"{mode:12s} 0x{updated:04x}  value={result:<10g} "
              f"selected={len(bits)}  changed={changed}")


if __name__ == "__main__":
    main()
