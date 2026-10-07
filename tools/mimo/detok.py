"""Token ids -> text with the vocabulary in a MiMo GGUF (GPT-2 byte-level BPE).  usage: detok.py <gguf shard 1> id id ..."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from gguf_reader import GGUFFile  # noqa: E402


def byte_decoder():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def main():
    toks = GGUFFile(pathlib.Path(sys.argv[1])).metadata["tokenizer.ggml.tokens"]
    dec = byte_decoder()
    out = bytearray()
    for t in sys.argv[2:]:
        s = toks[int(t)]
        out += s.encode("utf-8") if s.startswith("<|") else bytes(dec.get(c, 63) for c in s)
    sys.stdout.buffer.write(bytes(out) + b"\n")


if __name__ == "__main__":
    main()
