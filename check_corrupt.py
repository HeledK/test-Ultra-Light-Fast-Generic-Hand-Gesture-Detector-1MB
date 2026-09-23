fpath = r"data\wider_face_add_lm_10_10\ImageSets\Main\trainval.txt"

with open(fpath, "rb") as f:
    content = f.read()

lines = content.split(b"\n")
for i, line in enumerate(lines):
    if b"\x00" in line:
        print(f"Line {i} has null byte: {repr(line)}")