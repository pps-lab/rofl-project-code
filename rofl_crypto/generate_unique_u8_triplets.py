def main():
    num_entries = 900000
    file_name = "unique_u8_triplets.rs"

    with open(file_name, "w") as f:
        f.write("pub static UNIQUE_U8_TRIPLETS: [[u8; 3]; {}] = [\n".format(num_entries))

        for i in range(num_entries):
            # You can replace the following lines with your own logic to generate unique [u8; 3] values.
            entry1 = (i * 3) % 256
            entry2 = (i * 3 + 1) % 256
            entry3 = (i * 3 + 2) % 256
            f.write("[{}, {}, {}],".format(entry1, entry2, entry3))

        f.write("];\n")

if __name__ == "__main__":
    main()
