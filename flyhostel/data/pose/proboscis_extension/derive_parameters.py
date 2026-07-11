from .proboscis_candidates import load_and_derive_parameters

# ==========================================================================
if __name__ == "__main__":

    with open("files.txt", "r") as h:
        files = [l.strip() for l in h]

    load_and_derive_parameters(files)

