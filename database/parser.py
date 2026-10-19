import fitz


def parse_pdf(pdf_path):
    document = fitz.open(pdf_path)

    all_blocks = []

    for page_number, page in enumerate(document):

        blocks = page.get_text("dict")["blocks"]

        for block in blocks:

            if "lines" not in block:
                continue

            for line in block["lines"]:

                line_text = ""

                font = ""
                size = 0
                flags = 0
                bbox = line["bbox"]

                for span in line["spans"]:

                    line_text += span["text"]

                    font = span["font"]

                    size = span["size"]

                    flags = span["flags"]

                line_text = line_text.strip()

                if line_text == "":
                    continue

                all_blocks.append(
                    {
                        "page": page_number + 1,
                        "text": line_text,
                        "font": font,
                        "size": size,
                        "flags": flags,
                        "bbox": bbox
                    }
                )

    document.close()

    return all_blocks