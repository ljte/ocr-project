import argparse
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import cv2 as cv2
import easyocr.imgproc
import pymupdf

EasyOcrExtractedData = tuple[list[list[float]], str, float]


@dataclass
class Point:
    x: float
    y: float


@dataclass
class Bbox:
    top_left: Point
    bottom_right: Point
    width: float
    height: float
    text: str
    confidence: float | None = None

    @classmethod
    def from_extracted_tuple(cls, data: EasyOcrExtractedData) -> "Bbox":
        points, text, conf = data
        tl, _, br, _ = points
        top_left = Point(float(tl[0]), float(tl[1]))
        bottom_right = Point(float(br[0]), float(br[1]))
        return cls(
            top_left=top_left,
            bottom_right=bottom_right,
            width=bottom_right.x - top_left.x,
            height=bottom_right.y - top_left.y,
            text=text,
            confidence=float(conf),
        )

    @classmethod
    def from_batch(cls, data: list[EasyOcrExtractedData]) -> list["Bbox"]:
        return [cls.from_extracted_tuple(bbox) for bbox in data]


@dataclass
class Page:
    number: int
    content: bytes

    def as_nparray(self) -> np.ndarray:
        return np.frombuffer(self.content, np.uint8)

    def grayscale(self) -> np.ndarray:
        return cv2.cvtColor(
            cv2.imdecode(self.as_nparray(), -1),
            cv2.COLOR_BGR2GRAY,
        )


DocumentPages = dict[int, list[Bbox]]


@dataclass
class Document:
    name: str
    pages: DocumentPages = field(default_factory=dict)

    def save_to(self, outfile: str) -> None:
        with open(outfile, "w") as f:
            json.dump(asdict(self), f)


def iter_pages(document: pymupdf.Document) -> Iterable[Page]:
    for page in document:
        yield Page(
            number=page.number + 1,
            content=page.get_pixmap().tobytes(),
        )


def extract_bboxes_from_page(reader: easyocr.Reader, img: np.ndarray) -> list[Bbox]:
    return Bbox.from_batch(reader.readtext(img))


def extract_bboxes(reader: easyocr.Reader, document: pymupdf.Document) -> DocumentPages:
    print(f"Processing document {document.name}. Number of pages: {len(document)}.")
    pages = {}
    for page in iter_pages(document):
        pages[page.number] = extract_bboxes_from_page(
            reader=reader,
            img=page.grayscale(),
        )
        print(f"Page {page.number}/{len(document)} processed.")
    return pages


def process_document(reader: easyocr.Reader, document: str) -> Document:
    with pymupdf.open(document) as pdf:
        return Document(
            name=document,
            pages=extract_bboxes(
                reader=reader,
                document=pdf,
            ),
        )


def main():
    argparser = argparse.ArgumentParser(
        prog="Document Extraction",
        description="Extracts data from pdf documents",
        usage="extraction.py document.pdf",
    )
    argparser.add_argument("document")
    argparser.add_argument("-l", "--languages", action="append", default=["en"])
    argparser.add_argument("-o", "--outfile")
    args = argparser.parse_args()

    outfile = args.outfile or Path(args.document).with_suffix(".json")
    try:
        reader = easyocr.Reader(lang_list=args.languages, gpu=False)
        (
            process_document(
                reader=reader,
                document=args.document,
            )
            .save_to(outfile=outfile)
        )

        print(
            f"{args.document} has been processed. Extracted data is saved to {outfile}."
        )
    except pymupdf.FileDataError:
        print(f"Can't open {args.document}. It is either corrupted or not a pdf.")
    except ValueError as err:
        if "is not supported" in err.args:
            languages = err.args[0]

            print(f"Languages are not supported: {', '.join(languages)}")
            return
        print(err)



if __name__ == "__main__":
    main()
