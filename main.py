import logging
import os
import re

import click
import numpy as np
import pytesseract
from pdf2image import convert_from_path

from process_functions import *
from DocScanner import *

# Global variables
verbose_mode = False
basewidth = 1500  # Default image width
custom_config = r'--oem 3 --psm 6'  # Config pytesseract
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def setup(verbose):
    """
    Setup function to set modes before start
    """
    global verbose_mode

    logging.basicConfig(
        format='%(message)s', level=logging.INFO
    )  # message format from logging and changing level to INFO
    if (verbose):
        logging.info('Starting...')

    if (verbose):
        verbose_mode = True
        logging.info("We are in the verbose mode.")


def check_input_path(input):
    """
    Check if the given path input is valid
    if valid path and supoorted type format, return : 0 - '.png', 1- 'jpg' or '.jpeg', 2- '.pdf'
    if valid path but not supported format, return 3
    if not valid path, return 4
    """
    try:  # Check if path exists
        if (os.path.exists(input)):
            if (verbose_mode):
                logging.info("Valid Input Path...")
        else:
            logging.info(
                "Not Valid Input Path"
            )  # Logging info anyway even if verbose_mode is False, to let the user knows
            return 4
    except Exception as e:
        logging.info(f"Exception {e}")
        return 4

    try:
        if (input.lower().endswith('.png')):
            if (verbose_mode):
                logging.info("Your input is a PNG file")
            return 0
        elif (input.lower().endswith('.jpeg')
              or input.lower().endswith('.jpg')):
            if (verbose_mode):
                logging.info("Your input is a JPEG file")
            return 1
        elif (input.lower().endswith('.pdf')):
            if (verbose_mode):
                logging.info("Your input is a PDF file")
            return 2
        else:
            logging.info(
                "NOT supported format"
            )  # Logging info anyway even if verbose_mode is False, to let the user knows
            return 3
    except Exception as e:
        logging.info(f"Exception {e}")
        return 3


def pre_process(input, file_type):
    """"
    Preprocessing Image:
    - Reszing
    - Grayscaling
    - Histogram equalization
    - Sharping
    """
    if verbose_mode:
        logging.info("Loading File...")
    imgs = []
    # read and resize:
    if (file_type < 2):
        img = crop_doc(input)
        imgs.append(resize_image(img, basewidth))
        # imgs.append(resize_image(cv2.imread(input, -1), basewidth))
    else:
        imgs = [
            np.array(pil_image)[:, :, ::-1].copy()
            for pil_image in convert_from_path(input, size=(basewidth, None))
        ]
    if verbose_mode:
        logging.info("Done.\n\nPreprocessing...")
    # grayscale:
    gray_imgs = [get_grayscale(image) for image in imgs]

    # histogram:
    eq_hist_imgs = [equalize_hist(image) for image in gray_imgs]

    # sharping:
    sharp_imgs = [sharping(image) for image in eq_hist_imgs]

    if verbose_mode:
        logging.info("Done.\n")

    return sharp_imgs


def post_process(text):
    def repl(m):
        return '.' * len(m.group()) + ' '

    text = re.sub(r'[^\x00-\x7F]+', '#',
                  text)  # replace non-Ascii with '#' for example
    text = text.replace('-\n', '')  # for splited words by endline
    text = re.sub(r'(\n\s+)', '\n\n',
                  text)  # replace multiple endline with one
    text = re.sub(
        r'(\.)\1{2,}\S+\s+', repl, text
    )  # correcting dotted gaps. Ex: replace '....dfj-kl. ' with '........... '
    return text


def gettext(input, file_type, outfile):
    """"
    To Extract Text, Three steps done:
    1- Pre processing the images.
    2- Extract text with 'image_to_string' from pytesseract.
    3- Post processing the extracted text.
    """
    # Pre_processing file:
    imgs = pre_process(input, file_type)

    # Extracting text:
    if (not outfile.endswith('.txt')):  # In case the output file is not a .txt
        outfile = outfile + '.txt'

    if (verbose_mode):
        if (os.path.exists(outfile)):
            logging.info("Output File name exists")
        else:
            logging.info(f"Creating New file {outfile}")
    f = open(outfile, "w", encoding='utf-8')
    if verbose_mode:
        logging.info("Extracting Text & Post Processing...")
    for i in range(len(imgs)):
        text = pytesseract.image_to_string(imgs[i], lang='eng')
        postproc_text = post_process(text)
        f.write(postproc_text)
    if verbose_mode:
        logging.info("Done.\n")
    f.close()


@click.command()
@click.option("--input", default='./samples/test_scan.jpg')
@click.option("--output", default='default.txt')
@click.option('--verbose', is_flag=True, help="Will print verbose messages.")
def OCRreader(input, output, verbose):
    setup(verbose)  # setup settings before starting
    suff = check_input_path(input)
    if (suff < 3):  # If detected a supported formats:
        gettext(input, suff, output)  # Get Text
    if verbose_mode:
        logging.info("Finished.\n")


if __name__ == '__main__':
    OCRreader()
