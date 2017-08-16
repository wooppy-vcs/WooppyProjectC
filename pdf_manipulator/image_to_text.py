import sys

import pyocr
import pyocr.builders
from PIL import Image

# tools = pyocr.get_available_tools()
# if len(tools) == 0:
#     print("No OCR tool found")
#     sys.exit(1)
# # The tools are returned in the recommended order of usage
# tool = tools[0]
# print("Will use tool '%s'" % (tool.get_name()))
# # Ex: Will use tool 'libtesseract'
#
# langs = tool.get_available_languages()
# print("Available languages: %s" % ", ".join(langs))
# lang = "eng"
# print("Will use lang '%s'" % (lang))
# # Ex: Will use lang 'fra'
# # Note that languages are NOT sorted in any way. Please refer
# # to the system locale settings for the default language
# # to use.
#
# txt = tool.image_to_string(
#     Image.open('pdftoimage.0.png'),
#     lang=lang,
#     builder=pyocr.builders.TextBuilder()
# )
# print("Text done")
# print(txt)
# txt is a Python string

# word_boxes = tool.image_to_string(
#     Image.open('test.jpg'),
#     lang="eng",
#     builder=pyocr.builders.WordBoxBuilder()
# )
# print("Word boxes done")

# print (word_boxes)
# # list of box objects. For each box object:
# #   box.content is the word in the box
# #   box.position is its position on the page (in pixels)
# #
# # Beware that some OCR tools (Tesseract for instance)
# # may return empty boxes

# line_and_word_boxes = tool.image_to_string(
#     Image.open('test.jpg'), lang="eng",
#     builder=pyocr.builders.LineBoxBuilder()
# )
# print("line and word boxes done")

# print (line_and_word_boxes)
# list of line objects. For each line object:
#   line.word_boxes is a list of word boxes (the individual words in the line)
#   line.content is the whole text of the line
#   line.position is the position of the whole line on the page (in pixels)
#
# Beware that some OCR tools (Tesseract for instance)
# may return empty boxes

# Digits - Only Tesseract (not 'libtesseract' yet !)
# digits = tool.image_to_string(
#     Image.open('test-digits.png'),
#     lang=lang,
#     builder=pyocr.tesseract.DigitBuilder()
# )
# digits is a python string


def extract_text_from_image(filepath, language='-'):

    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print("No OCR tool found")
        sys.exit(1)
    # The tools are returned in the recommended order of usage
    tool = tools[0]
    print("Will use tool '%s'" % (tool.get_name()))
    # Ex: Will use tool 'libtesseract'

    langs = tool.get_available_languages()
    print("Available languages: %s" % ", ".join(langs))
    if language == '-':
        lang = "eng"
        print("Using default language '%s'" % lang)
    else:
        lang = language
        print("Will use lang '%s'" % lang)

    # Ex: Will use lang 'fra'
    # Note that languages are NOT sorted in any way. Please refer
    # to the system locale settings for the default language
    # to use.

    extracted_text = tool.image_to_string(
        Image.open(filepath),
        lang=lang,
        builder=pyocr.builders.TextBuilder()
    )
    print("Image to text extraction done")

    return extracted_text
