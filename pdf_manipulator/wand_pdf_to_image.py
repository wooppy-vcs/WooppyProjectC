import wand
import os
import time
from wand.image import Image
from wand.display import display
from wand.color import Color
from tensorflow.python.platform import gfile


def try_make_dir(try_path):
    if not gfile.Exists(try_path):
        os.makedirs(try_path)


# multiple pages PDF
# all_pages = Image(filename='Lim Si Yan Resume - Liek Xi Bong.pdf', resolution=300)
# # all_pages = Image(filename='test.pdf', resolution=300)
# image_folder_id = time.strftime("%Y%m%dT%H%M%S")
# save_path = os.path.join('uploads', image_folder_id, 'image')
# try_make_dir(os.path.join(save_path))
# for page in range(2):
#     single_image = all_pages.sequence[page]  # Just work on first page
#     print(single_image.page_height)
#     with Image(single_image) as i:
#         # i.format = 'png'
#         i.format = 'jpeg'
#         # i.background_color = Color('red')  # Set white background.
#         # i.alpha_channel = 'remove'  # Remove transparency and replace with bg.
#         # i.save(filename="pdftoimage." + str(page) + ".png")
#         # i.save(filename="pdftoimage." + str(page) + ".jpg")
#         i.save(filename=os.path.join(save_path, "pdftoimage." + str(page) + ".jpg"))


def save_as_image(file, destination, dpi=300):
    if file is None:
        print("No pdf file found for image extraction.")
        return None

    print(file)

    all_pages = Image(filename=file, resolution=dpi)
    num_pages = len(all_pages.sequence)
    print("Number of pages: " + str(num_pages))

    # image_folder_id = time.strftime("%Y%m%dT%H%M%S")
    # save_path = destination
    # try_make_dir(os.path.join(save_path))

    for page in range(num_pages):
        single_image = all_pages.sequence[page]
        with Image(single_image) as i:
            i.format = 'png'
            # i.format = 'jpeg'
            i.background_color = Color('white')  # Set white background.
            i.alpha_channel = 'remove'  # Remove transparency and replace with bg.
            # i.save(filename="pdftoimage." + str(page) + ".png")
            # i.save(filename=os.path.join(save_path, "pdftoimage." + str(page) + ".png"))
            i.save(filename=os.path.join(destination, str(page) + ".png"))

    return num_pages
