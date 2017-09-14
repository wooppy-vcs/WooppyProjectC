import wand
import numpy as np
import PIL.Image as PILImage
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

    # print(file)

    all_pages = Image(filename=file, resolution=dpi)
    num_pages = len(all_pages.sequence)
    # print("Number of pages: " + str(num_pages))

    # image_folder_id = time.strftime("%Y%m%dT%H%M%S")
    # save_path = destination
    # try_make_dir(os.path.join(save_path))

    saved_image_path_list = []

    for page in range(num_pages):
        single_image = all_pages.sequence[page]
        with Image(single_image) as i:
            i.format = 'png'
            # i.format = 'jpeg'
            i.background_color = Color('white')  # Set white background.
            i.alpha_channel = 'remove'  # Remove transparency and replace with bg.
            # i.save(filename="pdftoimage." + str(page) + ".png")
            # i.save(filename=os.path.join(save_path, "pdftoimage." + str(page) + ".png"))
            save_destination = os.path.join(destination, str(page) + ".png")
            # save_destination = os.path.join(destination, str(page) + ".jpg")
            saved_image_path_list.append(save_destination)
            i.save(filename=save_destination)

    return num_pages, saved_image_path_list


def make_vertical_image(saved_image_path_list, save_image_folder, one_page=False):
    if not one_page:
        imgs = [PILImage.open(i).convert("RGBA") for i in saved_image_path_list]
        min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]

        # hard code this because some cv uses U.S. letter dimension instead of A4. this is A4 pixels at 300dpi.
        # min_shape = (2480, 3508)

        # mid = np.asarray(imgs)
        # mid = (np.asarray(i) for i in imgs)
        # print(mid)
        # imgs_comb = np.vstack(mid)'
        temp = []
        for i in imgs:
            array = np.asarray(i.resize(min_shape))
            # print(array.size)
            # temp.append(np.asarray(i))
            temp.append(array)
        # print(len(temp))
        imgs_comb = np.vstack(temp)
        # imgs_comb = np.vstack((np.asarray(i.resize(min_shape)) for i in imgs))
        imgs_comb = PILImage.fromarray(imgs_comb)

        background = PILImage.new(mode='RGBA', size=(2560, 7020), color=(255, 255, 255, 255))
        bg_w, bg_h = background.size
        offset = (int((bg_w - min_shape[0]) / 2), int((bg_h - min_shape[1]*2) / 2))
        background.paste(imgs_comb, offset)
        vertical_image_path = os.path.join(save_image_folder, 'vertical.png')
        background.save(vertical_image_path)
        # vertical_image_path = os.path.join(save_image_folder, 'vertical.png')
        # imgs_comb.save(vertical_image_path)

    # this is only for CV project, need not pollute the original code
    else:
        imgs = [PILImage.open(i).convert("RGBA") for i in saved_image_path_list]
        min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]

        # hard code this because some cv uses U.S. letter dimension instead of A4. this is A4 pixels at 300dpi.
        # min_shape = (2480, 3508)
        temp = []
        for i in imgs:
            # print(i.size)
            array = np.asarray(i.resize(min_shape))
            # print(array.size)
            # temp.append(np.asarray(i))
            temp.append(array)
        # print(len(temp))

        # make white page
        white_img = PILImage.new(mode="RGBA", size=min_shape, color=(255, 255, 255, 255))
        white_array = np.asarray(white_img)
        temp.append(white_array)

        imgs_comb = np.vstack(temp)
        # imgs_comb = np.vstack((np.asarray(i.resize(min_shape)) for i in imgs))
        imgs_comb = PILImage.fromarray(imgs_comb)

        background = PILImage.new(mode='RGBA', size=(2560, 7020), color=(255, 255, 255, 255))
        bg_w, bg_h = background.size
        offset = (int((bg_w - min_shape[0]) / 2), int((bg_h - min_shape[1] * 2) / 2))
        background.paste(imgs_comb, offset)
        vertical_image_path = os.path.join(save_image_folder, 'vertical.png')
        background.save(vertical_image_path)

        # vertical_image_path = os.path.join(save_image_folder, 'vertical.png')
        # imgs_comb.save(vertical_image_path)

    return vertical_image_path
