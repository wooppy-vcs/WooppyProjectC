from pdf_manipulator import wand_pdf_to_image

import os
import shutil

raw_training_data_folder_directory = os.path.join('raw_pdf_folder')
pdf_and_image_folder_directory = os.path.join('pdf_and_image_folder')


def next_incremental_folder_path(parent_folder_path):
    i = 1
    try_path = os.path.join(parent_folder_path, str(i))

    while os.path.exists(try_path):
        i += 1
        try_path = os.path.join(parent_folder_path, str(i))

    pdf_path = os.path.join(try_path, 'pdf')
    image_path = os.path.join(try_path, 'image')
    label_path = os.path.join(try_path, 'label')
    pdf_filename_path = os.path.join(pdf_path, str(i)+'.pdf')

    os.makedirs(try_path)
    os.makedirs(pdf_path)
    os.makedirs(image_path)
    os.makedirs(label_path)

    return try_path, pdf_filename_path


for file in os.listdir(raw_training_data_folder_directory):
    filename = os.fsdecode(file)
    print(filename)

    if filename.endswith('.pdf'):
        current_file_directory = os.path.join(raw_training_data_folder_directory, filename)
        current_training_data_destination_path, current_training_data_pdf_destination_path = next_incremental_folder_path(pdf_and_image_folder_directory)
        current_training_data_image_destination_path = os.path.join(current_training_data_destination_path, 'image')

        num_pages, saved_image_path_list = wand_pdf_to_image.save_as_image(file=current_file_directory,
                                                                           destination=current_training_data_image_destination_path)
        if num_pages < 2:
            vertical_image_path = wand_pdf_to_image.make_vertical_image(saved_image_path_list=saved_image_path_list,
                                                                        save_image_folder=current_training_data_image_destination_path,
                                                                        one_page=True)
        else:
            vertical_image_path = wand_pdf_to_image.make_vertical_image(saved_image_path_list=saved_image_path_list[:2],
                                                                        save_image_folder=current_training_data_image_destination_path)
        shutil.copy2(current_file_directory, current_training_data_pdf_destination_path)
