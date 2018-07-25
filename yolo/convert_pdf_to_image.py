from pdf_manipulator import wand_pdf_to_image
from pdf_manipulator import pdf_to_text

import os
import shutil

raw_training_data_folder_directory = os.path.join('raw_pdf_folder')
pdf_and_image_folder_directory = os.path.join('converted_folder')


def next_incremental_folder_path(parent_folder_path):
    i = 1
    try_path = os.path.join(parent_folder_path, str(i))

    while os.path.exists(try_path):
        i += 1
        try_path = os.path.join(parent_folder_path, str(i))

    pdf_path = os.path.join(try_path, 'pdf')
    image_path = os.path.join(try_path, 'image')
    text_path = os.path.join(try_path, 'text')
    label_path = os.path.join(try_path, 'label')
    pdf_filename_path = os.path.join(pdf_path, str(i)+'.pdf')

    os.makedirs(try_path)
    os.makedirs(pdf_path)
    os.makedirs(image_path)
    os.makedirs(text_path)
    os.makedirs(label_path)

    return try_path, pdf_filename_path

for file in os.listdir(raw_training_data_folder_directory):
    filename = os.fsdecode(file)
    print(filename)

    if filename.endswith('.pdf'):
        current_file_directory = os.path.join(raw_training_data_folder_directory, filename)
        current_training_data_destination_path, current_training_data_pdf_destination_path = next_incremental_folder_path(pdf_and_image_folder_directory)
        current_training_data_image_destination_path = os.path.join(current_training_data_destination_path, 'image')
        current_training_data_text_destination_path = os.path.join(current_training_data_destination_path, 'text')
        current_training_data_label_destination_path = os.path.join(current_training_data_destination_path, 'label')

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

        # Try to extract text. Removes trailing whitespaces
        extracted_text = pdf_to_text.extract_text(files=[current_file_directory]).strip()
        save_destination = os.path.join(current_training_data_text_destination_path, "data" + ".txt")
        destination_filename = os.path.join(current_training_data_destination_path, filename[:-4]  + ".txt")
        destination_multilabel = os.path.join(current_training_data_label_destination_path, "multilabel" + ".txt")
        # save_destination_root = os.path.join(pdf_and_image_folder_directory, filename[:-4] + ".txt")
        with open(save_destination, "w+") as f:
            f.write(str(extracted_text))
            f.close()

        # with open(save_destination_root, "w+") as f:
        #     f.write(str(extracted_text))
        #     f.close()

        with open(destination_filename, "w+") as f:
            f.close()

        with open(destination_multilabel, "w+") as f:
            f.write(
                    "Administration:	0\n"
                    "Audit:	0\n"
                    "FinanceAndAccounting:	0\n"
                    "HealthAndSafety:	0\n"
                    "HumanResources:	0\n"
                    "IT:	0\n"
                    "Legal:	0\n"
                    "OperationsConsultingManufacturing:	0\n"
                    "Others:	0\n"
                    "SalesAndMarketing:	0\n"
                    "StrategyAndBusinessDevelopment:	0\n"
                    "TrainingAndLearning:	0\n"
            )
            f.close()