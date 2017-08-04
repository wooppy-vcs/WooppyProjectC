Usage steps:

To Train & Evaluate:
1) Run image_file.py script to download and organise raw image data
    - ensure the directory variables at the top of the script has been set accordingly.
    - the output will be of the form:
        **nb** actually it's RAW_IMAGE_DIR not DATA_DIR
        {DATA_DIR}/train/label0
        {DATA_DIR}/train/label1
         ...
        {DATA_DIR}/validation/label0
        {DATA_DIR}/validation/label1
         ...
    - add a labels.txt file to {DATA_DIR} that lists the labels used:
        label0
        label1
         ...
2) Under cv_data.py, change the values for num_examples_per_epoch to reflect the data
3) Run shard_data_to_tfrecord.sh script to convert into TFRecords binary files
    - ensure DATA_DIR variable reflects {DATA_DIR}
    - set train_shards & validation_shards to approximately:
        train_shards = (num train data)/1024
        validation_shards = (num test data)/1024
    - set num_threads to factor of train_shards & validation_shards
    - num_threads allows parallel data conversion, set to a reasonable size
4) Run cv_train_and_eval.sh with the following settings:
    - ensure DATA_DIR, TRAIN_DIR & EVAL_DIR points to the correct paths
    - Set parameters accordingly (notable: NUM_VALID_EXAMPLES)
    - two usages:
        Retraining from pre-trained Inception v3 model on imagenet:
            ./cv_train_and_eval.sh new
            NOTE: ensure you have inception v3 model downloaded

        Continue from a previous checkpoint
            ./cv_train_and_eval.sh