=====================================================================================================
A simple readme from Bong
=====================================================================================================
<<ORGANISING DATASET>>
First, we should create 4 txt files from our dataset:
- Test_data.txt
- Training_data.txt
- tags_vocab.txt (dictionary for our tags)
- class_weights.txt (dictionary for our class weights for weighted loss calculation)
To do so, first go to config.yml and scroll to the bottom, change the [datalocalfile][data_folder][path] to the directory
of folder containing all your data. Then also change the [localfile] section to the folder that you want to store the 4 txt files
into.
Next, open data_organiser.py and uncomment the desired classification level (classes/tags e.g. Level-1, AnswerType). Run
the code and you will see the 4 txt files appear in the folder.

--------------------------------------------------------------------------------------------------------
<<TRAINING MODELS>>
You are welcome to tweak the hyperparameters of the training steps (e.g. epoch, batch sizes etc.).
Just run the code and if anything happen, you should contact me or you can debug by yourself, but it's faster to ask me =P.

---------------------------------------------------------------------------------------------------------
<< EVALUATING MODELS>>
 Spot this line "tf.flags.DEFINE_string("checkpoint_dir", "runs/1510924069-Scenario/checkpoints", "Checkpoint directory from training run")"

 Duplicate by using CTRL+D and commentize the previous line, change the name of the folder containing your models checkpoints (1510924069-Scenario)
 in the above case.

 Then run the code.

 You will get the result in P, R and F1 for each class. Remember to copy and paste to a txt file for recording purpose.

 If again, any error, ask me, or debug by yourself. Ofc ask me is better. LOL



