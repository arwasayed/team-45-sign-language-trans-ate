This task is to translate sign language in real time.
The model was trained on a database consisting of 25 words, for each punch 30, and 30 frames were taken from each video.

you should download this .rar file to down load data set and be able to test molde 
https://drive.google.com/file/d/1Yg_bNChDVrZ2tLn_eObdsKz8Q4ggDH03/view?usp=drive_link

If you want to add more words to data set, all you have to do is open the collectData.py file and put the new words in the newactions array newActions=np.array(['new word']), then open cmd and execute this commanded

python collectData.py collect

To train the model, all you have to do is open cmd and type this command
python train.py train

To test the model, all you have to do is open cmd and type this command
python test.py tes
