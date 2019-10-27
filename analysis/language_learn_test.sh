#!/bin/bash
for s in 1234 2233 3344 5555 12345 828 1123 1991 1992 2018
do
    # # ================================================================================================================================
    # # this part is to train listeners in generation game
    # # train generation listener to learn compositional language with len 2
    # python3 train_models/train_listener.py -s $s -d 2 --iter 500 --eval-freq 1 --save-freq 500 --msg-vocsize 10 --max-msg-len 2 --num-words 2 --max-len-word 5 --train-file data/2_perfect/all_data.txt --dev-file data/2_perfect/all_data.txt --data-file data/2_perfect/all_data.txt --save-dir params/test_learning_speed_0813/listener/comp_2/
    # # train generation to learn compositional language with len 4
    # python3 train_models/train_listener.py -s $s -d 2 --iter 500 --eval-freq 1 --save-freq 500 --msg-vocsize 10 --max-msg-len 4 --num-words 2 --max-len-word 5  --train-file data/2_perfect/train.txt --dev-file data/2_perfect/train.txt --data-file data/2_perfect/train.txt --save-dir params/test_learning_speed_0813/listener/comp_4/
    # # train generation to learn emergent language with len 4
    # python3 train_models/train_listener.py -s $s -d 2 --iter 500 --eval-freq 1 --save-freq 500 --msg-vocsize 10 --max-msg-len 4 --num-words 2 --max-len-word 5  --train-file data/rebuilt_language_2_0712.txt --dev-file data/rebuilt_language_2_0712.txt --data-file data/rebuilt_language_2_0712.txt --save-dir params/test_learning_speed_0813/listener/emergent_4/
    # # train generation to learn holistic language with len 4
    # python3 train_models/train_listener.py -s $s -d 2 --iter 500 --eval-freq 1 --save-freq 500 --msg-vocsize 10 --max-msg-len 4 --num-words 2 --max-len-word 5  --train-file data/2_holistic/train.txt --dev-file data/2_holistic/train.txt --data-file data/2_holistic/train.txt --save-dir params/test_learning_speed_0813/listener/holistic_4/
    # # train generation to learn holistic language with len 2
    # python3 train_models/train_listener.py -s $s -d 2 --iter 500 --eval-freq 1 --save-freq 500 --msg-vocsize 10 --max-msg-len 2 --num-words 2 --max-len-word 5  --train-file data/2_holistic/all_data.txt --dev-file data/2_holistic/all_data.txt --data-file data/2_holistic/all_data.txt --save-dir params/test_learning_speed_0813/listener/holistic_2/
    # # train seq2seq as a baseline of listener
    # python3 train_models/train_seq2seq.py -s $s -d 2 --iter 500 --eval-freq 1 --save-freq 500 --msg-vocsize 10 --max-msg-len 2 --num-words 2 --max-len-word 5  --train-file data/2/all_data.txt --dev-file data/2/all_data.txt --data-file data/2/all_data.txt --save-dir params/test_learning_speed_0813/listener/seq2seq/

    # # ================================================================================================================================
    # # this part is to train listeners in select game
    # # train select listener to learn compositional language with len 4
    # python3 train_models/train_choose_listener.py -s $s -d 2 --iter 150 --eval-freq 1 --save-freq 150 --msg-vocsize 10 --max-msg-len 4 --num-words 2 --max-len-word 5 --train-file data/2_perfect/train.txt --dev-file data/2_perfect/train.txt --data-file data/2_perfect/train.txt --save-dir params/test_learning_speed_0813/choose_listener/comp_4/
    # # train select listener to learn compositional language with len 2
    # python3 train_models/train_choose_listener.py -s $s -d 2 --iter 150 --eval-freq 1 --save-freq 150 --msg-vocsize 10 --max-msg-len 2 --num-words 2 --max-len-word 5 --train-file data/2_perfect/all_data.txt --dev-file data/2_perfect/all_data.txt --data-file data/2_perfect/all_data.txt --save-dir params/test_learning_speed_0813/choose_listener/comp_2/
    # # train select listener to learn emergent language with len 4
    # python3 train_models/train_choose_listener.py -s $s -d 2 --iter 150 --eval-freq 1 --save-freq 150 --msg-vocsize 10 --max-msg-len 4 --num-words 2 --max-len-word 5 --train-file data/rebuilt_language_2_refer_IL.txt --dev-file data/rebuilt_language_2_refer_IL.txt --data-file data/rebuilt_language_2_refer_IL.txt --save-dir params/test_learning_speed_0813/choose_listener/emergent_4/
    # # train select listener to learn emergent language with len 2
    # python3 train_models/train_choose_listener.py -s $s -d 2 --iter 150 --eval-freq 1 --save-freq 150 --msg-vocsize 10 --max-msg-len 2 --num-words 2 --max-len-word 5 --train-file data/rebuilt_language_2_2_refer_IL.txt --dev-file data/rebuilt_language_2_2_refer_IL.txt --data-file data/rebuilt_language_2_2_refer_IL.txt --save-dir params/test_learning_speed_0813/choose_listener/emergent_2/
    # # train select listener to learn holistic language with len 4
    # python3 train_models/train_choose_listener.py -s $s -d 2 --iter 150 --eval-freq 1 --save-freq 150 --msg-vocsize 10 --max-msg-len 4 --num-words 2 --max-len-word 5 --train-file data/2_holistic/train.txt --dev-file data/2_holistic/train.txt --data-file data/2_holistic/train.txt --save-dir params/test_learning_speed_0813/choose_listener/holistic_4/
    # # train select listener to learn holistic language with len 2
    # python3 train_models/train_choose_listener.py -s $s -d 2 --iter 150 --eval-freq 1 --save-freq 150 --msg-vocsize 10 --max-msg-len 2 --num-words 2 --max-len-word 5 --train-file data/2_holistic/all_data.txt --dev-file data/2_holistic/all_data.txt --data-file data/2_holistic/all_data.txt --save-dir params/test_learning_speed_0813/choose_listener/holistic_2/

    # # ================================================================================================================================
    # # this part is to train speaker in both games
    # # train speaker to learn compositional language with len 2
    # python3 train_models/train_speaker.py -s $s -d 2 --iter 500 --eval-freq 1 --save-freq 500 --msg-vocsize 10 --max-msg-len 2 --num-words 2 --max-len-word 5 --train-file data/2_perfect/all_data.txt --dev-file data/2_perfect/all_data.txt --data-file data/2_perfect/all_data.txt --save-dir params/test_learning_speed_0813/speaker/comp_2/
    # # train speaker to learn compositional language with len 4
    # python3 train_models/train_speaker.py -s $s -d 2 --iter 500 --eval-freq 1 --save-freq 500 --msg-vocsize 10 --max-msg-len 4 --num-words 2 --max-len-word 5 --train-file data/2_perfect/train.txt --dev-file data/2_perfect/train.txt --data-file data/2_perfect/train.txt --save-dir params/test_learning_speed_0813/speaker/comp_4/
    # # train speaker to learn emergent language in refer game with len 2
    # python3 train_models/train_speaker.py -s $s -d 2 --iter 500 --eval-freq 1 --save-freq 500 --msg-vocsize 10 --max-msg-len 2 --num-words 2 --max-len-word 5 --train-file data/rebuilt_language_2_2_refer_IL.txt --dev-file data/rebuilt_language_2_2_refer_IL.txt --data-file data/rebuilt_language_2_2_refer_IL.txt --save-dir params/test_learning_speed_0813/speaker/emergent_2/
    # # train speaker to learn emergent language in refer game with len 4
    # python3 train_models/train_speaker.py -s $s -d 2 --iter 500 --eval-freq 1 --save-freq 500 --msg-vocsize 10 --max-msg-len 4 --num-words 2 --max-len-word 5 --train-file data/rebuilt_language_2_refer_IL.txt --dev-file data/rebuilt_language_2_refer_IL.txt --data-file data/rebuilt_language_2_refer_IL.txt --save-dir params/test_learning_speed_0813/speaker/emergent_4_refer/
    # # train speaker to learn emergent language in generation game with len 4
    # python3 train_models/train_speaker.py -s $s -d 2 --iter 500 --eval-freq 1 --save-freq 500 --msg-vocsize 10 --max-msg-len 2 --num-words 2 --max-len-word 5 --train-file data/rebuilt_language_2_0712.txt --dev-file data/rebuilt_language_2_0712.txt --data-file data/rebuilt_language_2_0712.txt --save-dir params/test_learning_speed_0813/speaker/emergent_4_gen/
    # # train speaker to learn holistic language with len 2
    # python3 train_models/train_speaker.py -s $s -d 2 --iter 500 --eval-freq 1 --save-freq 500 --msg-vocsize 10 --max-msg-len 2 --num-words 2 --max-len-word 5 --train-file data/2_holistic/all_data.txt --dev-file data/2_holistic/all_data.txt --data-file data/2_holistic/all_data.txt --save-dir params/test_learning_speed_0813/speaker/holistic_2/
    # # train speaker to learn holistic language with len 4
    # python3 train_models/train_speaker.py -s $s -d 2 --iter 500 --eval-freq 1 --save-freq 500 --msg-vocsize 10 --max-msg-len 4 --num-words 2 --max-len-word 5 --train-file data/2_holistic/train.txt --dev-file data/2_holistic/train.txt --data-file data/2_holistic/train.txt --save-dir params/test_learning_speed_0813/speaker/holistic_4/
    # # train set2seq as baseline
    # python3 train_models/train_set2seq.py -s $s -d 2 --iter 500 --eval-freq 1 --save-freq 500 --msg-vocsize 10 --max-msg-len 4 --num-words 2 --max-len-word 5 --train-file data/2/train.txt --dev-file data/2/train.txt --data-file data/2/train.txt --save-dir params/test_learning_speed_0813/speaker/set2seq/

    # # ================================================================================================================================
    # # this part is to train listener in img select game
    # # train listener to learn compositional language with len 2
    # python3 train_models/train_img_listener.py -s $s -d 2 --iter 150 --eval-freq 1 --save-freq 150 --msg-vocsize 10 --max-msg-len 2 --train-file data/# img_set_25/ --dev-file data/img_set_25/ --num-distractors 14 --data-file ./data/img_languages/holistic0.txt --save-dir ./params/# img_listener_learning/comp/
    # # train listener to learn emergent language with len 2
    # python3 train_models/train_img_listener.py -s $s -d 2 --iter 150 --eval-freq 1 --save-freq 150 --msg-vocsize 10 --max-msg-len 2 --train-file data/# img_set_25/ --dev-file data/img_set_25/ --num-distractors 14 --data-file ./data/img_languages/emergent.txt --save-dir ./params/# img_listener_learning/emergent/
    # # train listener to learn holistic language with len 2
    # python3 train_models/train_img_listener.py -s $s -d 2 --iter 150 --eval-freq 1 --save-freq 150 --msg-vocsize 10 --max-msg-len 2 --train-file data/# img_set_25/ --dev-file data/img_set_25/ --num-distractors 14 --data-file ./data/img_languages/holistic6.txt --save-dir ./params/# img_listener_learning/holistic/

    # ================================================================================================================================
    # this part is to train speaker in img select game
    # train speaker to learn compositional language with len 2
    python3 train_models/train_img_speaker.py -s $s -d 1 --iter 500 --save-freq 500 --eval-freq 1 --msg-vocsize 10 --max-msg-len 2 --save-dir params/test_speaker_learning_speed/imgs/comp --train-file data/img_set_25/ --dev-file data/img_set_25/ --data-file data/img_languages/compositional.txt
    # train speaker to learn emergent language with len 2
    python3 train_models/train_img_speaker.py -s $s -d 1 --iter 500 --save-freq 500 --eval-freq 1 --msg-vocsize 10 --max-msg-len 2 --save-dir params/test_speaker_learning_speed/imgs/emergent --train-file data/img_set_25/ --dev-file data/img_set_25/ --data-file data/img_languages/emergent.txt
    # train speaker to learn holistic language with len 2
    python3 train_models/train_img_speaker.py -s $s -d 1 --iter 500 --save-freq 500 --eval-freq 1 --msg-vocsize 10 --max-msg-len 2 --save-dir params/test_speaker_learning_speed/imgs/holistic --train-file data/img_set_25/ --dev-file data/img_set_25/ --data-file data/img_languages/holistic6.txt

done