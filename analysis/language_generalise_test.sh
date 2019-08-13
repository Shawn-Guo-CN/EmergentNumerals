#!/bin/bash
for s in 1234 2233 3344 5555 12345 828 1123 1991 1992 2018
do
    # ================================================================================================================================
    # this part is to train listeners in generation game
    # train generation listener to learn compositional language with len 4
    python3 train_models/train_listener.py -s $s -d 3 -b 1024 --iter 300 --eval-freq 1 --save-freq 300 --msg-vocsize 10 --max-msg-len 4 --num-words 4 --max-len-word 9 --train-file data/4_perfect/train.txt --dev-file data/4_perfect/dev.txt --data-file data/4_perfect/all_data.txt --save-dir params/test_language_generalise_0813/generate/comp_4/
    # train generation listener to learn holistic language with len 4
    python3 train_models/train_listener.py -s $s -d 3 -b 1024 --iter 300 --eval-freq 1 --save-freq 300 --msg-vocsize 10 --max-msg-len 4 --num-words 4 --max-len-word 9 --train-file data/4_holistic/train.txt --dev-file data/4_holistic/dev.txt --data-file data/4_holistic/all_data.txt --save-dir params/test_language_generalise_0813/generate/holistic_4/
    # train generation listener to learn emergent language from generate game with len 8
    python3 train_models/train_listener.py -s $s -d 3 -b 1024 --iter 300 --eval-freq 1 --save-freq 300 --msg-vocsize 10 --max-msg-len 8 --num-words 4 --max-len-word 9 --train-file data/4_emergent_gen/train.txt --dev-file data/4_emergent_gen/dev.txt --data-file data/4_emergent_gen/all_data.txt --save-dir params/test_language_generalise_0813/generate/emergent_gen_8/
    # train generation listener to learn emergent language from refer game with len 8
    python3 train_models/train_listener.py -s $s -d 3 -b 1024 --iter 300 --eval-freq 1 --save-freq 300 --msg-vocsize 10 --max-msg-len 8 --num-words 4 --max-len-word 9 --train-file data/4_emergent_refer/train.txt --dev-file data/4_emergent_refer/dev.txt --data-file data/4_emergent_refer/all_data.txt --save-dir params/test_language_generalise_0813/generate/emergent_refer_8/
    # train seq2seq as a baseline of listener
    python3 train_models/train_seq2seq.py -s $s -d 3 -b 1024 --iter 300 --eval-freq 1 --save-freq 300 --msg-vocsize 10 --max-msg-len 2 --num-words 4 --max-len-word 0  --train-file data/4/train.txt --dev-file data/4/dev.txt --data-file data/4/all_data.txt --save-dir params/test_language_generalise_0813/generate/seq2seq/

    # ================================================================================================================================
    # this part is to train listeners in select game
    # train select listener to learn compositional language with len 4
    python3 train_models/train_choose_listener.py -s $s -d 3 --iter 10 --eval-freq 1 --save-freq 10 --msg-vocsize 10 --max-msg-len 4 --num-words 4 --max-len-word 9 --train-file data/4_perfect/train.txt --dev-file data/4_perfect/dev.txt --data-file data/4_perfect/all_data.txt --save-dir params/test_language_generalise_0813/select/comp_4/
    # train select listener to learn emergent language from refer game with len 8
    python3 train_models/train_choose_listener.py -s $s -d 3 --iter 10 --eval-freq 1 --save-freq 10 --msg-vocsize 10 --max-msg-len 8 --num-words 4 --max-len-word 9 --train-file data/4_emergent_refer/train.txt --dev-file data/4_emergent_refer/dev.txt --data-file data/4_emergent_refer/all_data.txt --save-dir params/test_language_generalise_0813/select/emergent_refer_8/
    # train select listener to learn emergent language from generate game with len 8
    python3 train_models/train_choose_listener.py -s $s -d 3 --iter 10 --eval-freq 1 --save-freq 10 --msg-vocsize 10 --max-msg-len 8 --num-words 4 --max-len-word 9 --train-file data/4_emergent_gen/train.txt --dev-file data/4_emergent_gen/dev.txt --data-file data/4_emergent_gen/all_data.txt --save-dir params/test_language_generalise_0813/select/emergent_gen_8/
    # train select listener to learn holistic language with len 4
    python3 train_models/train_choose_listener.py -s $s -d 3 --iter 10 --eval-freq 1 --save-freq 10 --msg-vocsize 10 --max-msg-len 4 --num-words 4 --max-len-word 9 --train-file data/4_holistic/train.txt --dev-file data/4_holistic/dev.txt --data-file data/4_holistic/all_data.txt --save-dir params/test_language_generalise_0813/select/holistic_4/

done