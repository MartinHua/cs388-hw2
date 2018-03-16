-------------------------------------------------
README : cs388-hw2
-------------------------------------------------

---------------------------
AUTHOR
---------------------------
 Xinrui Hua, UT EID: xh3426
 
-------------------------------
EXECUTING CODE FOR TESTING
-------------------------------
1. After you unzip the project folder I sent you navigate to project path in command line: cd /"path to project folder"/xinrui_hua_xh3426/cs388-hw2/
2. Run in command line: 
   For training:
   baseline: python ./bilstm-pos/pos_bilstm.py /projects/nlp/penn-treebank3/tagged/pos/wsj ./baseline standard train normal
   input: python ./bilstm-pos/pos_bilstm.py /projects/nlp/penn-treebank3/tagged/pos/wsj ./input standard train input
   output: python ./bilstm-pos/pos_bilstm.py /projects/nlp/penn-treebank3/tagged/pos/wsj ./output standard train output
   
   For Tensorboard:
   python /"path to tensorborad py"/ --logdir=./"training dir"/
   
   For testing:
   baseline: python ./bilstm-pos/pos_bilstm.py /projects/nlp/penn-treebank3/tagged/pos/wsj ./baseline standard test normal
   input: python ./bilstm-pos/pos_bilstm.py /projects/nlp/penn-treebank3/tagged/pos/wsj ./input standard test input
   output: python ./bilstm-pos/pos_bilstm.py /projects/nlp/penn-treebank3/tagged/pos/wsj ./output standard test output
   
   
