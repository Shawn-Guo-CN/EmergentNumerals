# Baseline Models
## 1. Seq2Seq
Accuracy
 * 100%

CONF
 - hidden size: 256
 - voc size: 6
 - ...

LOSS
 * 0.000



---
## 2. Set2Seq
Accuracy
 * 100%

CONF
 - hidden size: 256
 - voc size: 6
 - ...

LOSS
 * 0.000

---
## 3. Set2Seq+Seq2Seq with Softmax
Accuracy
 * 100%

CONF
 - hidden size: 256
 - voc size: 6
 - MSG voc size: 10
 - MSG length: 6
 - ...

---
## 4. Set2Seq+Seq2Seq with Gumbel (Soft)
Accuracy
 * 83.4984%

CONF
 - hidden size: 256
 - voc size: 6
 - MSG voc size: 10
 - MSG length: 6
 - TAU in training: 2
 - TAU in testing: 0.5
 - ...

---
## 5. Set2Seq+Seq2Seq with Gumbel (Hard)
Accuracy
 * 87.4916%

CONF
 - hidden size: 256
 - voc size: 6
 - MSG voc size: 10
 - MSG length: 6
 - TAU in training: 2
 - TAU in testing: 0.1
 - ...
