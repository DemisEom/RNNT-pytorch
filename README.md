RNNT-pytorch  
---
Implementation of "EXPLORING RNN-TRANSDUCER FOR CHINESE SPEECH RECOGNITION"

Installation
---
0. pip isntall -r requirments.txt
1. Install torch  
2. Install rnnt loss  
[hawk aron's implementation](https://github.com/HawkAaron/warp-transducer/tree/master/pytorch_binding)  
```
## ref hawk aron's read me
git clone https://github.com/HawkAaron/warp-transducer
cd warp-transducer
mkdir build; cd build
cmake ..
make

cd pytorch_binding
python setup.py install
```  
3. install torch audio

TO DO Things
---
1. learning rate sharping
2. initialize prediction network
3. 특징 추출을 두가지 방식으로 작동되게 하기.
    1. mel filterbank
4. 차원 점검하기.
5. RNNTLOSS 가져오는거 세팅 편리하게 하기.
6. 불필요한 파서들 제거.
    
    
References
---
1. [EXPLORING RNN-TRANSDUCER FOR CHINESE SPEECH RECOGNITION](https://arxiv.org/pdf/1811.05097.pdf)
2. [speech, RNNT Loss by awni](https://github.com/awni/speech)
3. [E2E-ASR by hawk aron](https://github.com/HawkAaron/E2E-ASR)