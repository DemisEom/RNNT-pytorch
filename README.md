# RNNT-pytorch
implementing of "EXPLORING RNN-TRANSDUCER FOR CHINESE SPEECH RECOGNITION"
---
## TO DO....
~~1. prediction network에서 one-hot 벡터를 어떻게 임베딩 할것인지.~~  
~~임베딩을 워드 임베딩에 국한해서 생각하는게 아니라 본질적으로는 dimension reduction 이라는걸 생각하자.~~  

~~2. joint네트워크에서 두개의 아웃풋 사이즈를 어떻게 맞출것인지.~~  
    ~~1. joint network에서 각각 FC layer를 거쳐서 사이즈를 통일.~~
 
    
3. loss 부분은 어떻게 할거임...?
    1. warp RNNT Loss 사용(있는거를 쓰자)
    2. 구현하자.
        1. input과 target의 dim, shape는 어떻게 해야할까?

4. learning rate sharping
5. initialize prediction network