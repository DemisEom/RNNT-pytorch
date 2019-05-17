# RNNT-pytorch
implementaion RNN tranceducer for personal study
---
## TO DO....
~~1. prediction network에서 one-hot 벡터를 어떻게 임베딩 할것인지.~~  
    - 이거는 해결.
    - 임베딩을 워드 임베딩에 국한해서 생각하는게 아니라 본질적으로는 dimension reduction 이라는걸 생각하자.  

2. joint네트워크에서 두개의 아웃풋 사이즈를 어떻게 맞출것인지.  
    1. joint network에서 각각 FC layer를 거쳐서 사이즈를 통일.
    2. 각 아웃풋을 파라미터 조정해서 사이즈를 맞춘다.
    
3. loss 부분은 어떻게 할거임...?
    1. 일단 앞에 두개 해결하고 보자....