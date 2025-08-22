## projects 폴더

이 폴더에서 당신의 프로젝트를 저장합니다.
이 폴더는 단순히 train_net.py 와 같은 동작만 수행합니다.
그것에 필요한 config 파일이나 딥러닝 아키텍처는 상위 폴더에서 끌어다 올 뿐입니다.

프로젝트 마다의 딥러닝 모델링은 여기에 저장하지 않습니다.
딥러닝 모델들은 aidriven/modeling에 저장해야 합니다.

aidriven/modeling과 같이 프로젝트와 무관하게 같은 공간에 저장되어 파일이 섞일 염려가 있는 폴더는 아래 예시와 같이 prefix를 통해 구분합니다.

aidriven/modeling/backbone/acc_fastrcnn_r50_fpn.py
aidriven/modeling/backbone/lka_fastrcnn_r50_fpn.py
