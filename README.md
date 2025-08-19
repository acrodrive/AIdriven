# ADAS Development with Detectron2

이 프로젝트는 **Detectron2**를 기반으로 하여 자율주행 보조 시스템(ADAS)을 구현하는 것을 목표로 합니다.

## 1차 목표
- **Faster R-CNN ResNet-50 FPN 3x** 백본을 사용
- Head를 확장하여 **3D 박스 회귀(x, y, z, w, l, h, ψ)** 수행
- 시계열 데이터를 활용하여 **RNN 계열 (예: LSTM, GRU)** 기반 추론 적용
- 차량의 **가속/감속 제어 출력** 생성

## 2차 목표
- Lane Detection 및 **주행 가능 공간 인식**
- 이를 기반으로 **조향 제어(steering)**까지 확장

---

본 프로젝트는 차량 주변 객체 인식과 주행 제어를 통합하여 **지능형 ADAS 시스템**을 연구/개발하는 것을 지향합니다.