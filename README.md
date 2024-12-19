# 상공회의소 부산기술교육센터 인텔교육 2기

## Clone code 

```shell
git clone --recurse-submodules https://github.com/pskcci/DX-01
```

* `--recurse-submodules` option 없이 clone 한 경우, 아래를 통해 submodule update

```shell
git submodule update --init --recursive
```

## Preparation

### Git LFS(Large File System)

* 크기가 큰 바이너리 파일들은 LFS로 관리됩니다.

* git-lfs 설치 전

```shell
# Note bin size is 132 bytes before LFS pull

$ find ./ -iname *.bin|xargs ls -l
-rw-rw-r-- 1 <ID> <GROUP> 132 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 132 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 132 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
-rwxrwxr-x 1 <ID> <GROUP> 132 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
```

* git-lfs 설치 후, 다음의 명령어로 전체를 가져 올 수 있습니다.

```shell
$ sudo apt install git-lfs

$ git lfs pull
$ find ./ -iname *.bin|xargs ls -l
-rw-rw-r-- 1 <ID> <GROUP> 3358630 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 3358630 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 8955146 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
-rwxrwxr-x 1 <ID> <GROUP> 8955146 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
```

### 환경설정

* [Ubuntu](./doc/environment/ubuntu.md)
* [OpenVINO](./doc/environment/openvino.md)
* [OTX](./doc/environment/otx.md)

## Team projects

### 제출방법

1. 팀구성 및 프로젝트 세부 논의 후, 각 팀은 프로젝트 진행을 위한 Github repository 생성

2. [doc/project/README.md](./doc/project/README.md)을 각 팀이 생성한 repository의 main README.md로 복사 후 팀 프로젝트에 맞게 수정 활용

3. 과제 제출시 `인텔교육 2기 Github repository`에 `New Issue` 생성. 생성된 Issue에 하기 내용 포함되어야 함.

    * Team name : Project Name
    * Project 소개
    * 팀원 및 팀원 역활
    * Project Github repository
    * Project 발표자료 업로드

4. 강사가 생성한 `Milestone`에 생성된 Issue에 추가 

### 평가방법

* [assessment-criteria.pdf](./doc/project/assessment-criteria.pdf) 참고

### 제출현황

<!-- 이건 주석 시작입니다. 여기부터 "끝"주석까지 복사하여 사용해주세요. -->
### Team: TOST (Team of Safe T)
<프로젝트 요약>

산업 현장에서 발생할 수 있는 다양한 위험 상황을 실시간으로 탐지하고, 구조 활동을 지원하는 감시 시스템을 구축하는 프로그램 개발 프로젝트

* Members
  | Name | Role |
  |----|----|
  | 박경규 | Project Leader , 프로젝트 기획 및 관리, 기술/비기술 요인 업무 담당 및 개발 통괄 |
  | 김범규 | Lead Developer, 주요 알고리즘 설계 및 구현,AI 모델링 적용 및 최적화, 아키텍처 설계 및 통합 개발 |
  | 양영준 | Sub Developer, 서브 기능 개발 및 유지 보수, 개발 업무 수행 및 지원, 디버깅 및 최적화 |
  | 장해준 | AI Engineer, 머신러닝 모델 검색/학습, 모델 환경 구축 및 파이프 파인 설계 및 평가 |
* Project Github : https://github.com/creator928/IndustrialSafetyAiCctv
* 발표자료 : https://github.com/creator928/IndustrialSafetyAiCctv/blob/main/draft_ISAC.pdf
<!-- 여기가 주석 끝입니다. -->

### Team: 쥬엘
<프로젝트 요약>
평범한 일상생활에서 시력이 안좋은 사람들이 겪을 수 있는 상황 중 장애물 탐지와 제품 인식 및 정보 제공을 통해 불편함을 개선하는 보조vision 프로젝트
* Members
  | Name | Role |
  |----|----|
  | 문재웅 | 조장 : mono-depth를 활용한 장애물 감지 및 회피 위치 안내, 조원 개발 코드 통합 |
  | 김형석 | 조원 : 제품 인식을 위한 데이터 annotation 및 학습 |
  | 김헌우 | 조원 : hand-estimation을 활용한 user 손 인식 및 제품-손동작 인식 시 제품정보 제공 |
  | 정수빈 | 조원 : 제품 인식을 위한 데이터 셋 수집 및 학습 |
* Project Github : https://github.com/spotSide/projJewel
* 발표자료 : https://github.com/spotSide/projJewel/blob/main/process/2%EC%A1%B0%EB%B0%9C%ED%91%9C%EC%9E%90%EB%A3%8C.odp

### Team: JOJO
<프로젝트 요약>
초보자를 위한 피아노 학습 보조 프로그램입니다.

사용자가 연주하고자하는 악보에 대한 건반 안내를 가상건반에서 제공하고
AI 모델로 학습된 건반 계이름 인식 데이터를 활용하여 비교하여 
올바른 연주를 안내하는 시스템을 개발하는 것을 목표로 합니다.

* Members
  | Name | Role |
  |----|----|
  | 양승용 | 조장 : 가상키보드 및 UI 구현 및 프로젝트 관리 |
  | 오태영 | 조원 : 기본 클래스 구현 및 패키지 구조 설계 |
  | 박철우 | 조원 : Object Detection 모델을 이용한 건반 데이터 학습 |
  | 최은호 | 조원 : 악보 데이터와 웹캠에서 추출한 데이터 동기화 구현 |
* Project Github : https://github.com/syyang0127/TEAM_JOJO
* 발표자료 : https://github.com/syyang0127/TEAM_JOJO
