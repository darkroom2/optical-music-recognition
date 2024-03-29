# Optical Music Recognition

Aplikacja odtwarzająca melodię na podstawie rozpoznanych na zdjęciu nut muzycznych.

Wymagane:
-----

* [Python](https://www.python.org) >= 3.8.0
* [YOLOv5](https://github.com/ultralytics/yolov5)

Instalacja:
---

```sh
git clone https://github.com/darkroom2/omr2.git
cd omr2
python -m pip install --upgrade midiutil more_itertools opencv-python pysimplegui pygame
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
```

Uruchomienie:
---
Dla użytkownika końcowego (wersja okienkowa):
```sh
python gui.py
```
Dla administratora (możliwość samodzielnego uczenia sieci i zmiany parametrów uczenia):
```sh
python main.py [-h] [--mode MODE] [--path PATH] [--tempo TEMPO] [--key KEY] [--generate] [--cpu]
```

optional arguments:\
  ```-h, --help```     Show this help message and exit\
  ```--mode MODE```    User mode or admin mode (admin can train network)\
  ```--path PATH```    Path to user image files\
  ```--tempo TEMPO```  Tempo in beats per minute\
  ```--key KEY```      The key of the track\
  ```--generate```     Detects & generates music\
  ```--cpu```          Changes device from gpu to cpu
  
Działanie:
---
[![OMR in action](https://img.youtube.com/vi/qjuRwqIHdXc/0.jpg)](https://www.youtube.com/watch?v=qjuRwqIHdXc)
