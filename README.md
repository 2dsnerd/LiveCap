## This Project Was Created With Help From Claude and Qwen3.5 Along With Touch Ups by Me  |  Windows and Linux Binaries Coming Soon

# livecaption - offline live captions for Linux/Windows.

Captures system audio (whatever you hear) or your microphone, transcribes it
locally with faster-whisper, and shows rolling captions in a draggable,
always-on-top overlay (or in the terminal).

# Requirements
## Linux
```bash
pip install --user --break-system-packages faster-whisper
```
```bash
sudo apt install pulseaudio-utils python3-tk numpy   # parec/pactl + tkinter
```

## Windows - Coming Soon

Usage - All Can Be Done Within The Settings Menu As Well
-----
```bash
python3 livecaption.py                  # captions for system audio
```
```bash
python3 livecaption.py --source mic     # captions for your microphone
```
```bash
python3 livecaption.py --no-gui         # terminal only
```
```bash
python3 livecaption.py --list-sources   # show audio sources
```
```bash
python3 livecaption.py --reset-config   # forget saved settings
```
Overlay controls
----------------
  Drag with left mouse button  - move
  
  Double-click / right-click   - settings and menu

<br>
Settings you "Save" are stored in ~/.config/livecaption/config.json.
Command-line flags override saved settings for that run.
