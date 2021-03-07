import subprocess
from io import BytesIO
from pathlib import Path

import PySimpleGUI as sg
import pygame
from PIL import Image


def update_network_list(network_path, window):
    # Get list of files in folder
    input_path = Path(network_path)
    if input_path.is_dir():
        fnames = [
            f.name for f in input_path.iterdir()
        ]

        fnames = sorted(fnames, key=lambda x: int(x.split('_')[3][1:]), reverse=True)

        window['-NETWORK-'].update(value=fnames[0], values=fnames)


def gui():
    # First the window layout in 2 columns
    file_list_column = [
        [
            sg.Text("Input directory"),
            sg.In(size=(55, 1), enable_events=True, key="-FOLDER-"),
            sg.FolderBrowse(),
        ],
        [
            sg.Listbox(
                values=[], enable_events=True, size=(75, 20), key="-FILE LIST-"
            )
        ],
    ]

    parameters_column = [
        [
            sg.Text("Key"),
            sg.Combo(values=[], key='-NETWORK-'),
        ],
        [
            sg.Text('Tempo'),
            sg.InputText('80', size=(5, 1), key='-TEMPO-'),
            sg.Text("Key"),
            sg.Combo(values=['C', 'G', 'D', 'A', 'E', 'H', 'Fis', 'Cis', 'F', 'B', 'Es',
                             'As', 'Des', 'Ges', 'Ces'], default_value='C', size=(5, 15), key='-KEY-'),
            sg.VSeperator(),
            sg.Checkbox('Generate?', tooltip='Check if you want to generate MIDI file', key='-GENERATE-'),
            sg.Checkbox('CPU?', tooltip='Check if device should be CPU instead of GPU', key='-CPU-'),
            sg.Button('Execute', tooltip='Executes program with preset settings', key='-EXECUTE-'),
        ],
    ]

    music_list_column = [
        [
            sg.Listbox(
                values=[], enable_events=True, size=(75, 5), key="-MUSIC LIST-"
            ),
            sg.Button('Stop', tooltip='Stops playing music', key='-STOP-'),
        ]
    ]

    image_viewer_column = [
        [sg.Image(key="-IMAGE-")],
    ]

    layout = [
        [
            sg.Column(file_list_column),
            sg.VSeperator(),
            sg.Column(image_viewer_column),
        ],
        [
            sg.Column(parameters_column)
        ],
        [
            sg.Column([
                [sg.Text('Track list')],
            ])
        ],
        [
            sg.Column(music_list_column),
        ]
    ]

    # Create main window
    window = sg.Window("Optical Music Recognition", layout, finalize=True)

    # Default network path
    network_path = r'./yolov5/runs/train'
    # Preload network names combolist
    update_network_list(network_path, window)

    # Default music path
    music_path = r'./music'
    # Preload music list with existing tracks
    update_music_list(music_path, window)

    # Init the music playing library
    freq = 44100  # audio CD quality
    bitsize = -16  # unsigned 16 bit
    channels = 2  # 1 is mono, 2 is stereo
    buffer = 1024  # number of samples
    pygame.mixer.init(freq, bitsize, channels, buffer)
    # optional volume 0 to 1.0
    pygame.mixer.music.set_volume(0.8)

    # Create an event loop
    while True:
        event, values = window.read()

        if event == "-FOLDER-":
            folder = values["-FOLDER-"]

            # Get list of files in folder
            file_list = Path(folder).iterdir()

            fnames = [
                f.name
                for f in file_list
                if f.is_file() and f.name.lower().endswith((".jpg", ".png", ".gif"))
            ]

            window["-FILE LIST-"].update(fnames)

        elif event == "-FILE LIST-":  # A file was chosen from the listbox
            try:
                filename = Path(values['-FOLDER-']) / values["-FILE LIST-"][0]
                with BytesIO() as f:
                    im = Image.open(filename)
                    im.thumbnail((370, 370), Image.ANTIALIAS)
                    im.save(f, format='PNG')
                    window["-IMAGE-"].update(data=f.getvalue())
            except IndexError:
                pass

        elif event == '-EXECUTE-':
            tempo = values['-TEMPO-']
            key = values['-KEY-']
            generate = values['-GENERATE-']
            cpu = values['-CPU-']
            input_path = values['-FOLDER-']
            network_name = values['-NETWORK-']

            if Path(input_path).is_dir():

                if input_path:
                    input_path = f'--path {input_path}'
                else:
                    sg.popup('Path was not specified!', custom_text='Cancel', no_titlebar=True)
                    continue

                if generate:
                    generate = '--generate'
                else:
                    generate = ''

                if network_name:
                    network_name = f'--weights {network_name}'
                else:
                    network_name = ''

                if cpu:
                    cpu = '--cpu'
                else:
                    cpu = ''

                if key:
                    key = f'--key {key}'
                else:
                    key = ''

                if tempo:
                    tempo = f'--tempo {tempo}'
                else:
                    tempo = ''

                cmd = f'python main.py {input_path} --mode user {generate} {cpu} {key} {tempo} {network_name}'

                fail = False
                try:
                    subprocess.check_call(cmd, shell=True)
                except subprocess.CalledProcessError as err:
                    fail = True
                    sg.popup('Error! CUDA unavailable. Check the CPU flag.', custom_text='Ok', no_titlebar=True)

                if not fail:
                    if generate:
                        sg.popup('Music was generated successfully!', custom_text='Ok', no_titlebar=True)
                        update_music_list(music_path, window)
                    else:
                        sg.popup('User data was preprocessed successfully!', custom_text='Ok', no_titlebar=True)

            else:
                continue

        elif event == '-MUSIC LIST-':
            try:
                filename = Path(music_path) / values["-MUSIC LIST-"][0]
                pygame.mixer.music.load(filename)
                pygame.mixer.music.play()
            except IndexError:
                pass

        elif event == '-STOP-':
            pygame.mixer.music.fadeout(500)
            # pygame.mixer.music.stop()

        elif event == sg.WIN_CLOSED:
            break

    window.close()


def update_music_list(music_path, window):
    # Get list of files in folder
    try:
        file_list = Path(music_path).iterdir()
        fnames = [
            f.name
            for f in file_list
            if f.is_file() and f.name.lower().endswith('.mid')
        ]
        window["-MUSIC LIST-"].update(fnames)
    except FileNotFoundError:
        pass


if __name__ == '__main__':
    gui()
