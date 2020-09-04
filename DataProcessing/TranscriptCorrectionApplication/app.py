import json
import os
from os import path
import tkinter
from tkinter import *
import tkinter.filedialog as filedialog

import pygame
from mutagen.mp3 import MP3
from distutils.dir_util import copy_tree

tk = Tk()
tk.geometry("800x500")
global visits



class MyApp:

    def __init__(self, parent):

        # Get Visits
        global visits
        visits = self.getVisitsData()
        print('visits', visits)

        # Files Button List
        self.files_button_list = []

        # Pygame init
        self.basename = ''
        self.basepath = ''
        # pygame.init()
        # pygame.mixer.init()
        # self.paused = pygame.mixer.music.get_busy()
        self.player = None  # Music player
        self.track = None  # Audio file
        self.trackLength = None  # Audio file length
        self.message_var = StringVar()
        self.message_var.set('No File Loaded')

        # Layout Setup
        Grid.rowconfigure(parent, 0, pad=5, weight=20)
        Grid.rowconfigure(parent, 1, pad=5, weight=1)
        Grid.columnconfigure(parent, 0, pad=5, weight=1)
        Grid.columnconfigure(parent, 1, pad=5, weight=1)

        # Create & Configure frame
        self.browser_frame = Frame(parent, bd=1, relief=RAISED)
        self.browser_frame.grid(row=0, column=0, sticky=N + S + E + W)

        Grid.columnconfigure(self.browser_frame, 0, pad=1, weight=1)
        Grid.rowconfigure(self.browser_frame, 0, pad=1, weight=20)
        Grid.rowconfigure(self.browser_frame, 1, pad=1, weight=1)

        load_dir_button = Button(self.browser_frame, text="Load JSON Dir")
        load_dir_button.grid(row=1, column=0, sticky=E + W)
        load_dir_button.bind("<Button-1>", self.loadJsonDir)

        # Create & Configure frame
        self.dynamic_btn_frame = Frame(self.browser_frame, bd=1, relief=FLAT)
        self.dynamic_btn_frame.grid(row=0, column=0, sticky=N + S + E + W)

        # Create & Configure frame
        frame = Frame(parent, bd=1, relief=RAISED)
        frame.grid(row=0, column=1, sticky=N + S + E + W)

        Grid.columnconfigure(frame, 0, pad=1, weight=1)
        Grid.columnconfigure(frame, 1, pad=1, weight=1)
        Grid.columnconfigure(frame, 2, pad=1, weight=1)
        Grid.columnconfigure(frame, 3, pad=1, weight=1)

        Grid.rowconfigure(frame, 0, pad=1, weight=1)
        Grid.rowconfigure(frame, 1, pad=1, weight=20)
        Grid.rowconfigure(frame, 2, pad=1, weight=1)
        Grid.rowconfigure(frame, 3, pad=1, weight=1)

        # PLAY
        play_button = Button(frame, text="Play")
        play_button.grid(row=0, column=0, sticky=E + W)
        play_button.bind("<Button-1>", self.playAudio)

        # PAUSE
        pause_button = Button(frame, text="Stop")
        pause_button.grid(row=0, column=1, sticky=E + W)
        pause_button.bind("<Button-1>", self.stopAudio)

        self.slider_value = DoubleVar()
        self.slider = Scale(frame, to=self.trackLength, orient=HORIZONTAL, length=100,
                            resolution=0.5, showvalue=True, tickinterval=0, digit=4,
                            variable=self.slider_value, command=self.updateSlider)
        # self.slider.grid(row=0, column=2, sticky=E + W)
        self.slider.grid(row=0, column=2, columnspan=2, sticky=E + W)

        # JSON
        self.json_entry = Text(frame, undo=True)
        self.json_entry.grid(row=1, columnspan=4, sticky=N + S + E + W)

        save_button = Button(frame, text="Save JSON")
        save_button.grid(row=2, column=2, sticky=E + W)
        save_button.bind("<Button-1>", self.saveJsonFile)

        close_button = Button(frame, text="Close", command=tk.destroy)
        close_button.grid(row=2, column=3, sticky=E + W)

        export_button = Button(frame, text="Export All")
        export_button.grid(row=3, column=3, sticky=E + W)
        export_button.bind("<Button-1>", self.exportAll)

        # Message Bar
        self.audio_label = Label(parent, textvariable=self.message_var, anchor=W, justify=LEFT)
        self.audio_label.grid(row=1, columnspan=2, sticky=N + S + E + W)

    def loadAudio(self):
        '''Initialise pygame mixer, load audio file and set volume.'''
        player = pygame.mixer
        player.init()
        print("self.track ", self.track)
        player.music.load(self.track)
        player.music.set_volume(20)
        audio = MP3(self.track)
        self.trackLength = audio.info.length

        self.player = player
        self.slider.configure(to=self.trackLength)
        self.stopAudio(None)
        self.slider_value.set(0.0);

    def playAudio(self, event):
        '''Play track from slider location.'''
        # 1. Get slider location.
        # 2. Play music from slider location.
        # 3. Update slider location (use tk's .after loop)
        playtime = self.slider_value.get();
        self.player.music.play(start=playtime);
        self.trackPlay(playtime)

    def trackPlay(self, playtime):
        '''Slider to track the playing of the track.'''
        # 1.When track is playing
        #   1. Set slider position to playtime
        #   2. Increase playtime by interval (1 sec)
        #   3. start TrackPlay loop
        # 2.When track is not playing
        #   1. Print 'Track Ended'
        if self.player is not None:
            if self.player.music.get_busy():
                self.slider_value.set(playtime);
                playtime += 1.0
                self.loopID = tk.after(1000, lambda: self.trackPlay(playtime));
            else:
                print('Track Ended')

    def updateSlider(self, value):
        '''Move slider position when tk.Scale's trough is clicked or when slider is clicked.'''
        if self.player is not None:
            if self.player.music.get_busy():

                tk.after_cancel(self.loopID)  # Cancel PlayTrack loop
                self.stopAudio(None)
                self.slider_value.set(value)  # Move slider to new position
                self.playAudio(None) # Play track from new postion
            else:
                self.slider_value.set(value)  # Move slider to new position

    def stopAudio(self, event):
        '''Stop the playing of the track.'''
        if self.player.music.get_busy():
            self.player.music.stop()

    def loadJsonDir(self, event):
        global visits

        dirpath = filedialog.askdirectory(
            initialdir="",
            title="Select input directory",
        )
        if not dirpath: return
        self.basepath = dirpath
        try:
            directories = [d for d in os.listdir(dirpath) if os.path.isdir(os.path.join(dirpath, d))]
            directories = sorted(directories)
            blank_image = PhotoImage()
            for directory in directories:
                visit = [visit for visit in visits if visit['filepath'] == os.path.join(self.basepath, directory)]
                status_color = '#2E4053'
                if len(visit) > 0:
                    if visit[0]['status'] == 'Completed' and visit[0]['hasVisited'] is True:
                        status_color = '#1E8449'
                    elif visit[0]['status'] == 'Unknown' and visit[0]['hasVisited'] is True:
                        status_color = '#CB4335'

                # image = blank_image
                btn = Button(self.dynamic_btn_frame, text=directory, compound=CENTER,
                             foreground=status_color)
                btn.pack(fill=X)
                self.files_button_list.append(btn)
                index = directory
                btn.bind("<Button-1>", lambda event, i=index: self.loadJsonFile(i))

                dir_path = os.path.join(self.basepath, directory)
                if not any(visit['filepath'] == dir_path for visit in visits):
                    visits.append({
                        'filepath': dir_path,
                        'hasVisited': False,
                        'status': 'Unknown',
                    })
            self.saveVisitsData();
        except Exception as e:
            raise
            tkinter.messagebox.showerror('Error Loading Chart',
                                         'Unable to open file: %r' % filename)

    def loadJsonFile(self, event):
        global visits
        self.basename = event
        try:
            # Remove file extensions
            filename = os.path.join(self.basepath, self.basename + '/transcript.json')
            print(filename)

            with open(filename, 'rb') as infile:
                json_data = json.load(infile)
                dump_data = json.dumps(json_data, sort_keys=True, indent=2)

                self.json_entry.replace(0.0, END, dump_data)
                self.track = os.path.join(self.basepath, self.basename + '/audio.mp3')
                self.loadAudio()
                self.message_var.set('\'' + self.basename + '\' loaded')
                # Change dir color
                for button in self.files_button_list:
                    if os.path.join(self.basepath, self.basename) == os.path.join(self.basepath,
                                                                                  button['text']) and not button.cget(
                        "foreground") == '#1E8449':
                        button.configure(foreground='#CB4335')
                # Mark the dir
                for location in visits:
                    if location['filepath'] == os.path.join(self.basepath, self.basename):
                        location['hasVisited'] = True
                self.saveVisitsData()

        except Exception as e:
            raise
            tkinter.messagebox.showerror('Error Loading Chart',
                                         'Unable to open file: %r' % filename)
    def exportAll(self,event):

        processed_folder = 'Processed'
        unprocessed_folder = 'Unprocessed'
        if not os.path.exists(processed_folder):
            os.makedirs(processed_folder)
        if not os.path.exists(unprocessed_folder):
            os.makedirs(unprocessed_folder)
        counter = 0
        for location in visits:
            counter +=1
            print(location['filepath'],  '  ', os.path.basename(os.path.normpath(location['filepath'])))
            if location['status'] == 'Completed':
                copy_tree(location['filepath'], os.path.join(processed_folder,os.path.basename(os.path.normpath(location['filepath']))))
            else:
                copy_tree(location['filepath'], os.path.join(unprocessed_folder,os.path.basename(os.path.normpath(location['filepath']))))
        print(counter)



    def saveJsonFile(self, event):
        filename = os.path.join(self.basepath, self.basename + '/transcript.json')
        f = open(filename, "w")
        try:
            print('Save', filename)
            text_data = str(self.json_entry.get(0.0, END))
            print(text_data)
            f.write(text_data)
            f.close()
            # Change dir color
            for button in self.files_button_list:
                if os.path.join(self.basepath, self.basename) == os.path.join(self.basepath, button['text']):
                    button.configure(foreground='#1E8449')

            # Mark the dir
            for location in visits:
                if location['filepath'] == os.path.join(self.basepath, self.basename):
                    location['status'] = 'Completed'
            self.saveVisitsData()


        except Exception as e:
            raise
            tkinter.messagebox.showerror('Error Loading Chart',
                                         'Unable to open file: %r' % filename)

    def saveAsJsonFile(self, event):
        filename = filedialog.asksaveasfile(
            initialdir="",
            title="Save file (.json)",
            filetypes=(("json files", "*.json"), ("all files", "*.*"))
        )

        if not filename: return
        try:
            print('Save', filename)
            text_data = str(self.json_entry.get(0.0, END))  # starts from `1.0`, not `0.0`
            filename.write(text_data)
            filename.close()

        except Exception as e:
            raise
            tkinter.messagebox.showerror('Error Loading Chart',
                                         'Unable to open file: %r' % filename)

    def getVisitsData(self):
        global visits
        if path.exists("./visits.json"):
            try:
                with open("./visits.json") as jsonfile:
                    visits = json.load(jsonfile)
                    print('Successfully read visits file')
            except ValueError:
                visits = []
                print('Successfully read empty visits file')
        else:
            visits = []
            self.saveVisitsData()
        return visits

    def saveVisitsData(self):
        global visits
        with open('./visits.json', 'w') as outfile:
            json.dump(visits, outfile)
            print('Successfully created visits file')

#This app was used to correct the transcripts
#Just click 'Load JSON dir' and navigate to `audio_segmentation_transcripts` folder and click OK

myapp = MyApp(tk)
tk.mainloop()
