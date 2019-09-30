# coding: utf-8
import os.path
from Modelizer import *
from ConvertOsim2Biorbd import *
from Converter import *

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog


class ToolMenu(tk.Menu):
    def __init__(self, mainmenubar, *args, ** kwargs):
        tk.Menu.__init__(self, mainmenubar, *args, **kwargs)
        self.mainmenubar = mainmenubar


class MainMenuBar(tk.Menu):
    def __init__(self, parent, *args, ** kwargs):
        tk.Menu.__init__(self, parent, *args, **kwargs)
        self.config(tearoff=0)
        self.parent = parent


class MainWindow(tk.Tk):
    def __init__(self, *args, ** kwargs):
        # inheritance of tk window
        tk.Tk.__init__(self, *args, **kwargs)
        # general window
        self.title("Biorbd Model Converter")
        self.geometry("%dx%d%+d%+d" % (1000, 100, 350, 125))

        # menu bar
        self.main_menu_bar = MainMenuBar(self)

        self.menu1 = ToolMenu(self.main_menu_bar, tearoff=0)
        self.menu1.add_command(label="Open File", command=self.find_path)
        self.menu1.add_command(label="Save File", command=self.fun('Save'))
        self.menu1.add_separator()
        self.menu1.add_command(label="Quit", command=self.quit)
        self.main_menu_bar.add_cascade(label="File", menu=self.menu1)

        self.menu2 = ToolMenu(self.main_menu_bar, tearoff=0)
        self.menu2.add_command(label="Read file", command=self.analyse, state=tk.DISABLED)
        self.main_menu_bar.add_cascade(label="Read", menu=self.menu2)

        self.menu3 = ToolMenu(self.main_menu_bar, tearoff=0)
        self.menu3.add_command(label="About", command=self.fun('*'))
        self.main_menu_bar.add_cascade(label="Help", menu=self.menu3)

        self.config(menu=self.main_menu_bar)

        # Initializing
        self.file_type = 'None'
        self.is_checked = False
        self.original_path = ''
        self.biorbd_path = ''
        self.initial_dir = os.getcwd()
        self.filename = ''
        self.check_status = 'file not found'
        self.file_exist = False
        self.unknown_extension = False
        self.model = None
        self.is_analyzed = False

        # General frame for file management
        self.general_frame = tk.Frame(self, height=100)
        self.general_frame.pack(fill=tk.BOTH)

        # Frame for analysis
        self.frame_analyse = tk.Frame(self, borderwidth=4, relief=tk.GROOVE)
        self.label_analyse = tk.Label(self.frame_analyse, text="Reader", borderwidth=2, relief=tk.GROOVE)
        self.tree = ttk.Treeview(self.frame_analyse, columns=('type', 'value'))
        self.tree.heading('type', text='Object type')
        self.tree.heading('value', text='Value')

        # Frame for reader menu
        self.frame_original = tk.Frame(self.general_frame, borderwidth=4, relief=tk.GROOVE)
        self.frame_original.pack(side=tk.LEFT, fill=tk.BOTH, expand='yes')
        self.label_original = tk.Label(self.frame_original, text="Reader menu", borderwidth=2, relief=tk.GROOVE)
        self.label_original.pack()

        # Frame for converter menu
        self.frame_converted = tk.Frame(self.general_frame, borderwidth=4, relief=tk.GROOVE)
        self.frame_converted.pack(side=tk.LEFT, fill=tk.BOTH, expand='yes')
        self.label_converted = tk.Label(self.frame_converted, text="Converter menu", borderwidth=2, relief=tk.GROOVE)
        self.label_converted.pack()

        # Modify button
        self.modify_button = tk.Button(self.frame_converted, text="Modify model", width=10, command=self.modify, state=tk.DISABLED)
        self.modify_button.pack()

        # Frame for writer menu
        self.frame_exportation = tk.Frame(self.general_frame, borderwidth=4, relief=tk.GROOVE)
        self.frame_exportation.pack(side=tk.LEFT, fill=tk.BOTH, expand='yes')
        self.label_exportation = tk.Label(self.frame_exportation, text="Writer menu", borderwidth=2, relief=tk.GROOVE)
        self.label_exportation.pack()

        # Window quit button
        self.button_quit = tk.Button(self, text="Quit", command=self.quit)
        self.button_quit.pack(side=tk.BOTTOM)

        # Find path
        self.path_to_find = tk.StringVar()
        self.find_path_button = tk.Button(self.frame_original, text='Find', width=5, command=self.find_path)
        self.find_path_button.pack(side=tk.LEFT)

        # entry
        self.value = tk.StringVar()
        self.value.set("Enter path of original model")
        self.entree = tk.Entry(self.frame_original, textvariable=self.value, width=20)
        self.entree.pack(side=tk.LEFT)
        self.entree.focus_set()

        # Check path
        self.check_button = tk.Button(self.frame_original, text="Check path", width=10, command=self.check_path)
        self.check_button.pack(side=tk.LEFT)

        self.status = tk.Label(self.frame_original, text="File found : " + self.file_type)
        self.status.pack(side=tk.BOTTOM)

        self.analyse_button =\
            tk.Button(self.frame_original, text='Read file', command=self.analyse, state=tk.DISABLED)
        self.analyse_button.pack(side=tk.BOTTOM)

    def fun(self, name):
        def _fun():
            print(name)
        return _fun

    def find_path(self):
        self.filename = filedialog.askopenfilename(initialdir=self.initial_dir, title="Select file",
                                                   filetypes=(("bioMod files", "*.bioMod"),
                                                              ("S2mMod files", "*.s2mMod"), ("OpenSim files", "*.osim"),
                                                              ("all files", "*.*")))
        self.value.set(os.path.relpath(self.filename))

    def check(self, _window, is_error=False):
        def _check():
            _window.destroy()
            self.status.config(text="File found : " + self.file_type)
            if is_error:
                self.analyse_button.config(state=tk.DISABLED)
                self.menu2.entryconfig(0, state=tk.DISABLED)
            else:
                self.is_checked = True
                self.analyse_button.config(state=tk.NORMAL)
                self.menu2.entryconfig(0, state=tk.NORMAL)
        return _check

    def unknown_extension_check(self):
        res = self.var1.get()
        if res == 'Biorbd file (.bioMod)':
            self.file_type = 'Biorbd'
            self.is_checked = True
            self.status.config(text="File found : " + self.file_type)
            self.analyse_button.config(state=tk.NORMAL)
            self.top.destroy()
            return 0
        elif res == 'OpenSim file (.osim)':
            self.file_type = 'OpenSim'
            self.is_checked = True
            self.status.config(text="File found : " + self.file_type)
            self.analyse_button.config(state=tk.NORMAL)
            self.top.destroy()
            return 0
        else:
            error_window = tk.Toplevel()
            error_window.geometry("%dx%d%+d%+d" % (300, 160, 250, 125))
            error_window.title("Error")
            label_error = tk.LabelFrame(error_window, text="Message", padx=20, pady=20)
            label_error.pack(fill="both", expand="yes")
            tk.Label(label_error, text="You must choose a type of file").pack()
            tk.Button(label_error, text="Ok", command=self.check(error_window, True)).pack()

    def check_path(self):
        self.modify_button.config(state=tk.DISABLED)
        self.file_type = 'None'
        self.original_path = self.entree.get()
        if os.path.exists(self.original_path):
            if self.original_path.find('.osim') > -1:
                self.check_status = 'OpenSim file found'
                self.file_exist = True
                self.unknown_extension = False
                self.file_type = 'OpenSim'
            elif self.original_path.find('.bioMod') > -1:
                self.check_status = 'Biorbd file found'
                self.file_exist = True
                self.unknown_extension = False
                self.file_type = 'Biorbd'
            elif self.original_path.find('.biomod') > -1:
                self.check_status = 'Biorbd file found'
                self.file_exist = True
                self.unknown_extension = False
                self.file_type = 'Biorbd'
            else:
                self.check_status = 'file found but extension is not recognized.\n' \
                               'Please indicate the type of file : '
                self.file_exist = True
                self.unknown_extension = True
        else:
            self.check_status = 'file not found'
            self.file_exist = False
            self.unknown_extension = False

        # Message box
        self.top = tk.Toplevel()
        self.top.geometry("%dx%d%+d%+d" % (300, 160, 250, 125))
        self.top.title("File checker")
        label_top = tk.LabelFrame(self.top, text="Message", padx=20, pady=20)
        label_top.pack(fill="both", expand="yes")
        tk.Label(label_top, text=self.check_status).pack()

        if self.unknown_extension:
            var1 = tk.StringVar()
            options = ["Biorbd file (.bioMod)", "OpenSim file (.osim)"]
            list_file = tk.OptionMenu(label_top, var1, *options)
            var1.set('Choose a file type')  # default value
            list_file.pack()
            button_quit = tk.Button(label_top, text="Ok", command=self.unknown_extension_check)
            button_quit.pack(side=tk.BOTTOM)
        else:
            if self.file_exist:
                button_quit = tk.Button(label_top, text="Ok", command=self.check(self.top))
                button_quit.pack()
            else:
                button_quit = tk.Button(label_top, text="Ok", command=self.check(self.top, True))
                button_quit.pack()
        if self.file_exist and self.is_checked:
            self.status.config(text="File found : " + self.file_type)

    def actualise_file(self):
        if self.file_type == 'OpenSim':
            self.biorbd_path = self.original_path[-2:]+'-converted.bioMod'
            try:
                ConvertedFromOsim2Biorbd3(self.biorbd_path, self.original_path)
                self.model = BiorbdModel()
                self.model.read(self.biorbd_path)
            except:
                ConvertedFromOsim2Biorbd4(self.biorbd_path, self.original_path)
                self.model = BiorbdModel()
                self.model.read(self.biorbd_path)
        elif self.file_type == 'Biorbd':
            self.biorbd_path = self.original_path
            try:
                self.model = BiorbdModel()
                self.model.read(self.biorbd_path)
            except:
                assert "Biorbd model could not be read"

    def analyse(self):
        self.actualise_file()
        self.modify_button.config(state=tk.NORMAL)
        # Frame for analyse
        if self.is_analyzed:
            self.frame_analyse.destroy()
            self.frame_analyse = tk.Frame(self, borderwidth=4, relief=tk.GROOVE)
            self.label_analyse = tk.Label(self.frame_analyse, text="Reader", borderwidth=2, relief=tk.GROOVE)
            self.tree = ttk.Treeview(self.frame_analyse, columns=('type', 'value'))
            self.tree.heading('type', text='Object type')
            self.tree.heading('value', text='Value')
            self.is_analyzed = False
        if not self.is_analyzed:
            self.geometry("%dx%d%+d%+d" % (1000, 400, 350, 125))
            self.frame_analyse.pack(side=tk.BOTTOM, fill=tk.BOTH, expand='yes')
            self.label_analyse.pack()
            self.is_analyzed = True

        index_model = self.tree.insert('', 'end', 'Model', text='Model')
        index_segments = self.tree.insert(index_model, 0, 'Segments', text='Segments')
        index_muscle_groups = self.tree.insert(index_model, 1, 'MuscleGroups', text='MuscleGroups')

        for segment in self.model.get_segments():
            index_segment = self.tree.insert(index_segments, 'end', segment.get_name(),
                                             text='segment '+segment.get_name())
            index_segment_parameters = self.tree.insert(index_segment, 'end', segment.get_name()+'_parameters',
                                                        text='Parameters')
            self.tree.insert(index_segment_parameters, 'end', segment.get_name() + 'parent',
                             text='parent', values=('', segment.get_parent()))
            self.tree.insert(index_segment_parameters, 'end', segment.get_name() + '_mass',
                             text='mass', values=('', segment.get_mass()))
            self.tree.insert(index_segment_parameters, 'end', segment.get_name() + '_inertia',
                             text='inertia', values=('', segment.get_inertia()))
            self.tree.insert(index_segment_parameters, 'end', segment.get_name() + '_com',
                             text='center of mass', values=('', segment.get_com()))
            self.tree.insert(index_segment_parameters, 'end', segment.get_name() + '_rt_in_matrix',
                             text='rt_in_matrix', values=('', segment.get_rt_in_matrix()))
            self.tree.insert(index_segment_parameters, 'end', segment.get_name() + 'rot_trans_matrix',
                             text='rot_trans_matrix', values=('', segment.get_rot_trans_matrix()))
            self.tree.insert(index_segment_parameters, 'end', segment.get_name() + 'dof_translation',
                             text='dof_translation', values=('', segment.get_dof_translation()))
            self.tree.insert(index_segment_parameters, 'end', segment.get_name() + 'dof_rotation',
                             text='dof_rotation', values=('', segment.get_dof_rotation()))
            markers = segment.get_markers()
            if markers:
                index_markers = self.tree.insert(index_segment, 'end', 'markers'+segment.get_name(), text='markers')
                for marker in markers:
                    index_marker = self.tree.insert(index_markers, 'end', marker.get_name(),
                                                    text='marker '+marker.get_name())
                    self.tree.insert(index_marker, 'end', marker.get_name() + '_position',
                                     text='position in segment', values=('', marker.get_position()))
                    # TODO add technical and anatomical
        for muscle_group in self.model.get_muscle_groups():
            index_muscle_group = \
                self.tree.insert(index_muscle_groups, 'end', muscle_group.get_name(),
                                 text='muscle group '+muscle_group.get_name())
            muscles = muscle_group.get_muscles()
            if muscles:
                for muscle in muscles:
                    index_muscle = \
                        self.tree.insert(index_muscle_group, 'end', muscle.get_name(), text='muscle '+muscle.get_name())
                    index_muscle_parameters = self.tree.insert(index_muscle, 'end',
                                                               muscle.get_name() + '_parameters', text='Parameters')
                    self.tree.insert(index_muscle_parameters, 'end', muscle.get_name() + 'type',
                                     text='type', values=('', muscle.get_type()))
                    self.tree.insert(index_muscle_parameters, 'end', muscle.get_name() + 'state_type',
                                     text='state_type', values=('', muscle.get_state_type()))
                    self.tree.insert(index_muscle_parameters, 'end', muscle.get_name() + 'optimal_length',
                                     text='optimal_length', values=('', muscle.get_optimal_length()))
                    self.tree.insert(index_muscle_parameters, 'end', muscle.get_name() + 'maximal_force',
                                     text='maximal_force', values=('', muscle.get_maximal_force()))
                    self.tree.insert(index_muscle_parameters, 'end', muscle.get_name() + 'tendon_slack_length',
                                     text='tendon_slack_length', values=('', muscle.get_tendon_slack_length()))
                    self.tree.insert(index_muscle_parameters, 'end', muscle.get_name() + 'pennation_angle',
                                     text='pennation_angle', values=('', muscle.get_pennation_angle()))
                    self.tree.insert(index_muscle_parameters, 'end', muscle.get_name() + 'max_velocity',
                                     text='max_velocity', values=('', muscle.get_max_velocity()))
                    pathpoints = muscle.get_pathpoints()
                    if pathpoints:
                        index_pathpoints = self.tree.insert(index_muscle, 'end', 'pathpoints' + muscle.get_name(),
                                                            text='pathpoints')
                        for pathpoint in pathpoints:
                            index_pathpoint = self.tree.insert(index_pathpoints, 'end', pathpoint.get_name(),
                                                               text='pathpoint '+pathpoint.get_name())
                            self.tree.insert(index_pathpoint, 'end', pathpoint.get_name() + '_parent',
                                             text='parent', values=('', pathpoint.get_parent()))
                            self.tree.insert(index_pathpoint, 'end', pathpoint.get_name() + '_position',
                                             text='position in parent muscle', values=('', pathpoint.get_position()))
        self.tree.pack(side=tk.BOTTOM, fill=tk.BOTH, expand='yes')

    def state(self, boolean):
        if boolean:
            self.analyse_button.config(state=tk.NORMAL)

    def modify(self):
        # Modify window
        self.add = False
        self.delete = False
        self.modify = False
        self.modify_window = tk.Toplevel()
        self.modify_window.geometry("%dx%d%+d%+d" % (300, 400, 250, 125))
        self.modify_window.title("Modify model")
        # Label
        self.label_modify = tk.LabelFrame(self.modify_window, text="Choose modification", padx=20, pady=20)
        self.label_modify.pack(fill="both", expand="yes")
        # Action choice button
        self.var_action = tk.StringVar()
        self.options_action = ["Add item", "Delete item", "Modify item"]
        self.list_file_action = tk.OptionMenu(self.label_modify, self.var_action, *self.options_action, command=self.show_item_type)
        self.var_action.set('Choose an action')  # default value
        self.list_file_action.pack()
        # Item choice button
        self.var_item = tk.StringVar()
        self.options_item = ["Segment", "MuscleGroup"]
        self.list_file_item = tk.OptionMenu(self.label_modify, self.var_item, *self.options_item, command=self.show_item_names)
        self.var_item.set('Choose an item type')  # default value
        # Modify button
        self.button_quit_modify = tk.Button(self.modify_window, text="Modify", command=self.modify_file)
        self.button_quit_modify.pack()
        # Quit Button
        self.button_quit = tk.Button(self.modify_window, text="Quit", command=self.modify_window.destroy)
        self.button_quit.pack()

    def modify_file(self, value):
        # warn if action not chosen
        # re-load model in tree with analyse or read
        pass

    def show_item_type(self, value):
        self.list_file_item.pack()
        if self.var_action == "Add item":
            self.add = True
            self.delete = False
            self.modify = False
        elif self.var_action == "Delete item":
            self.add = False
            self.delete = True
            self.modify = False
        elif self.var_action == "Modify item":
            self.add = False
            self.delete = False
            self.modify = True


    def show_item_names(self, valu):
        self.var_names = tk.StringVar()
        self.options_names = []
        if self.var_item == "Segment":
            for segment in self.model.get_segments():
                self.options_names.append(segment.get_name())
        if self.var_item == "MuscleGroup":
            for musclegroup in self.model.get_muscle_groups():
                self.options_names.append(musclegroup.get_name())
        self.list_file_names = tk.OptionMenu(self.label_modify, self.var_names, *self.options_names,
                                            command=self.show_item_parameters)
        self.var_names.set('Choose a '+self.var_item)

    def show_item_parameters(self, val):
        self.list_file_names.pack()



if __name__ == "__main__":

    main_window = MainWindow()
    main_window.mainloop()


