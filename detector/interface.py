import tkinter as tk
from tkinter import filedialog
from PIL import ImageGrab
import cv2 as cv
from cv2.typing import MatLike

from detector.app import App
from detector.video_capture import NO_VIDEO

AFTER_DELAY = 1

class GUI:
    def __init__(self, communication_interface: App) -> None:
        self._communication_interface = communication_interface

        # Initializing screen resolution (for window size)
        screen = ImageGrab.grab()
        screen_width, screen_height = screen.size
        self._communication_interface.set_max_display_dimention(screen_width, screen_height)

        self._selected_video_source_id = None
        self._initialize_control_panel()
        

    def _initialize_control_panel(self) -> None:
        self._root = tk.Tk()
        self._root.title('Control panel')
        self._root.resizable(False, False)

        screen = ImageGrab.grab()
        screen_width, screen_height = screen.size
        scaling_factor = 0.3

        window_width = int(screen_width * scaling_factor)
        window_height = int(screen_height * scaling_factor)
        self._root.geometry(f'{window_width}x{window_height}')
        self._root.geometry(f'+{0}+{0}')

        self._menubar = tk.Menu(master=self._root)
        self._root.config(menu=self._menubar)
        self._initialize_video_source_menu()
        self._initialize_detector_parameters_menu()


    def _initialize_video_source_menu(self) -> None:
        self._source_menu = tk.Menu(master=self._menubar, tearoff=0)
        self._selected_video_source_id = tk.IntVar()

        sources = self._communication_interface.get_available_sources()
        for source_name in sources:
            self._source_menu.add_radiobutton(
                label=source_name,
                variable=self._selected_video_source_id,
                value=sources[source_name],
                command=lambda source_index=sources[source_name]: self._on_video_source_select(source_index)
            )

        self._selected_video_source_id.set(NO_VIDEO)
        self._menubar.add_cascade(menu=self._source_menu, label='Video source')


    def _on_video_source_select(self, source_id: int) -> None:
        is_source_on = self._communication_interface.set_video_source(source_id)
        self._selected_video_source_id.set(source_id)

        if is_source_on:
            self._start_displaying()
        else:
            self._stop_displaying()


    def _start_displaying(self) -> None:
        self._is_displaying = True
        cv.namedWindow('Display', cv.WINDOW_NORMAL)
        cv.moveWindow('Display', 0, 0)
        self._update_frame()


    def _stop_displaying(self) -> None:
        self._is_displaying = False
        cv.destroyAllWindows()
    

    def _initialize_detector_parameters_menu(self) -> None:
        # This part is now moved to the main window
        self._detector_frame = tk.Frame(self._root)
        self._detector_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        # Add confidence threshold slider
        confidence_threshold_label = tk.Label(self._detector_frame, text='Confidence threshold:')
        confidence_threshold_label.pack(pady=1)

        confidence_threshold_slider = tk.Scale(
            self._detector_frame,
            from_=0.00,
            to=1.00,
            resolution=0.01,
            orient='horizontal',
            command=self._update_confidence_threshold
        )
        confidence_threshold_slider.set(self._communication_interface.get_confidence_threshold())
        confidence_threshold_slider.pack(pady=10)

        # Add object class selection
        classes_label = tk.Label(self._detector_frame, text='Classes of objects:')
        classes_label.pack(pady=1)

        checkbox_canvas = tk.Canvas(self._detector_frame)
        checkbox_canvas.pack(side='left', fill='both', expand=True)

        scrollbar = tk.Scrollbar(self._detector_frame, orient='vertical', command=checkbox_canvas.yview)
        scrollbar.pack(side='right', fill='y')

        checkbox_canvas.configure(yscrollcommand=scrollbar.set)

        checkbox_frame = tk.Frame(checkbox_canvas)
        checkbox_canvas.create_window((0, 0), window=checkbox_frame, anchor='nw')

        checkbox_vars = []
        row = 0
        col = 0
        max_items_per_row = 4
        all_classes = self._communication_interface.get_available_classes()
        detected_classes = self._communication_interface.get_detected_classes()

        for index, class_name in enumerate(all_classes):
            var = tk.BooleanVar(value=(index in detected_classes))
            checkbox = tk.Checkbutton(
                checkbox_frame,
                text=class_name,
                variable=var,
                command=lambda idx=index, v=var: self._update_classes(idx, v)
            )

            # Place the checkbox in the grid
            checkbox.grid(row=row, column=col, sticky='w', padx=5, pady=2)

            checkbox_vars.append(var)

            col += 1
            if col == max_items_per_row:
                col = 0
                row += 1

        checkbox_frame.update_idletasks()
        checkbox_canvas.config(scrollregion=checkbox_canvas.bbox('all'))

        # Bind the scroll events
        checkbox_canvas.bind_all('<MouseWheel>', self._on_mouse_scroll)  # For Windows/macOS
        checkbox_canvas.bind_all('<Button-4>', self._on_mouse_scroll)    # For Linux scroll up
        checkbox_canvas.bind_all('<Button-5>', self._on_mouse_scroll)    # For Linux scroll down

        # Save checkbox_canvas reference
        self._checkbox_canvas = checkbox_canvas

    
    def _on_mouse_scroll(self, event) -> None:
        if event.num == 4 or event.delta > 0:
            self._checkbox_canvas.yview_scroll(-1, 'units')
        elif event.num == 5 or event.delta < 0:
            self._checkbox_canvas.yview_scroll(1, 'units')
        

    def _update_confidence_threshold(self, value) -> None:
        self._communication_interface.set_confidence_threshold(float(value))


    def _update_classes(self, class_index, var) -> None:
        if var.get():
            self._communication_interface.add_detected_class(class_index)
        else:
            self._communication_interface.remove_detected_class(class_index)


    def select_video_file(self) -> str:
        filetypes = [
            ('Video files', '.mp4 *.avi *.mkv *.mov *.wmv')
        ]
        file_path = filedialog.askopenfilename(title='Select a recorded video', filetypes=filetypes)

        return file_path


    def show(self) -> None:
        self._root.mainloop()


    def _show_frame(self, frame: MatLike) -> None:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        cv.imshow('Display', frame)


    def _update_frame(self) -> None:
        if not self._is_displaying:
            self._stop_displaying()
            return

        is_capture_on, frame = self._communication_interface.get_processed_frame()

        if is_capture_on:
            if frame is not None:
                self._show_frame(frame)
            self._root.after(AFTER_DELAY, self._update_frame)
        else:
            self._communication_interface.set_video_source(NO_VIDEO)
            self._selected_video_source_id.set(NO_VIDEO)
            self._stop_displaying()
