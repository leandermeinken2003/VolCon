"""Create an app to draw the masks for a micrograph."""
# pylint: disable=attribute-defined-outside-init, too-many-instance-attributes, too-few-public-methods

import os
import tkinter as tk
from tkinter import (
    Canvas,
    Entry,
    Label,
)
from PIL import (
    Image,
    ImageDraw,
    ImageTk,
)
import numpy as np


COMPOSITION_ID = "Bi70Sn30"
IMAGE_ID = "000763"
TEST_SAMPLES_PATH = "testing/unprocessed_testing_data/"
IMAGE_PATH = TEST_SAMPLES_PATH + f"{COMPOSITION_ID}/images/{IMAGE_ID}.jpg"
MASKS_PATH = TEST_SAMPLES_PATH + f"{COMPOSITION_ID}/masks/{IMAGE_ID}/"

MIN_IMAGE_SIZE = 256
MAX_IMAGE_SIZE = 384
ORIGINAL_MASK_COLOR = (0, 0, 0, 0)
MASK_COLOR = (255, 0, 0, 30)


class MaskDrawingApp:
    """
    A small desktop Application to draw masks for a selected micrograph. The app has
    the following functionalities:

        - add background image
        - draw on background image
        - eraser for drawing
        - clear entire drawing
        - create a rectangle
        - do a basic analysis of the image to get general contours
        - define save file name
        - save file
    """
    def __init__(self, app: tk.Tk):
        self.app = app
        self.app.title("Paint App")
        self._create_mask_drawing_canvas(app)
        self._create_brush(app)
        self._create_eraser()
        self._create_rectangle_maker()
        self._create_pre_analyse_button(app)
        self._create_save_button(app)
        self._create_clear_button(app)
        self._create_blank_mask() # initialize mask to draw on

    def _create_mask_drawing_canvas(self, app: tk.Tk) -> None:
        microstructure_image = Image.open(IMAGE_PATH).convert("L").convert("RGBA")
        self.microstructure_image = self._resize_image(microstructure_image)
        self.image_height = self.microstructure_image.height
        self.image_width = self.microstructure_image.width
        self.projected_image = ImageTk.PhotoImage(self.microstructure_image) # microstructure + mask
        self.canvas = Canvas(app, width=self.image_width, height=self.image_height)
        self.canvas.pack()
        self.canvas.create_image(
            0, 0, anchor=tk.NW, image=self.projected_image,
        )

    def _resize_image(self, microstructure_image: Image.Image) -> Image.Image:
        image_height = microstructure_image.height
        image_width = microstructure_image.width
        if image_height < MIN_IMAGE_SIZE and image_height <= image_width:
            scaling_factor = MIN_IMAGE_SIZE / image_height
        elif image_width < MIN_IMAGE_SIZE and image_width < image_height:
            scaling_factor = MIN_IMAGE_SIZE / image_width
        elif image_height > MAX_IMAGE_SIZE and image_height >= image_width:
            scaling_factor = MAX_IMAGE_SIZE / image_height
        elif image_width > MAX_IMAGE_SIZE and image_width > image_height:
            scaling_factor = MAX_IMAGE_SIZE / image_width
        else:
            scaling_factor = 1
        return microstructure_image.resize(
            (int(scaling_factor * image_width), int(scaling_factor * image_height)),
        )

    def _create_brush(self, app: tk.Tk) -> None:
        self.canvas.bind("<B1-Motion>", self._paint)
        self.paint_size = 1
        self.paint_size_label = Label(app, text="Paint Size:")
        self.paint_size_label.pack()
        self.paint_size_entry = Entry(app)
        self.paint_size_entry.pack()
        self.paint_size_entry.insert(0, "1")
        self.paint_size_button = tk.Button(app, text='Set paint size', command=self._set_paint_size)
        self.paint_size_button.pack()

    def _create_eraser(self) -> None:
        self.canvas.bind("<B3-Motion>", self._erase)

    def _create_rectangle_maker(self) -> None:
        self.canvas.bind('<Button-2>', self._start_rectangle)
        self.canvas.bind('<ButtonRelease-2>', self._create_rectangle)
        self.rect_start_x = None
        self.rect_start_y = None

    def _create_pre_analyse_button(self, app: tk.Tk) -> None:
        self.clear_button = tk.Button(
            app, text='Pre-Analyse-Micrograph', command=self._pre_analyse_micrograph,
        )
        self.clear_button.pack()

    def _create_save_button(self, app: tk.Tk) -> None:
        self.mask_name_label = Label(app, text="Mask Name:")
        self.mask_name_label.pack()
        self.mask_name_entry = Entry(app)
        self.mask_name_entry.pack()
        self.mask_name_entry.insert(0, "mask name")
        self.save_button = tk.Button(app, text='Save and Extract', command=self._save_and_extract)
        self.save_button.pack()

    def _create_clear_button(self, app: tk.Tk) -> None:
        self.clear_button = tk.Button(app, text='Clear', command=self._clear_canvas)
        self.clear_button.pack()

    def _create_blank_mask(self) -> None:
        self.mask = Image.new(
            'RGBA', (self.image_width, self.image_height), color=ORIGINAL_MASK_COLOR,
        )
        self.draw = ImageDraw.Draw(self.mask)

    def _paint(self, event) -> None:
        x1, y1 = (event.x - self.paint_size), (event.y - self.paint_size)
        x2, y2 = (event.x + self.paint_size), (event.y + self.paint_size)
        self.draw.ellipse([x1, y1, x2, y2], fill=MASK_COLOR, width=1)
        self.projected_image = ImageTk.PhotoImage(
            Image.alpha_composite(self.microstructure_image, self.mask)
        )
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.projected_image)

    def _set_paint_size(self) -> None:
        self.paint_size = int(self.paint_size_entry.get())

    def _erase(self, event) -> None:
        x1, y1 = (event.x - self.paint_size), (event.y - self.paint_size)
        x2, y2 = (event.x + self.paint_size), (event.y + self.paint_size)
        self.draw.ellipse([x1, y1, x2, y2], fill=ORIGINAL_MASK_COLOR, width=1)
        self.projected_image = ImageTk.PhotoImage(
            Image.alpha_composite(self.microstructure_image, self.mask)
        )
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.projected_image)

    def _start_rectangle(self, event) -> None:
        self.rect_start_x = event.x
        self.rect_start_y = event.y

    def _create_rectangle(self, event):
        if self.rect_start_x and self.rect_start_y:
            self.draw.rectangle([
                self.rect_start_x,
                self.rect_start_y,
                event.x,
                event.y,
            ],
            fill=MASK_COLOR,
            )
            self.rect_start_x = None
            self.rect_start_y = None
            self.projected_image = ImageTk.PhotoImage(
                Image.alpha_composite(self.microstructure_image, self.mask)
            )
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.projected_image)

    def _pre_analyse_micrograph(self) -> None:
        pre_analysis_mask = self._get_pre_analysis_mask()
        pre_analysis_mask_array = []
        for mask_row in pre_analysis_mask:
            row = []
            for mask_val in mask_row:
                if mask_val:
                    row.append(list(MASK_COLOR))
                else:
                    row.append(list(ORIGINAL_MASK_COLOR))
            pre_analysis_mask_array.append(row)
        pre_analysis_mask_array = np.array(pre_analysis_mask_array).astype(np.uint8)
        pre_analysis_mask = Image.fromarray(pre_analysis_mask_array, mode="RGBA")
        self.mask.paste(pre_analysis_mask)
        self.projected_image = ImageTk.PhotoImage(
            Image.alpha_composite(self.microstructure_image, self.mask)
        )
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.projected_image)

    def _get_pre_analysis_mask(self) -> np.ndarray:
        info_box_mask = Image.open(MASKS_PATH + "info_box.png").convert("L")
        info_box_bool_mask = np.array(info_box_mask).astype(bool)
        gray_scale_micro_structure_image = self.microstructure_image.convert("L")
        gray_scale_micro_structure_array = np.array(gray_scale_micro_structure_image)
        average_value = gray_scale_micro_structure_array[info_box_bool_mask].mean()
        return np.where(
            gray_scale_micro_structure_array > average_value,
            True,
            False,
        )

    def _save_and_extract(self) -> None:
        mask_name = self.mask_name_entry.get()
        file_path = MASKS_PATH + f"{mask_name}.png"
        save_mask = np.array(self.mask.convert("L"))
        save_mask = np.where(save_mask > 0, 0, 254)
        image = Image.fromarray(save_mask.astype(np.uint8), "L")
        image.save(file_path)

    def _clear_canvas(self) -> None:
        self.projected_image = ImageTk.PhotoImage(self.microstructure_image)
        self.canvas.delete("all")
        self.canvas.create_image(
            0, 0, anchor=tk.NW, image=self.projected_image,
        )
        self._create_blank_mask()


if __name__ == "__main__":
    if not os.path.exists(MASKS_PATH):
        os.makedirs(MASKS_PATH)
    root = tk.Tk()
    _ = MaskDrawingApp(root)
    root.mainloop()
