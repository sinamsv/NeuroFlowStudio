import sys
import math
import json
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim

# =========================================================
# CONFIG
# =========================================================
pygame.init()
torch.manual_seed(42)
np.random.seed(42)

# Base design resolution (all positions/sizes defined for this)
BASE_W, BASE_H = 1280, 820
INITIAL_WIDTH, INITIAL_HEIGHT = 1280, 820
FPS = 60

# Drawing area (in base coordinates)
DRAW_AREA_X = 40
DRAW_AREA_Y = 40
DRAW_AREA_W = 1200
DRAW_AREA_H = 560

# Toolbar (moved down a bit)
TOOLBAR_Y = 660
TOOLBAR_H = 140

MIN_NEURONS = 1
MAX_NEURONS = 128

ACTIVATIONS = ['relu', 'sigmoid', 'tanh', 'linear']
ACT_DISPLAY = {'relu': 'ReLU', 'sigmoid': 'Sigmoid', 'tanh': 'Tanh', 'linear': 'Linear'}

# ---------- Neon Theme ----------
NEON_THEME = {
    "bg": (5, 5, 15),
    "card": (12, 12, 25, 180),
    "card_solid": (12, 12, 25),
    "card_shadow": (0, 0, 0),
    "text": (220, 220, 255),
    "subtext": (160, 160, 220),
    "outline": (80, 80, 120),
    "neon_green": (57, 255, 20),
    "neon_yellow": (255, 255, 0),
    "neon_pink": (255, 20, 147),
    "neon_purple": (180, 0, 255),
    "neon_blue": (0, 200, 255),
    "neon_red": (255, 50, 50),
    "green": (88, 200, 120),
    "green_soft": (60, 150, 90),
    "yellow": (255, 214, 84),
    "yellow_soft": (180, 160, 50),
    "white": (240, 240, 255),
    "black": (5, 5, 15),
    "glass_fill": (255, 255, 255, 30),
    "glass_border": (100, 200, 255, 150),
    "layer_even": (32, 32, 50),
    "layer_odd": (40, 28, 55),
    "popup_bg": (20, 20, 35),
    "popup_overlay": (0, 0, 0, 200),
}

# ---------- Font helpers ----------
_font_cache = {}
def get_font(size, bold=False):
    key = (size, bold)
    if key not in _font_cache:
        _font_cache[key] = pygame.font.SysFont("arial", size, bold=bold)
    return _font_cache[key]

FONT_TITLE = get_font(38, True)
FONT_BIG = get_font(28, True)
FONT_MED = get_font(22, True)
FONT = get_font(18, False)
FONT_SMALL = get_font(15, False)

# =========================================================
# UI HELPERS
# =========================================================
def draw_text(surface, text, font, color, x, y, center=False):
    img = font.render(text, True, color)
    rect = img.get_rect()
    if center:
        rect.center = (x, y)
    else:
        rect.topleft = (x, y)
    surface.blit(img, rect)
    return rect

def draw_rounded_rect(surface, rect, color, radius=16, border=0, border_color=(0,0,0)):
    if len(color) == 4:  # with alpha
        shape_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, color, shape_surf.get_rect(), border_radius=radius)
        surface.blit(shape_surf, rect.topleft)
    else:
        pygame.draw.rect(surface, color, rect, border_radius=radius)
    if border > 0:
        pygame.draw.rect(surface, border_color, rect, width=border, border_radius=radius)

def draw_card(surface, rect, theme, color=None, shadow=True, radius=20, glass=False):
    if color is None:
        color = theme["card_solid"] if not glass else theme["glass_fill"]
    if shadow:
        shadow_rect = pygame.Rect(rect.x + 4, rect.y + 6, rect.w, rect.h)
        pygame.draw.rect(surface, theme["card_shadow"], shadow_rect, border_radius=radius)
    if glass:
        draw_rounded_rect(surface, rect, theme["glass_fill"], radius=radius)
        pygame.draw.rect(surface, theme["glass_border"], rect, width=2, border_radius=radius)
    else:
        draw_rounded_rect(surface, rect, color, radius=radius, border=2, border_color=theme["outline"])

class Button:
    def __init__(self, x, y, w, h, text, color_key, hover_color_key=None, glass=True):
        self.rect = pygame.Rect(x, y, w, h)  # base coordinates
        self.text = text
        self.default_text = text
        self.color_key = color_key
        self.hover_color_key = hover_color_key if hover_color_key else color_key
        self.radius = 16
        self.glass = glass

    def set_text(self, new_text):
        self.text = new_text

    def draw(self, surface, theme, mouse_pos, scale_x=1.0, scale_y=1.0):
        r = pygame.Rect(
            int(self.rect.x * scale_x),
            int(self.rect.y * scale_y),
            int(self.rect.w * scale_x),
            int(self.rect.h * scale_y)
        )
        hovered = r.collidepoint(mouse_pos)
        col = theme[self.hover_color_key] if hovered else theme[self.color_key]

        if self.glass:
            shadow_rect = r.copy()
            shadow_rect.x += int(4 * scale_x)
            shadow_rect.y += int(6 * scale_y)
            pygame.draw.rect(surface, theme["card_shadow"], shadow_rect, border_radius=self.radius)
            draw_rounded_rect(surface, r, theme["glass_fill"], radius=self.radius)
            border_color = theme["glass_border"]
            if hovered:
                border_color = theme[self.hover_color_key]
            pygame.draw.rect(surface, border_color, r, width=max(1, int(2*scale_x)), border_radius=self.radius)
            text_color = theme["white"]
        else:
            shadow_rect = r.copy()
            shadow_rect.x += int(4 * scale_x)
            shadow_rect.y += int(6 * scale_y)
            pygame.draw.rect(surface, theme["card_shadow"], shadow_rect, border_radius=self.radius)
            pygame.draw.rect(surface, col, r, border_radius=self.radius)
            pygame.draw.rect(surface, theme["white"], r, width=max(1, int(2*scale_x)), border_radius=self.radius)
            text_color = theme["white"]

        draw_text(surface, self.text, FONT, text_color, r.centerx, r.centery, center=True)

    def clicked(self, event, scale_x=1.0, scale_y=1.0):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx = event.pos[0] / scale_x
            my = event.pos[1] / scale_y
            return self.rect.collidepoint(mx, my)
        return False

class Popup:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.visible = False
        self.rect = pygame.Rect(0, 0, width, height)

    def show(self):
        self.visible = True

    def hide(self):
        self.visible = False

    def center(self, screen_width, screen_height):
        self.rect.center = (screen_width // 2, screen_height // 2)

    def draw(self, surface, theme, scale_x=1.0, scale_y=1.0):
        if not self.visible:
            return
        overlay = pygame.Surface((surface.get_width(), surface.get_height()), pygame.SRCALPHA)
        overlay.fill(theme["popup_overlay"])
        surface.blit(overlay, (0, 0))
        r = pygame.Rect(
            int(self.rect.x * scale_x),
            int(self.rect.y * scale_y),
            int(self.rect.w * scale_x),
            int(self.rect.h * scale_y)
        )
        draw_card(surface, r, theme, color=theme["popup_bg"], shadow=True, radius=24)
        pygame.draw.rect(surface, theme["outline"], r, width=2, border_radius=24)

    def handle_event(self, event, scale_x=1.0, scale_y=1.0):
        if not self.visible:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx = event.pos[0] / scale_x
            my = event.pos[1] / scale_y
            if not self.rect.collidepoint(mx, my):
                return True
        return False

# =========================================================
# NEURAL NET
# =========================================================
class MLP(nn.Module):
    def __init__(self, layer_configs):
        super().__init__()
        layers = []
        in_dim = 2
        for cfg in layer_configs:
            layers.append(nn.Linear(in_dim, cfg['neurons']))
            act = cfg['activation']
            if act == 'relu':
                layers.append(nn.ReLU())
            elif act == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif act == 'tanh':
                layers.append(nn.Tanh())
            elif act == 'linear':
                layers.append(nn.Identity())
            in_dim = cfg['neurons']
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# =========================================================
# APP
# =========================================================
class App:
    def __init__(self, theme, screen_width, screen_height):
        self.theme = theme
        self.state = "MAIN_MENU"
        self.clock = pygame.time.Clock()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.scale_x = screen_width / BASE_W
        self.scale_y = screen_height / BASE_H

        # Model config
        self.layer_configs = [
            {'neurons': 8, 'activation': 'relu'},
            {'neurons': 8, 'activation': 'relu'}
        ]
        self.learning_rate = 0.01
        self.model = None
        self.optimizer = None
        self.loss_value = None
        self.acc_value = None
        self.boundary_surface = None
        self.training_active = False
        self.points_dirty = True
        self.X_train = None
        self.Y_train = None
        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.selected_color = 0   # 0 green, 1 yellow
        self.points = []
        self.status_text = ""

        # Blinking title animation
        self.title_alpha = 255
        self.title_fade_dir = -1

        # UI elements (base coordinates)
        self._init_main_menu_ui()
        self._init_app_settings_ui()
        self._init_game_ui()
        self._init_popup()

    def update_scaling(self, new_width, new_height):
        self.screen_width = new_width
        self.screen_height = new_height
        self.scale_x = new_width / BASE_W
        self.scale_y = new_height / BASE_H

    # ==================== MAIN MENU ====================
    def _init_main_menu_ui(self):
        self.btn_playground = Button(380, 300, 520, 80, "Playground", "neon_purple", "neon_pink", glass=True)
        self.btn_settings = Button(380, 420, 520, 80, "Settings", "neon_blue", "neon_purple", glass=False)
        self.btn_quit = Button(560, 540, 160, 56, "Quit", "neon_red", "neon_pink", glass=True)

    def draw_main_menu(self, surface):
        surface.fill(self.theme["bg"])
        # blinking title
        self.title_alpha += self.title_fade_dir * 8
        if self.title_alpha <= 100:
            self.title_fade_dir = 1
        elif self.title_alpha >= 255:
            self.title_fade_dir = -1
        title_color = self.theme["neon_pink"]
        title_surf = FONT_TITLE.render("NeuroFlow Studio", True, title_color)
        title_surf.set_alpha(self.title_alpha)
        title_rect = title_surf.get_rect(center=(int(BASE_W/2 * self.scale_x), int(120 * self.scale_y)))
        surface.blit(title_surf, title_rect)

        mouse = pygame.mouse.get_pos()
        self.btn_playground.draw(surface, self.theme, mouse, self.scale_x, self.scale_y)
        self.btn_settings.draw(surface, self.theme, mouse, self.scale_x, self.scale_y)
        self.btn_quit.draw(surface, self.theme, mouse, self.scale_x, self.scale_y)

    def handle_main_menu_event(self, event):
        if self.btn_playground.clicked(event, self.scale_x, self.scale_y):
            self.state = "GAME"
            self.create_model()
            self.status_text = "Draw points or select dataset."
        elif self.btn_settings.clicked(event, self.scale_x, self.scale_y):
            self.state = "APP_SETTINGS"
        elif self.btn_quit.clicked(event, self.scale_x, self.scale_y):
            pygame.quit()
            sys.exit()

    # ==================== APP SETTINGS ====================
    def _init_app_settings_ui(self):
        self.btn_back_settings = Button(40, 40, 120, 48, "Back", "neon_red", "neon_pink", glass=True)
        self.btn_resolution = Button(290, 350, 700, 70, "Resolution: 1280x820", "neon_purple", "neon_pink", glass=False)
        self.resolution_options = [
            ("3840x2160 (4K)", 3840, 2160),
            ("2560x1440 (2K)", 2560, 1440),
            ("1920x1080 (Full HD)", 1920, 1080),
            ("1280x820 (Default)", 1280, 820),
            ("720x480 (SD)", 720, 480),
            ("480x320 (Mobile)", 480, 320),
        ]
        self.show_resolution_menu = False
        self.res_buttons = []

    def draw_app_settings(self, surface):
        surface.fill(self.theme["bg"])
        draw_text(surface, "Application Settings", FONT_TITLE, self.theme["text"],
                  int(BASE_W/2 * self.scale_x), int(70 * self.scale_y), center=True)
        mouse = pygame.mouse.get_pos()
        self.btn_back_settings.draw(surface, self.theme, mouse, self.scale_x, self.scale_y)
        self.btn_resolution.draw(surface, self.theme, mouse, self.scale_x, self.scale_y)

        if self.show_resolution_menu:
            popup_x = int(290 * self.scale_x)
            popup_y = int(430 * self.scale_y)
            popup_w = int(700 * self.scale_x)
            popup_h = len(self.res_buttons) * int(50 * self.scale_y) + 20
            popup_rect = pygame.Rect(popup_x, popup_y, popup_w, popup_h)
            draw_card(surface, popup_rect, self.theme, color=self.theme["popup_bg"], radius=16, glass=False)
            for btn in self.res_buttons:
                btn.draw(surface, self.theme, mouse, self.scale_x, self.scale_y)

    def handle_app_settings_event(self, event):
        if self.btn_back_settings.clicked(event, self.scale_x, self.scale_y):
            self.state = "MAIN_MENU"
            self.show_resolution_menu = False
            return None
        if self.btn_resolution.clicked(event, self.scale_x, self.scale_y):
            self.show_resolution_menu = not self.show_resolution_menu
            if self.show_resolution_menu:
                self._build_resolution_buttons()
            return None
        if self.show_resolution_menu:
            for i, btn in enumerate(self.res_buttons):
                if btn.clicked(event, self.scale_x, self.scale_y):
                    _, w, h = self.resolution_options[i]
                    self.change_resolution(w, h)
                    self.show_resolution_menu = False
                    return "RESIZE"
        return None

    def _build_resolution_buttons(self):
        self.res_buttons.clear()
        base_x, base_y = 290, 440
        for i, (label, w, h) in enumerate(self.resolution_options):
            btn = Button(base_x + 20, base_y + i*55, 660, 44, label, "neon_blue", "neon_purple", glass=True)
            self.res_buttons.append(btn)
        self.btn_resolution.set_text(f"Resolution: {self.screen_width}x{self.screen_height}")

    def change_resolution(self, new_width, new_height):
        self.screen_width = new_width
        self.screen_height = new_height
        self.scale_x = new_width / BASE_W
        self.scale_y = new_height / BASE_H
        self.popup.center(new_width, new_height)
        self.boundary_surface = None
        self.points_dirty = True

    # ==================== GAME ====================
    def _init_game_ui(self):
        self.btn_back_game = Button(1120, 20, 120, 44, "Back", "neon_red", "neon_pink", glass=True)
        self.btn_model_settings = Button(30, 760, 155, 44, "Model Settings", "neon_purple", "neon_pink", glass=False)

        # Training controls (glass)
        self.btn_toggle_train = Button(740, 710, 190, 56, "Start Training", "neon_purple", "neon_pink", glass=True)
        self.btn_clear_points = Button(945, 710, 140, 56, "Clear Points", "neon_pink", "neon_red", glass=True)
        self.btn_clear_boundary = Button(1100, 710, 140, 56, "Clear Result", "neon_blue", "neon_purple", glass=True)

        # Dataset controls
        dataset_y = 720
        self.btn_ds_easy = Button(70, dataset_y, 110, 44, "Easy", "neon_green", "neon_yellow", glass=True)
        self.btn_ds_medium = Button(190, dataset_y, 110, 44, "Medium", "neon_yellow", "neon_pink", glass=True)
        self.btn_ds_hard = Button(310, dataset_y, 110, 44, "Hard", "neon_pink", "neon_red", glass=True)
        self.btn_ds_vhard = Button(430, dataset_y, 110, 44, "V.Hard", "neon_red", "neon_purple", glass=True)
        # Save / Load data
        self.btn_save_data = Button(560, dataset_y, 120, 44, "Save Data", "neon_blue", "neon_purple", glass=True)

        # Color palettes
        self.green_rect = pygame.Rect(70, 660, 120, 42)
        self.yellow_rect = pygame.Rect(210, 660, 120, 42)

    # ------- Model settings popup -------
    def _init_popup(self):
        self.popup = Popup(800, 600)
        self.popup_elements = []
        self.btn_close_popup = Button(0, 0, 100, 44, "Close", "neon_red", "neon_pink", glass=True)
        self.btn_add_layer_popup = Button(0, 0, 140, 44, "+ Layer", "neon_purple", "neon_pink", glass=False)
        self.btn_remove_layer_popup = Button(0, 0, 140, 44, "- Layer", "neon_pink", "neon_red", glass=False)
        self.btn_lr_minus = Button(0, 0, 44, 36, "-", "neon_pink", "neon_red", glass=False)
        self.btn_lr_plus = Button(0, 0, 44, 36, "+", "neon_purple", "neon_pink", glass=False)
        # Model save button
        self.btn_save_model = Button(0, 0, 140, 44, "Save Model", "neon_blue", "neon_purple", glass=False)

    def open_model_settings_popup(self):
        self.popup.show()
        self._build_popup_elements()

    def _build_popup_elements(self):
        self.popup_elements.clear()
        x0 = self.popup.rect.x + 30
        y = self.popup.rect.y + 70
        for i in range(len(self.layer_configs)):
            row_y = y + i * 55
            btn_minus = Button(x0 + 300, row_y, 44, 36, "-", "neon_pink", "neon_red", glass=False)
            btn_plus = Button(x0 + 350, row_y, 44, 36, "+", "neon_purple", "neon_pink", glass=False)
            btn_act_left = Button(x0 + 450, row_y, 36, 36, "<", "neon_purple", "neon_pink", glass=False)
            btn_act_right = Button(x0 + 500, row_y, 36, 36, ">", "neon_purple", "neon_pink", glass=False)
            self.popup_elements.append((i, btn_minus, btn_plus, btn_act_left, btn_act_right))

        btn_y = y + len(self.layer_configs) * 55 + 20
        self.btn_add_layer_popup.rect.topleft = (x0, btn_y)
        self.btn_remove_layer_popup.rect.topleft = (x0 + 160, btn_y)
        lr_y = y + len(self.layer_configs) * 55 + 90
        self.btn_lr_minus.rect.topleft = (x0 + 300, lr_y)
        self.btn_lr_plus.rect.topleft = (x0 + 350, lr_y)
        self.btn_save_model.rect.topleft = (x0 + 500, lr_y)
        self.btn_close_popup.rect.topleft = (self.popup.rect.right - 140, self.popup.rect.bottom - 70)

    def draw_model_settings_popup(self, surface):
        if not self.popup.visible:
            return
        self.popup.draw(surface, self.theme, self.scale_x, self.scale_y)
        draw_text(surface, "Model Settings", FONT_BIG, self.theme["text"],
                  int(self.popup.rect.centerx * self.scale_x),
                  int((self.popup.rect.y + 25) * self.scale_y), center=True)
        x0 = self.popup.rect.x + 30
        y = self.popup.rect.y + 70
        mouse = pygame.mouse.get_pos()

        for i, cfg in enumerate(self.layer_configs):
            row_y = y + i * 55
            draw_text(surface, f"Layer {i+1}: {cfg['neurons']} neurons", FONT,
                      self.theme["text"], int(x0 * self.scale_x), int(row_y * self.scale_y + 10))
            act_text = ACT_DISPLAY[cfg['activation']]
            draw_text(surface, f"Act: {act_text}", FONT,
                      self.theme["subtext"], int((x0 + 400) * self.scale_x), int(row_y * self.scale_y + 10))
            _, btn_minus, btn_plus, btn_act_left, btn_act_right = self.popup_elements[i]
            btn_minus.draw(surface, self.theme, mouse, self.scale_x, self.scale_y)
            btn_plus.draw(surface, self.theme, mouse, self.scale_x, self.scale_y)
            btn_act_left.draw(surface, self.theme, mouse, self.scale_x, self.scale_y)
            btn_act_right.draw(surface, self.theme, mouse, self.scale_x, self.scale_y)

        self.btn_add_layer_popup.draw(surface, self.theme, mouse, self.scale_x, self.scale_y)
        self.btn_remove_layer_popup.draw(surface, self.theme, mouse, self.scale_x, self.scale_y)

        lr_y = y + len(self.layer_configs) * 55 + 90
        draw_text(surface, f"Learning Rate: {self.learning_rate:.4f}", FONT_MED,
                  self.theme["text"], int(x0 * self.scale_x), int(lr_y * self.scale_y + 5))
        self.btn_lr_minus.draw(surface, self.theme, mouse, self.scale_x, self.scale_y)
        self.btn_lr_plus.draw(surface, self.theme, mouse, self.scale_x, self.scale_y)
        self.btn_save_model.draw(surface, self.theme, mouse, self.scale_x, self.scale_y)
        self.btn_close_popup.draw(surface, self.theme, mouse, self.scale_x, self.scale_y)

        # Network topology diagram
        diagram_rect = pygame.Rect(
            int(self.popup.rect.x + 30) * self.scale_x,
            int(self.popup.rect.y + 220) * self.scale_y,
            int(self.popup.rect.w - 60) * self.scale_x,
            int(self.popup.rect.h - 300) * self.scale_y
        )
        draw_rounded_rect(surface, diagram_rect, self.theme["card_solid"], radius=12, border=2, border_color=self.theme["outline"])
        self.draw_network_topology(surface, diagram_rect)

    def draw_network_topology(self, surface, rect):
        layers = [2] + [cfg['neurons'] for cfg in self.layer_configs] + [1]
        n_layers = len(layers)
        if n_layers == 0:
            return
        spacing_x = rect.width / (n_layers - 1) if n_layers > 1 else rect.width
        max_neurons = max(layers)
        neuron_radius = min(18, rect.height / (max_neurons * 2.5))
        start_x = rect.left + 30
        end_x = rect.right - 30

        # Draw edges
        for l in range(n_layers - 1):
            x1 = start_x + l * spacing_x
            x2 = start_x + (l + 1) * spacing_x
            neurons_curr = layers[l]
            neurons_next = layers[l + 1]
            step_y1 = rect.height / max(neurons_curr, 1)
            step_y2 = rect.height / max(neurons_next, 1)
            for i in range(neurons_curr):
                y1 = rect.top + step_y1 * i + step_y1 / 2
                for j in range(neurons_next):
                    y2 = rect.top + step_y2 * j + step_y2 / 2
                    color = self.theme["outline"]
                    if self.model and l < len(self.layer_configs):
                        linear_idx = l * 2
                        if linear_idx < len(self.model.net):
                            w = self.model.net[linear_idx].weight.data[j, i].item()
                            if w > 0:
                                color = self.theme["neon_green"]
                            else:
                                color = self.theme["neon_red"]
                            alpha = min(255, max(80, int(abs(w) * 100)))
                            color = (*color[:3], alpha)
                    pygame.draw.line(surface, color, (x1, y1), (x2, y2), max(1, int(neuron_radius/4)))

        # Draw neurons
        for l in range(n_layers):
            x = start_x + l * spacing_x
            neurons = layers[l]
            step_y = rect.height / max(neurons, 1)
            for i in range(neurons):
                y = rect.top + step_y * i + step_y / 2
                pygame.draw.circle(surface, self.theme["neon_blue"], (int(x), int(y)), int(neuron_radius))
                pygame.draw.circle(surface, self.theme["white"], (int(x), int(y)), int(neuron_radius), 1)

    def handle_popup_event(self, event):
        if not self.popup.visible:
            return False
        if self.popup.handle_event(event, self.scale_x, self.scale_y):
            return True

        if self.btn_close_popup.clicked(event, self.scale_x, self.scale_y):
            self.popup.hide()
            return True

        # Layer-specific buttons
        for i, btn_minus, btn_plus, btn_act_left, btn_act_right in self.popup_elements:
            if btn_minus.clicked(event, self.scale_x, self.scale_y):
                self.layer_configs[i]['neurons'] = max(MIN_NEURONS, self.layer_configs[i]['neurons'] - 1)
                self._rebuild_model()
                return True
            if btn_plus.clicked(event, self.scale_x, self.scale_y):
                self.layer_configs[i]['neurons'] = min(MAX_NEURONS, self.layer_configs[i]['neurons'] + 1)
                self._rebuild_model()
                return True
            if btn_act_left.clicked(event, self.scale_x, self.scale_y):
                idx = ACTIVATIONS.index(self.layer_configs[i]['activation'])
                self.layer_configs[i]['activation'] = ACTIVATIONS[(idx - 1) % len(ACTIVATIONS)]
                self._rebuild_model()
                return True
            if btn_act_right.clicked(event, self.scale_x, self.scale_y):
                idx = ACTIVATIONS.index(self.layer_configs[i]['activation'])
                self.layer_configs[i]['activation'] = ACTIVATIONS[(idx + 1) % len(ACTIVATIONS)]
                self._rebuild_model()
                return True

        if self.btn_add_layer_popup.clicked(event, self.scale_x, self.scale_y):
            self.layer_configs.append({'neurons': 8, 'activation': 'relu'})
            self._rebuild_model()
            self._build_popup_elements()
            return True
        if self.btn_remove_layer_popup.clicked(event, self.scale_x, self.scale_y) and len(self.layer_configs) > 0:
            self.layer_configs.pop()
            self._rebuild_model()
            self._build_popup_elements()
            return True

        if self.btn_lr_minus.clicked(event, self.scale_x, self.scale_y):
            self.learning_rate = max(0.0001, self.learning_rate * 0.5)
            self._update_optimizer_lr()
            return True
        if self.btn_lr_plus.clicked(event, self.scale_x, self.scale_y):
            self.learning_rate = min(10.0, self.learning_rate * 2.0)
            self._update_optimizer_lr()
            return True

        if self.btn_save_model.clicked(event, self.scale_x, self.scale_y):
            self.save_model_config()
            return True

        return True

    # ------- Model management -------
    def create_model(self):
        self.model = MLP(self.layer_configs)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_value = None
        self.acc_value = None
        self.boundary_surface = None
        self.status_text = "Model ready."
        self.points_dirty = True

    def _rebuild_model(self):
        was_training = self.training_active
        if was_training:
            self.stop_training()
        self.model = MLP(self.layer_configs)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.points_dirty = True
        if was_training:
            self.start_training()

    def _update_optimizer_lr(self):
        if self.optimizer:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate

    def start_training(self):
        if self.model is None:
            self.create_model()
        if len(self.points) < 2 or len(set(p[2] for p in self.points)) < 2:
            self.status_text = "Need both green and yellow points."
            return
        self.training_active = True
        self.status_text = "Training live... (press Pause)"

    def stop_training(self):
        self.training_active = False
        self.status_text = "Training paused."

    def prepare_data(self):
        xs, ys = [], []
        for x_px, y_px, label in self.points:
            nx = (x_px - DRAW_AREA_X) / DRAW_AREA_W
            ny = (y_px - DRAW_AREA_Y) / DRAW_AREA_H
            xs.append([nx, ny])
            ys.append([label])
        self.X_train = torch.tensor(xs, dtype=torch.float32)
        self.Y_train = torch.tensor(ys, dtype=torch.float32).view(-1, 1)
        self.points_dirty = False

    def perform_training_step(self, epochs_per_frame=10):
        if not self.training_active or self.model is None:
            return
        if len(self.points) < 2 or len(set(p[2] for p in self.points)) < 2:
            return
        if self.points_dirty:
            self.prepare_data()
        self.model.train()
        for _ in range(epochs_per_frame):
            self.optimizer.zero_grad()
            logits = self.model(self.X_train)
            loss = self.criterion(logits, self.Y_train)
            loss.backward()
            self.optimizer.step()
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.X_train)
            loss_val = self.criterion(logits, self.Y_train).item()
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            acc = (preds == self.Y_train).float().mean().item()
        self.loss_value = loss_val
        self.acc_value = acc * 100.0
        self._generate_boundary()

    def _generate_boundary(self, step=8):
        if self.model is None:
            return
        w = int(DRAW_AREA_W * self.scale_x)
        h = int(DRAW_AREA_H * self.scale_y)
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        surf.set_alpha(165)
        coords, positions = [], []
        for y in range(0, h, step):
            for x in range(0, w, step):
                nx, ny = x / w, y / h
                coords.append([nx, ny])
                positions.append((x, y))
        X = torch.tensor(coords, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X)
            probs = torch.sigmoid(logits).squeeze().numpy()
        for (x, y), p in zip(positions, probs):
            color = self.theme["yellow_soft"] if p >= 0.5 else self.theme["green_soft"]
            pygame.draw.rect(surf, color, (x, y, step, step))
        self.boundary_surface = surf

    # ------- Pre-made datasets -------
    def generate_dataset(self, difficulty):
        """Generate synthetic dataset in base coordinates."""
        self.clear_points()
        if difficulty == 'easy':
            # A simple circle
            for _ in range(300):
                angle = np.random.uniform(0, 2*math.pi)
                r = np.random.uniform(0, 0.7)
                x = 0.5 + r * math.cos(angle) * 0.8
                y = 0.5 + r * math.sin(angle) * 0.8
                px = DRAW_AREA_X + x * DRAW_AREA_W
                py = DRAW_AREA_Y + y * DRAW_AREA_H
                self.points.append((px, py, 1))
            for _ in range(300):
                angle = np.random.uniform(0, 2*math.pi)
                r = np.random.uniform(0.6, 1.0)
                x = 0.5 + r * math.cos(angle) * 0.9
                y = 0.5 + r * math.sin(angle) * 0.9
                px = DRAW_AREA_X + x * DRAW_AREA_W
                py = DRAW_AREA_Y + y * DRAW_AREA_H
                self.points.append((px, py, 0))
        elif difficulty == 'medium':
            # XOR pattern with noise
            for _ in range(400):
                x = np.random.uniform(0, 1)
                y = np.random.uniform(0, 1)
                label = 1 if (x > 0.5) != (y > 0.5) else 0
                px = DRAW_AREA_X + x * DRAW_AREA_W
                py = DRAW_AREA_Y + y * DRAW_AREA_H
                self.points.append((px, py, label))
        elif difficulty == 'hard':
            # Spiral
            for i in range(500):
                r = np.random.uniform(0, 1)
                angle = r * 4 * math.pi
                noise = np.random.normal(0, 0.05)
                if i % 2 == 0:
                    x = 0.5 + (r * math.cos(angle) + noise) * 0.8
                    y = 0.5 + (r * math.sin(angle) + noise) * 0.8
                    px = DRAW_AREA_X + x * DRAW_AREA_W
                    py = DRAW_AREA_Y + y * DRAW_AREA_H
                    self.points.append((px, py, 1))
                else:
                    x = 0.5 + (r * math.cos(angle + math.pi) + noise) * 0.8
                    y = 0.5 + (r * math.sin(angle + math.pi) + noise) * 0.8
                    px = DRAW_AREA_X + x * DRAW_AREA_W
                    py = DRAW_AREA_Y + y * DRAW_AREA_H
                    self.points.append((px, py, 0))
        elif difficulty == 'vhard':
            # Concentric circles with noise
            for _ in range(400):
                angle = np.random.uniform(0, 2*math.pi)
                r = np.random.uniform(0, 0.35)
                x = 0.5 + r * math.cos(angle) * 0.8
                y = 0.5 + r * math.sin(angle) * 0.8
                px = DRAW_AREA_X + x * DRAW_AREA_W
                py = DRAW_AREA_Y + y * DRAW_AREA_H
                self.points.append((px, py, 1))
            for _ in range(400):
                angle = np.random.uniform(0, 2*math.pi)
                r = np.random.uniform(0.4, 0.6)
                x = 0.5 + r * math.cos(angle) * 0.8
                y = 0.5 + r * math.sin(angle) * 0.8
                px = DRAW_AREA_X + x * DRAW_AREA_W
                py = DRAW_AREA_Y + y * DRAW_AREA_H
                self.points.append((px, py, 0))
            for _ in range(200):
                angle = np.random.uniform(0, 2*math.pi)
                r = np.random.uniform(0.65, 1.0)
                x = 0.5 + r * math.cos(angle) * 0.8
                y = 0.5 + r * math.sin(angle) * 0.8
                px = DRAW_AREA_X + x * DRAW_AREA_W
                py = DRAW_AREA_Y + y * DRAW_AREA_H
                self.points.append((px, py, 1))
        self.points_dirty = True
        self.status_text = f"Loaded {difficulty} dataset."

    # ------- JSON save/load -------
    def save_custom_dataset(self):
        """Save current points to custom_dataset.json"""
        data = []
        for px, py, label in self.points:
            data.append({'x': px, 'y': py, 'label': label})
        try:
            with open('custom_dataset.json', 'w') as f:
                json.dump(data, f)
            self.status_text = "Dataset saved to custom_dataset.json"
        except Exception as e:
            self.status_text = f"Error saving dataset: {e}"

    def save_model_config(self):
        """Save model architecture to model.json"""
        config = {
            'layers': self.layer_configs,
            'learning_rate': self.learning_rate
        }
        try:
            with open('model.json', 'w') as f:
                json.dump(config, f, indent=2)
            self.status_text = "Model config saved to model.json"
        except Exception as e:
            self.status_text = f"Error saving model: {e}"

    # ------- Drawing / Data -------
    def add_point(self, pos):
        bx = pos[0] / self.scale_x
        by = pos[1] / self.scale_y
        if DRAW_AREA_X <= bx <= DRAW_AREA_X + DRAW_AREA_W and DRAW_AREA_Y <= by <= DRAW_AREA_Y + DRAW_AREA_H:
            self.points.append((bx, by, self.selected_color))
            self.points_dirty = True

    def clear_points(self):
        self.points.clear()
        self.loss_value = None
        self.acc_value = None
        self.boundary_surface = None
        self.points_dirty = True
        self.status_text = "Points cleared."

    def clear_boundary(self):
        self.boundary_surface = None
        self.loss_value = None
        self.acc_value = None
        self.status_text = "Result cleared."

    # ------- Game screen draw -------
    def draw_game(self, surface):
        self.perform_training_step()
        surface.fill(self.theme["bg"])
        mouse = pygame.mouse.get_pos()

        draw_text(surface, "Draw your colored map – Live Training", FONT_TITLE,
                  self.theme["text"], int(50 * self.scale_x), int(20 * self.scale_y))
        draw_text(surface, "Pick a color, place points. Press Pause to stop.", FONT,
                  self.theme["subtext"], int(52 * self.scale_x), int(68 * self.scale_y))

        # Back & Model Settings buttons hide while popup is open
        if not self.popup.visible:
            self.btn_back_game.draw(surface, self.theme, mouse, self.scale_x, self.scale_y)
            self.btn_model_settings.draw(surface, self.theme, mouse, self.scale_x, self.scale_y)

        # Canvas
        canvas_x = int(DRAW_AREA_X * self.scale_x)
        canvas_y = int(DRAW_AREA_Y * self.scale_y)
        canvas_w = int(DRAW_AREA_W * self.scale_x)
        canvas_h = int(DRAW_AREA_H * self.scale_y)
        canvas_rect = pygame.Rect(canvas_x, canvas_y, canvas_w, canvas_h)
        draw_card(surface, canvas_rect, self.theme, radius=28)
        inner_rect = pygame.Rect(canvas_x+10, canvas_y+10, canvas_w-20, canvas_h-20)
        draw_rounded_rect(surface, inner_rect, self.theme["card_solid"], radius=22, border=2, border_color=self.theme["outline"])

        if self.boundary_surface:
            scaled_boundary = pygame.transform.scale(self.boundary_surface, (canvas_w, canvas_h))
            surface.blit(scaled_boundary, (canvas_x, canvas_y))

        for bx, by, label in self.points:
            sx = bx * self.scale_x
            sy = by * self.scale_y
            color = self.theme["green"] if label == 0 else self.theme["yellow"]
            pygame.draw.circle(surface, color, (int(sx), int(sy)), int(8 * self.scale_x))
            pygame.draw.circle(surface, self.theme["white"], (int(sx), int(sy)), int(8 * self.scale_x), 2)

        # Toolbar
        toolbar_rect = pygame.Rect(int(40 * self.scale_x), int(TOOLBAR_Y * self.scale_y),
                                   int(1200 * self.scale_x), int(TOOLBAR_H * self.scale_y))
        draw_card(surface, toolbar_rect, self.theme, radius=28)

        # Color Palette section
        draw_text(surface, "Color Palette", FONT_MED, self.theme["text"],
                  int(70 * self.scale_x), int(665 * self.scale_y))
        gx, gy = int(self.green_rect.x * self.scale_x), int(self.green_rect.y * self.scale_y)
        gw, gh = int(self.green_rect.w * self.scale_x), int(self.green_rect.h * self.scale_y)
        green_r = pygame.Rect(gx, gy, gw, gh)
        pygame.draw.rect(surface, self.theme["green"], green_r, border_radius=16)
        yx, yy = int(self.yellow_rect.x * self.scale_x), int(self.yellow_rect.y * self.scale_y)
        yw, yh = int(self.yellow_rect.w * self.scale_x), int(self.yellow_rect.h * self.scale_y)
        yellow_r = pygame.Rect(yx, yy, yw, yh)
        pygame.draw.rect(surface, self.theme["yellow"], yellow_r, border_radius=16)
        sel_color = self.theme["neon_purple"]
        if self.selected_color == 0:
            pygame.draw.rect(surface, sel_color, green_r.inflate(int(8*self.scale_x), int(8*self.scale_y)),
                             width=max(1, int(4*self.scale_x)), border_radius=18)
        else:
            pygame.draw.rect(surface, sel_color, yellow_r.inflate(int(8*self.scale_x), int(8*self.scale_y)),
                             width=max(1, int(4*self.scale_x)), border_radius=18)
        draw_text(surface, "Green", FONT, self.theme["white"], green_r.centerx, green_r.centery, center=True)
        draw_text(surface, "Yellow", FONT, self.theme["black"], yellow_r.centerx, yellow_r.centery, center=True)

        # Dataset section
        draw_text(surface, "Datasets", FONT_MED, self.theme["text"],
                  int(70 * self.scale_x), int(730 * self.scale_y))
        self.btn_ds_easy.draw(surface, self.theme, mouse, self.scale_x, self.scale_y)
        self.btn_ds_medium.draw(surface, self.theme, mouse, self.scale_x, self.scale_y)
        self.btn_ds_hard.draw(surface, self.theme, mouse, self.scale_x, self.scale_y)
        self.btn_ds_vhard.draw(surface, self.theme, mouse, self.scale_x, self.scale_y)
        self.btn_save_data.draw(surface, self.theme, mouse, self.scale_x, self.scale_y)

        # Training Control section
        draw_text(surface, "Training Control", FONT_MED, self.theme["text"],
                  int(740 * self.scale_x), int(665 * self.scale_y))
        if self.training_active:
            self.btn_toggle_train.set_text("Pause Training")
            self.btn_toggle_train.color_key = "neon_pink"
            self.btn_toggle_train.hover_color_key = "neon_red"
        else:
            self.btn_toggle_train.set_text("Start Training")
            self.btn_toggle_train.color_key = "neon_purple"
            self.btn_toggle_train.hover_color_key = "neon_pink"
        self.btn_toggle_train.draw(surface, self.theme, mouse, self.scale_x, self.scale_y)
        self.btn_clear_points.draw(surface, self.theme, mouse, self.scale_x, self.scale_y)
        self.btn_clear_boundary.draw(surface, self.theme, mouse, self.scale_x, self.scale_y)

        # Info panel
        info_x = int(360 * self.scale_x)
        info_y = int(660 * self.scale_y)
        info_w = int(360 * self.scale_x)
        info_h = int(84 * self.scale_y)
        info_rect = pygame.Rect(info_x, info_y, info_w, info_h)
        draw_rounded_rect(surface, info_rect, self.theme["card_solid"], radius=20, border=2, border_color=self.theme["outline"])
        arch_text = "Arch: " + " → ".join([f"{cfg['neurons']}({cfg['activation'][:3]})" for cfg in self.layer_configs])
        draw_text(surface, arch_text, FONT_SMALL, self.theme["text"], info_rect.x+15, info_rect.y+12)
        draw_text(surface, f"Points: {len(self.points)}", FONT_SMALL, self.theme["subtext"], info_rect.x+15, info_rect.y+35)
        loss_txt = f"Loss: {self.loss_value:.4f}" if self.loss_value is not None else "Loss: -"
        acc_txt = f"Acc: {self.acc_value:.2f}%" if self.acc_value is not None else "Acc: -"
        draw_text(surface, loss_txt, FONT_SMALL, self.theme["subtext"], info_rect.x+15, info_rect.y+56)
        draw_text(surface, acc_txt, FONT_SMALL, self.theme["subtext"], info_rect.x+180, info_rect.y+56)

        # Status
        status_str = self.status_text
        if self.training_active and self.loss_value is not None:
            status_str += f"   |   live loss: {self.loss_value:.4f}"
        draw_text(surface, status_str, FONT, self.theme["subtext"],
                  int(50 * self.scale_x), int(795 * self.scale_y))

        # Popup (drawn last to be on top)
        self.draw_model_settings_popup(surface)

    # ------- Game event handler -------
    def handle_game_event(self, event):
        if self.popup.visible:
            if self.handle_popup_event(event):
                return

        if self.btn_back_game.clicked(event, self.scale_x, self.scale_y):
            self.stop_training()
            self.state = "MAIN_MENU"
            return
        if self.btn_model_settings.clicked(event, self.scale_x, self.scale_y):
            self.open_model_settings_popup()
            return

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.green_rect.collidepoint(event.pos[0] / self.scale_x, event.pos[1] / self.scale_y):
                self.selected_color = 0
                self.status_text = "Green selected."
                return
            if self.yellow_rect.collidepoint(event.pos[0] / self.scale_x, event.pos[1] / self.scale_y):
                self.selected_color = 1
                self.status_text = "Yellow selected."
                return
            if self.btn_toggle_train.clicked(event, self.scale_x, self.scale_y):
                if self.training_active:
                    self.stop_training()
                else:
                    self.start_training()
                return
            if self.btn_clear_points.clicked(event, self.scale_x, self.scale_y):
                self.clear_points()
                return
            if self.btn_clear_boundary.clicked(event, self.scale_x, self.scale_y):
                self.clear_boundary()
                return

            # Dataset buttons
            if self.btn_ds_easy.clicked(event, self.scale_x, self.scale_y):
                self.generate_dataset('easy')
                return
            if self.btn_ds_medium.clicked(event, self.scale_x, self.scale_y):
                self.generate_dataset('medium')
                return
            if self.btn_ds_hard.clicked(event, self.scale_x, self.scale_y):
                self.generate_dataset('hard')
                return
            if self.btn_ds_vhard.clicked(event, self.scale_x, self.scale_y):
                self.generate_dataset('vhard')
                return
            if self.btn_save_data.clicked(event, self.scale_x, self.scale_y):
                self.save_custom_dataset()
                return

            # Add point on canvas
            canvas = pygame.Rect(
                int(DRAW_AREA_X * self.scale_x),
                int(DRAW_AREA_Y * self.scale_y),
                int(DRAW_AREA_W * self.scale_x),
                int(DRAW_AREA_H * self.scale_y)
            )
            if canvas.collidepoint(event.pos):
                self.add_point(event.pos)
                return

    # ==================== GLOBAL DISPATCH ====================
    def handle_event(self, event):
        if self.state == "MAIN_MENU":
            self.handle_main_menu_event(event)
        elif self.state == "APP_SETTINGS":
            signal = self.handle_app_settings_event(event)
            if signal == "RESIZE":
                return "RESIZE"
        elif self.state == "GAME":
            self.handle_game_event(event)
        return None

    def draw(self, surface):
        if self.state == "MAIN_MENU":
            self.draw_main_menu(surface)
        elif self.state == "APP_SETTINGS":
            self.draw_app_settings(surface)
        elif self.state == "GAME":
            self.draw_game(surface)

# =========================================================
# MAIN LOOP
# =========================================================
def main():
    screen = pygame.display.set_mode((INITIAL_WIDTH, INITIAL_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("NeuroFlow Studio")
    clock = pygame.time.Clock()

    theme = NEON_THEME
    app = App(theme, INITIAL_WIDTH, INITIAL_HEIGHT)

    while True:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            signal = app.handle_event(event)
            if signal == "RESIZE":
                screen = pygame.display.set_mode((app.screen_width, app.screen_height), pygame.RESIZABLE)
                app.popup.center(app.screen_width, app.screen_height)

        app.draw(screen)
        pygame.display.flip()

if __name__ == "__main__":
    main()