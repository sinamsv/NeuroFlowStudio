import sys
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

WIDTH, HEIGHT = 1280, 820
FPS = 60

DRAW_AREA_X = 40
DRAW_AREA_Y = 40
DRAW_AREA_W = 1200
DRAW_AREA_H = 560

TOOLBAR_Y = 630
TOOLBAR_H = 150

MIN_NEURONS = 1
MAX_NEURONS = 128

ACTIVATIONS = ['relu', 'sigmoid', 'tanh', 'linear']
ACT_DISPLAY = {'relu': 'ReLU', 'sigmoid': 'Sigmoid', 'tanh': 'Tanh', 'linear': 'Linear'}

# ---------- Themes ----------
LIGHT_THEME = {
    "bg": (247, 244, 255),
    "card": (255, 255, 255),
    "card_shadow": (220, 215, 235),
    "text": (60, 52, 95),
    "subtext": (110, 102, 145),
    "outline": (215, 205, 240),
    "green": (88, 200, 120),
    "green_soft": (198, 245, 210),
    "yellow": (255, 214, 84),
    "yellow_soft": (255, 241, 188),
    "pink": (255, 145, 190),
    "purple": (160, 120, 255),
    "blue": (120, 185, 255),
    "red": (255, 105, 105),
    "gray": (235, 230, 245),
    "dark_gray": (170, 165, 190),
    "white": (255, 255, 255),
    "black": (20, 20, 20),
    "layer_even": (238, 247, 255),
    "layer_odd": (248, 239, 255),
    "popup_bg": (250, 250, 255),
    "popup_overlay": (0, 0, 0, 80),
}

DARK_THEME = {
    "bg": (30, 30, 45),
    "card": (45, 45, 65),
    "card_shadow": (20, 20, 35),
    "text": (220, 220, 255),
    "subtext": (170, 160, 210),
    "outline": (100, 90, 140),
    "green": (88, 200, 120),
    "green_soft": (60, 150, 90),
    "yellow": (255, 214, 84),
    "yellow_soft": (180, 160, 50),
    "pink": (255, 105, 180),
    "purple": (160, 120, 255),
    "blue": (100, 150, 255),
    "red": (255, 80, 80),
    "gray": (60, 55, 80),
    "dark_gray": (130, 120, 160),
    "white": (240, 240, 255),
    "black": (10, 10, 20),
    "layer_even": (50, 55, 90),
    "layer_odd": (60, 50, 85),
    "popup_bg": (55, 55, 75),
    "popup_overlay": (0, 0, 0, 120),
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
    pygame.draw.rect(surface, color, rect, border_radius=radius)
    if border > 0:
        pygame.draw.rect(surface, border_color, rect, width=border, border_radius=radius)

def draw_card(surface, rect, theme, color=None, shadow=True, radius=20):
    if color is None:
        color = theme["card"]
    if shadow:
        shadow_rect = pygame.Rect(rect.x + 6, rect.y + 8, rect.w, rect.h)
        pygame.draw.rect(surface, theme["card_shadow"], shadow_rect, border_radius=radius)
    pygame.draw.rect(surface, color, rect, border_radius=radius)

class Button:
    def __init__(self, x, y, w, h, text, color_key, hover_color_key=None):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.default_text = text
        self.color_key = color_key
        self.hover_color_key = hover_color_key if hover_color_key else color_key
        self.radius = 16

    def set_text(self, new_text):
        self.text = new_text

    def draw(self, surface, theme, mouse_pos):
        hovered = self.rect.collidepoint(mouse_pos)
        col = theme[self.hover_color_key] if hovered else theme[self.color_key]

        # shadow
        shadow_rect = self.rect.copy()
        shadow_rect.x += 4
        shadow_rect.y += 6
        pygame.draw.rect(surface, theme["card_shadow"], shadow_rect, border_radius=self.radius)

        pygame.draw.rect(surface, col, self.rect, border_radius=self.radius)
        pygame.draw.rect(surface, theme["white"], self.rect, width=2, border_radius=self.radius)

        draw_text(surface, self.text, FONT, theme["white"], self.rect.centerx, self.rect.centery, center=True)

    def clicked(self, event):
        return event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.rect.collidepoint(event.pos)

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

    def draw(self, surface, theme):
        if not self.visible:
            return
        # semi-transparent overlay
        overlay = pygame.Surface((surface.get_width(), surface.get_height()), pygame.SRCALPHA)
        overlay.fill(theme["popup_overlay"])
        surface.blit(overlay, (0, 0))
        # card
        draw_card(surface, self.rect, theme, color=theme["popup_bg"], shadow=True, radius=24)
        pygame.draw.rect(surface, theme["outline"], self.rect, width=2, border_radius=24)

    def handle_event(self, event):
        if not self.visible:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if not self.rect.collidepoint(event.pos):
                return True   # click outside popup – swallow
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
    def __init__(self, theme):
        self.theme = theme
        self.state = "MAIN_MENU"   # MAIN_MENU, GAME, APP_SETTINGS
        self.clock = pygame.time.Clock()

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

        # UI elements
        self._init_main_menu_ui()
        self._init_app_settings_ui()
        self._init_game_ui()
        self._init_popup()

    def switch_theme(self, new_theme):
        self.theme = new_theme

    # ==================== MAIN MENU ====================
    def _init_main_menu_ui(self):
        self.btn_playground = Button(380, 300, 520, 80, "Playground", "purple", "pink")
        self.btn_settings = Button(380, 420, 520, 80, "Settings", "blue", "purple")
        self.btn_quit = Button(560, 540, 160, 56, "Quit", "red", "pink")

    def draw_main_menu(self, surface):
        surface.fill(self.theme["bg"])
        draw_text(surface, "NeuroFlow Studio", FONT_TITLE, self.theme["text"], 640, 120, center=True)
        mouse = pygame.mouse.get_pos()
        self.btn_playground.draw(surface, self.theme, mouse)
        self.btn_settings.draw(surface, self.theme, mouse)
        self.btn_quit.draw(surface, self.theme, mouse)

    def handle_main_menu_event(self, event):
        if self.btn_playground.clicked(event):
            self.state = "GAME"
            self.create_model()
            self.status_text = "Draw points and start training."
        elif self.btn_settings.clicked(event):
            self.state = "APP_SETTINGS"
        elif self.btn_quit.clicked(event):
            pygame.quit()
            sys.exit()

    # ==================== APP SETTINGS ====================
    def _init_app_settings_ui(self):
        self.btn_back_settings = Button(40, 40, 120, 48, "Back", "red", "pink")
        self.btn_toggle_dark = Button(290, 350, 700, 70, "Toggle Dark / Light", "purple", "pink")

    def draw_app_settings(self, surface):
        surface.fill(self.theme["bg"])
        draw_text(surface, "Application Settings", FONT_TITLE, self.theme["text"], 640, 70, center=True)
        mouse = pygame.mouse.get_pos()
        self.btn_back_settings.draw(surface, self.theme, mouse)
        theme_name = "Dark" if self.theme == DARK_THEME else "Light"
        draw_text(surface, f"Current theme: {theme_name}", FONT_BIG, self.theme["text"], 640, 220, center=True)
        self.btn_toggle_dark.draw(surface, self.theme, mouse)

    def handle_app_settings_event(self, event):
        if self.btn_back_settings.clicked(event):
            self.state = "MAIN_MENU"
            return None
        if self.btn_toggle_dark.clicked(event):
            return "TOGGLE_THEME"   # signal to main loop
        return None

    # ==================== GAME ====================
    def _init_game_ui(self):
        self.btn_back_game = Button(1095, 30, 120, 44, "Back", "red", "pink")
        self.btn_model_settings = Button(1095, 755, 155, 44, "Model Settings", "purple", "pink")
        self.btn_toggle_train = Button(740, 685, 190, 56, "Start Training", "purple", "pink")
        self.btn_clear_points = Button(945, 685, 140, 56, "Clear Points", "pink", "red")
        self.btn_clear_boundary = Button(1100, 685, 140, 56, "Clear Result", "blue", "purple")
        self.green_rect = pygame.Rect(70, 690, 120, 42)
        self.yellow_rect = pygame.Rect(210, 690, 120, 42)

    # ------- Model settings popup -------
    def _init_popup(self):
        self.popup = Popup(800, 550)
        self.popup_elements = []           # dynamic list of layer-related buttons
        self.btn_close_popup = Button(0, 0, 100, 44, "Close", "red", "pink")   # position set later
        self.btn_add_layer_popup = Button(0, 0, 140, 44, "+ Layer", "purple", "pink")
        self.btn_remove_layer_popup = Button(0, 0, 140, 44, "- Layer", "pink", "red")
        self.btn_lr_minus = Button(0, 0, 44, 36, "-", "pink", "red")
        self.btn_lr_plus = Button(0, 0, 44, 36, "+", "purple", "pink")

    def open_model_settings_popup(self):
        self.popup.show()
        self._build_popup_elements()

    def _build_popup_elements(self):
        self.popup_elements.clear()
        x0 = self.popup.rect.x + 30
        y = self.popup.rect.y + 70
        for i in range(len(self.layer_configs)):
            row_y = y + i * 55
            btn_minus = Button(x0 + 300, row_y, 44, 36, "-", "pink", "red")
            btn_plus = Button(x0 + 350, row_y, 44, 36, "+", "purple", "pink")
            btn_act_left = Button(x0 + 450, row_y, 36, 36, "<", "purple", "pink")
            btn_act_right = Button(x0 + 500, row_y, 36, 36, ">", "purple", "pink")
            self.popup_elements.append((i, btn_minus, btn_plus, btn_act_left, btn_act_right))

        # Add/Remove layer buttons
        btn_y = y + len(self.layer_configs) * 55 + 20
        self.btn_add_layer_popup.rect.topleft = (x0, btn_y)
        self.btn_remove_layer_popup.rect.topleft = (x0 + 160, btn_y)
        # Learning rate buttons
        lr_y = y + len(self.layer_configs) * 55 + 90
        self.btn_lr_minus.rect.topleft = (x0 + 300, lr_y)
        self.btn_lr_plus.rect.topleft = (x0 + 350, lr_y)
        # Close button
        self.btn_close_popup.rect.topleft = (self.popup.rect.right - 140, self.popup.rect.bottom - 70)

    def draw_model_settings_popup(self, surface):
        if not self.popup.visible:
            return
        self.popup.draw(surface, self.theme)
        # Title
        draw_text(surface, "Model Settings", FONT_BIG, self.theme["text"],
                  self.popup.rect.centerx, self.popup.rect.y + 25, center=True)
        x0 = self.popup.rect.x + 30
        y = self.popup.rect.y + 70
        mouse = pygame.mouse.get_pos()

        # Draw each layer row
        for i, cfg in enumerate(self.layer_configs):
            row_y = y + i * 55
            draw_text(surface, f"Layer {i+1}: {cfg['neurons']} neurons", FONT, self.theme["text"], x0, row_y+10)
            act_text = ACT_DISPLAY[cfg['activation']]
            draw_text(surface, f"Act: {act_text}", FONT, self.theme["subtext"], x0 + 400, row_y+10)

            # buttons
            _, btn_minus, btn_plus, btn_act_left, btn_act_right = self.popup_elements[i]
            btn_minus.draw(surface, self.theme, mouse)
            btn_plus.draw(surface, self.theme, mouse)
            btn_act_left.draw(surface, self.theme, mouse)
            btn_act_right.draw(surface, self.theme, mouse)

        self.btn_add_layer_popup.draw(surface, self.theme, mouse)
        self.btn_remove_layer_popup.draw(surface, self.theme, mouse)

        lr_y = y + len(self.layer_configs) * 55 + 90
        draw_text(surface, f"Learning Rate: {self.learning_rate:.4f}", FONT_MED, self.theme["text"], x0, lr_y+5)
        self.btn_lr_minus.draw(surface, self.theme, mouse)
        self.btn_lr_plus.draw(surface, self.theme, mouse)

        self.btn_close_popup.draw(surface, self.theme, mouse)

    def handle_popup_event(self, event):
        if not self.popup.visible:
            return False
        if self.popup.handle_event(event):
            return True   # click outside, swallow

        if self.btn_close_popup.clicked(event):
            self.popup.hide()
            return True

        # Layer-specific buttons
        for i, btn_minus, btn_plus, btn_act_left, btn_act_right in self.popup_elements:
            if btn_minus.clicked(event):
                self.layer_configs[i]['neurons'] = max(MIN_NEURONS, self.layer_configs[i]['neurons'] - 1)
                self._rebuild_model()
                return True
            if btn_plus.clicked(event):
                self.layer_configs[i]['neurons'] = min(MAX_NEURONS, self.layer_configs[i]['neurons'] + 1)
                self._rebuild_model()
                return True
            if btn_act_left.clicked(event):
                idx = ACTIVATIONS.index(self.layer_configs[i]['activation'])
                self.layer_configs[i]['activation'] = ACTIVATIONS[(idx - 1) % len(ACTIVATIONS)]
                self._rebuild_model()
                return True
            if btn_act_right.clicked(event):
                idx = ACTIVATIONS.index(self.layer_configs[i]['activation'])
                self.layer_configs[i]['activation'] = ACTIVATIONS[(idx + 1) % len(ACTIVATIONS)]
                self._rebuild_model()
                return True

        if self.btn_add_layer_popup.clicked(event):
            self.layer_configs.append({'neurons': 8, 'activation': 'relu'})
            self._rebuild_model()
            self._build_popup_elements()
            return True
        if self.btn_remove_layer_popup.clicked(event) and len(self.layer_configs) > 0:
            self.layer_configs.pop()
            self._rebuild_model()
            self._build_popup_elements()
            return True

        if self.btn_lr_minus.clicked(event):
            self.learning_rate = max(0.0001, self.learning_rate * 0.5)
            self._update_optimizer_lr()
            return True
        if self.btn_lr_plus.clicked(event):
            self.learning_rate = min(10.0, self.learning_rate * 2.0)
            self._update_optimizer_lr()
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
        # Update boundary surface
        self._generate_boundary()

    def _generate_boundary(self, step=8):
        if self.model is None:
            return
        surf = pygame.Surface((DRAW_AREA_W, DRAW_AREA_H), pygame.SRCALPHA)
        surf.set_alpha(165)
        coords, positions = [], []
        for y in range(0, DRAW_AREA_H, step):
            for x in range(0, DRAW_AREA_W, step):
                nx, ny = x / DRAW_AREA_W, y / DRAW_AREA_H
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

    # ------- Drawing / Data -------
    def add_point(self, pos):
        x, y = pos
        if DRAW_AREA_X <= x <= DRAW_AREA_X + DRAW_AREA_W and DRAW_AREA_Y <= y <= DRAW_AREA_Y + DRAW_AREA_H:
            self.points.append((x, y, self.selected_color))
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
        # Run training step before drawing
        self.perform_training_step()

        surface.fill(self.theme["bg"])
        mouse = pygame.mouse.get_pos()

        draw_text(surface, "Draw your colored map – Live Training", FONT_TITLE, self.theme["text"], 50, 20)
        draw_text(surface, "Pick a color, place points. Press Pause to stop.", FONT, self.theme["subtext"], 52, 68)

        self.btn_back_game.draw(surface, self.theme, mouse)
        self.btn_model_settings.draw(surface, self.theme, mouse)

        # Canvas
        canvas_rect = pygame.Rect(DRAW_AREA_X, DRAW_AREA_Y, DRAW_AREA_W, DRAW_AREA_H)
        draw_card(surface, canvas_rect, self.theme, radius=28)
        inner_rect = pygame.Rect(DRAW_AREA_X+10, DRAW_AREA_Y+10, DRAW_AREA_W-20, DRAW_AREA_H-20)
        draw_rounded_rect(surface, inner_rect, self.theme["gray"], radius=22, border=2, border_color=self.theme["outline"])

        if self.boundary_surface:
            surface.blit(self.boundary_surface, (DRAW_AREA_X, DRAW_AREA_Y))

        for x, y, label in self.points:
            color = self.theme["green"] if label == 0 else self.theme["yellow"]
            pygame.draw.circle(surface, color, (x, y), 8)
            pygame.draw.circle(surface, self.theme["white"], (x, y), 8, 2)

        # Toolbar
        toolbar_rect = pygame.Rect(40, TOOLBAR_Y, 1200, TOOLBAR_H)
        draw_card(surface, toolbar_rect, self.theme, radius=28)
        draw_text(surface, "Color Palette", FONT_MED, self.theme["text"], 70, 655)
        draw_text(surface, "Training Control", FONT_MED, self.theme["text"], 740, 655)

        pygame.draw.rect(surface, self.theme["green"], self.green_rect, border_radius=16)
        pygame.draw.rect(surface, self.theme["yellow"], self.yellow_rect, border_radius=16)
        sel_color = self.theme["purple"]
        if self.selected_color == 0:
            pygame.draw.rect(surface, sel_color, self.green_rect.inflate(8,8), width=4, border_radius=18)
        else:
            pygame.draw.rect(surface, sel_color, self.yellow_rect.inflate(8,8), width=4, border_radius=18)
        draw_text(surface, "Green", FONT, self.theme["white"], self.green_rect.centerx, self.green_rect.centery, center=True)
        draw_text(surface, "Yellow", FONT, self.theme["black"], self.yellow_rect.centerx, self.yellow_rect.centery, center=True)

        # Info panel
        info_rect = pygame.Rect(360, 655, 360, 84)
        draw_rounded_rect(surface, info_rect, self.theme["gray"], radius=20, border=2, border_color=self.theme["outline"])
        arch_text = "Arch: " + " → ".join([f"{cfg['neurons']}({cfg['activation'][:3]})" for cfg in self.layer_configs])
        draw_text(surface, arch_text, FONT_SMALL, self.theme["text"], info_rect.x+15, info_rect.y+12)
        draw_text(surface, f"Points: {len(self.points)}", FONT_SMALL, self.theme["subtext"], info_rect.x+15, info_rect.y+35)
        loss_txt = f"Loss: {self.loss_value:.4f}" if self.loss_value is not None else "Loss: -"
        acc_txt = f"Acc: {self.acc_value:.2f}%" if self.acc_value is not None else "Acc: -"
        draw_text(surface, loss_txt, FONT_SMALL, self.theme["subtext"], info_rect.x+15, info_rect.y+56)
        draw_text(surface, acc_txt, FONT_SMALL, self.theme["subtext"], info_rect.x+180, info_rect.y+56)

        # Toggle train button
        if self.training_active:
            self.btn_toggle_train.set_text("Pause Training")
            self.btn_toggle_train.color_key = "pink"
            self.btn_toggle_train.hover_color_key = "red"
        else:
            self.btn_toggle_train.set_text("Start Training")
            self.btn_toggle_train.color_key = "purple"
            self.btn_toggle_train.hover_color_key = "pink"
        self.btn_toggle_train.draw(surface, self.theme, mouse)
        self.btn_clear_points.draw(surface, self.theme, mouse)
        self.btn_clear_boundary.draw(surface, self.theme, mouse)

        # Status
        status_str = self.status_text
        if self.training_active and self.loss_value is not None:
            status_str += f"   |   live loss: {self.loss_value:.4f}"
        draw_text(surface, status_str, FONT, self.theme["subtext"], 50, 777)

        # Popup
        self.draw_model_settings_popup(surface)

    # ------- Game event handler -------
    def handle_game_event(self, event):
        # اولویت با پاپ‌آپ
        if self.popup.visible:
            if self.handle_popup_event(event):
                return

        if self.btn_back_game.clicked(event):
            self.stop_training()
            self.state = "MAIN_MENU"
            return
        if self.btn_model_settings.clicked(event):
            self.open_model_settings_popup()
            return

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.green_rect.collidepoint(event.pos):
                self.selected_color = 0
                self.status_text = "Green selected."
                return
            if self.yellow_rect.collidepoint(event.pos):
                self.selected_color = 1
                self.status_text = "Yellow selected."
                return
            if self.btn_toggle_train.rect.collidepoint(event.pos):
                if self.training_active:
                    self.stop_training()
                else:
                    self.start_training()
                return
            if self.btn_clear_points.rect.collidepoint(event.pos):
                self.clear_points()
                return
            if self.btn_clear_boundary.rect.collidepoint(event.pos):
                self.clear_boundary()
                return
            # Add point on canvas
            canvas = pygame.Rect(DRAW_AREA_X, DRAW_AREA_Y, DRAW_AREA_W, DRAW_AREA_H)
            if canvas.collidepoint(event.pos):
                self.add_point(event.pos)
                return

    # ==================== GLOBAL DISPATCH ====================
    def handle_event(self, event):
        if self.state == "MAIN_MENU":
            self.handle_main_menu_event(event)
        elif self.state == "APP_SETTINGS":
            signal = self.handle_app_settings_event(event)
            if signal == "TOGGLE_THEME":
                return "TOGGLE_THEME"
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
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("NeuroFlow Studio")
    clock = pygame.time.Clock()

    current_theme = LIGHT_THEME
    app = App(current_theme)

    while True:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            signal = app.handle_event(event)
            if signal == "TOGGLE_THEME":
                current_theme = DARK_THEME if current_theme == LIGHT_THEME else LIGHT_THEME
                app.switch_theme(current_theme)

        app.draw(screen)
        pygame.display.flip()

if __name__ == "__main__":
    main()