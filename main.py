# main.py
import pygame
import pygame_gui
import math
import sys
import json
import os

# Controlador de gestos (requiere: pip install mediapipe opencv-python)
try:
    from gesture_controller import GestureController, MEDIAPIPE_OK
except ImportError:
    MEDIAPIPE_OK = False
    GestureController = None

class OscillationGraph:
    def __init__(self, rect):
        self.rect = pygame.Rect(rect)
        self.points = []
        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", 12)
        
    def add_point(self, t, y):
        self.points.append((t, y))
        # OPTIMIZACIÓN: Solo retener los últimos segundos. Evita consumo infinito de RAM y lag de recálculo
        if len(self.points) > 400:
            self.points = self.points[-300:]
        
    def draw(self, surface):
        # Fondo oscuro
        pygame.draw.rect(surface, (20, 20, 25), self.rect)
        pygame.draw.rect(surface, (100, 100, 100), self.rect, 2)
        
        y0_px = self.rect.y + self.rect.height / 2
        # Línea cero (eje horizontal)
        pygame.draw.line(surface, (80, 80, 80), (self.rect.x, y0_px), (self.rect.right, y0_px), 1)

        if not self.points: return
        
        max_t = self.points[-1][0]
        window_sec = 5.0
        min_t = max(0.0, max_t - window_sec)
        
        # Filtrar puntos visibles
        view_points = [(tf, yf) for tf, yf in self.points if tf >= min_t]
        if len(view_points) < 2: return
        
        # Determinar amplitud máxima visible
        max_abs_y = max(0.1, max(abs(yf) for _, yf in view_points)) * 1.5
        
        screen_points = []
        for tf, yf in view_points:
            px = self.rect.x + ((tf - min_t) / window_sec) * self.rect.width
            py = self.rect.y + self.rect.height / 2 - (yf / max_abs_y) * (self.rect.height / 2)
            screen_points.append((px, py))
            
        # Relleno semitransparente (área bajo la curva respecto a cero)
        fill_surface = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        poly_points = [(p[0] - self.rect.x, p[1] - self.rect.y) for p in screen_points]
        poly_points.append((screen_points[-1][0] - self.rect.x, y0_px - self.rect.y))
        poly_points.append((screen_points[0][0] - self.rect.x, y0_px - self.rect.y))
        
        if len(poly_points) >= 3:
            pygame.draw.polygon(fill_surface, (0, 150, 255, 60), poly_points)
        surface.blit(fill_surface, self.rect.topleft)
        
        # Línea de la señal
        pygame.draw.lines(surface, (0, 200, 255), False, screen_points, 2)
        
        # Etiquetas de los ejes
        try:
            lbl_max = self.font.render(f"{max_abs_y:.2f} m", True, (180, 180, 180))
            lbl_min = self.font.render(f"{-max_abs_y:.2f} m", True, (180, 180, 180))
            lbl_zero = self.font.render("0 m", True, (180, 180, 180))
            surface.blit(lbl_max, (self.rect.x + 5, self.rect.y + 5))
            surface.blit(lbl_min, (self.rect.x + 5, self.rect.bottom - 20))
            surface.blit(lbl_zero, (self.rect.x + 5, y0_px - 15))
            
            lbl_t_start = self.font.render(f"{min_t:.1f} s", True, (180, 180, 180))
            lbl_t_end = self.font.render(f"{min_t + window_sec:.1f} s", True, (180, 180, 180))
            surface.blit(lbl_t_start, (self.rect.x + 5, y0_px + 5))
            surface.blit(lbl_t_end, (self.rect.right - 40, y0_px + 5))
        except:
            pass

class CarSimulation:
    def __init__(self, rect, graph):
        self.rect = pygame.Rect(rect)
        self.graph = graph
        
        # Parámetros por defecto
        self.m = 1000.0
        self.k = 20000.0
        self.c = 1500.0
        self.h = 0.1
        self.v_kmh = 30.0
        self.L = 2.0
        
        # Dimensiones del auto y offsets de posicionamiento
        self.wheel_dist = 1.325
        self.body_offset_x = -5.25  # Offset horizontal del chasis vs centro de ruedas
        self.body_offset_y = 30     # Offset vertical del chasis
        self.fw_offset_x = 0.0     # Offset visual llanta delantera
        self.fw_offset_y = 0.0
        self.rw_offset_x = 0.0     # Offset visual llanta trasera
        self.rw_offset_y = 0.0
        self.bumps = [5.0]
        self.running = False
        self.debug_mode = False
        
        # Estado de drag para debug
        self._drag_target = None  # 'body', 'fw', 'rw'
        self._drag_offset = (0, 0)
        self._debug_positions = {}  # Posiciones de pantalla para hit-test
        
        self._load_config()
        self.load_assets()
        self.reset()
    
    def _config_path(self):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'car_config.json')
    
    def _load_config(self):
        try:
            with open(self._config_path(), 'r') as f:
                cfg = json.load(f)
            self.wheel_dist = cfg.get('wheel_dist', self.wheel_dist)
            self.body_offset_x = cfg.get('body_offset_x', self.body_offset_x)
            self.body_offset_y = cfg.get('body_offset_y', self.body_offset_y)
            self.fw_offset_x = cfg.get('fw_offset_x', self.fw_offset_x)
            self.fw_offset_y = cfg.get('fw_offset_y', self.fw_offset_y)
            self.rw_offset_x = cfg.get('rw_offset_x', self.rw_offset_x)
            self.rw_offset_y = cfg.get('rw_offset_y', self.rw_offset_y)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    
    def _save_config(self):
        cfg = {
            'wheel_dist': round(self.wheel_dist, 4),
            'body_offset_x': round(self.body_offset_x, 2),
            'body_offset_y': round(self.body_offset_y, 2),
            'fw_offset_x': round(self.fw_offset_x, 2),
            'fw_offset_y': round(self.fw_offset_y, 2),
            'rw_offset_x': round(self.rw_offset_x, 2),
            'rw_offset_y': round(self.rw_offset_y, 2)
        }
        with open(self._config_path(), 'w') as f:
            json.dump(cfg, f, indent=2)
        print(f'[DEBUG] Configuración guardada: {cfg}')
        
    def _remove_green(self, surface):
        pixel_array = pygame.PixelArray(surface)
        width, height = surface.get_size()
        for x in range(width):
            for y in range(height):
                color = surface.unmap_rgb(pixel_array[x, y])
                r, g, b, a = color
                # Remoción ultra-agresiva: Carga verde de bordes anti-aliased
                if g > 50 and g > r * 1.15 and g > b * 1.15:
                    pixel_array[x, y] = (0, 0, 0, 0)
        pixel_array.close()

    def _crop_alpha(self, surface):
        rect = surface.get_bounding_rect()
        if rect.width > 0 and rect.height > 0:
            return surface.subsurface(rect).copy()
        return surface

    def load_assets(self):
        # Buscar imágenes en la misma carpeta que este .py (funciona al hacer EXE también)
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        body_path  = os.path.join(BASE_DIR, "car_body.png")
        wheel_path = os.path.join(BASE_DIR, "car_wheel.png")

        self.car_body_img = None
        self.car_wheel_img = None
        try:
            img = pygame.image.load(body_path).convert_alpha()
            self._remove_green(img)
            img = self._crop_alpha(img)
            w, h = img.get_size()
            new_w = 230
            new_h = int(h * new_w / w) if w > 0 else 60
            img = pygame.transform.smoothscale(img, (new_w, new_h))
            self.car_body_img = pygame.transform.flip(img, True, False)
        except Exception as e:
            print(f"[ASSETS] car_body.png no cargó: {e}")

        try:
            img = pygame.image.load(wheel_path).convert_alpha()
            self._remove_green(img)
            img = self._crop_alpha(img)
            # Aumentar tamaño de rueda un poco para acentuar efecto deportivo
            self.car_wheel_img = pygame.transform.smoothscale(img, (48, 48))
        except Exception:
            pass
        
    def reset(self):
        self.time = 0.0
        self.car_x = 0.0
        self.y = 0.0
        self.v_y = 0.0
        self.prev_road = 0.0
        self.graph.points.clear()
        if getattr(self, 'bumps', None) is None:
            self.bumps = [5.0]
            
    def add_bump(self):
        last_x = max(self.bumps + [self.car_x])
        # Agregar el bache un poco más adelante para que no aparezca de golpe
        new_x = max(last_x + 3.0, self.car_x + 5.0)
        self.bumps.append(new_x)
        
    def get_road_y(self, x):
        total_y = 0.0
        for bx in self.bumps:
            if bx <= x <= bx + self.L:
                total_y += self.h * math.sin(math.pi * (x - bx) / self.L)
        return total_y
        
    def update_physics(self, dt=0.016):
        if not self.running: return
        
        velocity = self.v_kmh / 3.6
        self.car_x += velocity * dt
        self.time += dt
        
        # CLAVE PARA QUE LA FÍSICA SE ALINEE CON LA ANIMACIÓN VISUAL ("coliciones reales"):
        # Evaluamos el desnivel del asfalto individualmente para cada rueda,
        # para que la carrocería en un sistema '1 DOF' sienta el impacto exactamente
        # cuando cada una de las llantas reacciona, y no en un solo punto imaginario del centro.
        road_f = self.get_road_y(self.car_x + self.wheel_dist)
        road_r = self.get_road_y(self.car_x - self.wheel_dist)
        road = (road_f + road_r) / 2.0
        
        road_vel = (road - self.prev_road) / dt if self.time > dt else 0.0
        
        # Ec. Diferencial de movimiento para oscilador amortiguado
        accel = (-self.k * (self.y - road) - self.c * (self.v_y - road_vel)) / self.m
        
        # Integración de Euler explícita pura garantizando un paso de dt=0.016
        self.v_y += accel * dt
        self.y += self.v_y * dt
        
        self.prev_road = road
        self.graph.add_point(self.time, self.y)
        
    def draw(self, surface):
        # Cielo
        pygame.draw.rect(surface, (135, 206, 235), self.rect)
        
        # Nubes animadas para aumentar realismo e inmersión
        cloud_speed = 0.1
        velocity = self.v_kmh / 3.6
        scale_x = 50 
        for cx, cy in [(100, 80), (400, 50), (700, 100), (250, 120)]:
            shifted_x = cx - (self.time * velocity * cloud_speed * scale_x)
            shifted_x = shifted_x % (self.rect.width + 100) - 50
            
            pygame.draw.circle(surface, (255, 255, 255), (int(shifted_x), self.rect.y + cy), 30)
            pygame.draw.circle(surface, (255, 255, 255), (int(shifted_x + 25), self.rect.y + cy - 15), 40)
            pygame.draw.circle(surface, (255, 255, 255), (int(shifted_x + 50), self.rect.y + cy), 30)

        # Configuración de escalas visuales
        base_y = 350
        scale_y = 200 # Píxeles por metro vertical (hace visibles baches de 2-20 cm)
        
        # Cámara rastreando fijamente al auto en la pantalla (X estático)
        car_screen_x = self.rect.x + 200
        
        # Renderizado optimizado del asfalto (saltos iterativos para ahorrar muchísima CPU)
        road_points = []
        for px in range(self.rect.x, self.rect.right + 4, 4):
            phys_x = self.car_x + (px - car_screen_x) / scale_x
            r_y = self.get_road_y(phys_x)
            py = self.rect.y + base_y - r_y * scale_y
            road_points.append((px, py))
            
        road_poly = road_points.copy()
        road_poly.append((self.rect.right, self.rect.bottom))
        road_poly.append((self.rect.x, self.rect.bottom))
        pygame.draw.polygon(surface, (50, 50, 50), road_poly)
        if len(road_points) >= 2:
            pygame.draw.lines(surface, (200, 200, 200), False, road_points, 3)
            
        # Calcular elevación real de las ruedas pegadas al asfalto
        front_wheel_phys_x = self.car_x + self.wheel_dist
        rear_wheel_phys_x = self.car_x - self.wheel_dist
        
        fw_px = car_screen_x + self.wheel_dist * scale_x
        rw_px = car_screen_x - self.wheel_dist * scale_x
        
        fw_py = self.rect.y + base_y - self.get_road_y(front_wheel_phys_x) * scale_y - 20
        rw_py = self.rect.y + base_y - self.get_road_y(rear_wheel_phys_x) * scale_y - 20
        
        # Geometría y altura dinámica del chasis del carro según "self.y"
        body_y = self.rect.y + base_y - self.y * scale_y - 70

        # PITCH VISUAL PARA REALISMO: Calcular ángulo del coche basado en diferencia de llantas
        dy = rw_py - fw_py
        dx = fw_px - rw_px
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        
        # Desfasar ligeramente los anclajes de la suspensión si el auto está inclinado
        fw_body_y = body_y - (dy / 2.0)
        rw_body_y = body_y + (dy / 2.0)

        # Aplicar offsets visuales individuales a cada rueda
        fw_draw_x = fw_px + self.fw_offset_x
        fw_draw_y = fw_py + self.fw_offset_y
        rw_draw_x = rw_px + self.rw_offset_x
        rw_draw_y = rw_py + self.rw_offset_y

        # Resortes solo visibles si la llanta se despega bruscamente en el bache o salto
        if fw_draw_y > fw_body_y + 35:
            pygame.draw.line(surface, (30, 30, 30), (fw_draw_x, fw_body_y + 35), (fw_draw_x, fw_draw_y), 5)
        if rw_draw_y > rw_body_y + 35:
            pygame.draw.line(surface, (30, 30, 30), (rw_draw_x, rw_body_y + 35), (rw_draw_x, rw_draw_y), 5)
        
        # 1. DIBUJAR RUEDAS PRIMERO (Para que el chasis las tape y no sobresalgan)
        wheel_radius_m = 20 / scale_y
        spin_angle = (self.car_x / wheel_radius_m) if wheel_radius_m > 0 else 0
        
        def draw_wheel(x, y, angle):
            if self.car_wheel_img:
                rotated_img = pygame.transform.rotate(self.car_wheel_img, -math.degrees(angle))
                rect = rotated_img.get_rect(center=(int(x), int(y)))
                surface.blit(rotated_img, rect)
            else:
                pygame.draw.circle(surface, (30, 30, 30), (x, y), 20)
                pygame.draw.circle(surface, (150, 150, 150), (x, y), 8)
                for i in range(5):
                    theta = angle + i * (2 * math.pi / 5)
                    ex = x + math.cos(theta) * 20
                    ey = y + math.sin(theta) * 20
                    pygame.draw.line(surface, (200, 200, 200), (x, y), (ex, ey), 3)

        draw_wheel(fw_draw_x, fw_draw_y, spin_angle)
        draw_wheel(rw_draw_x, rw_draw_y, spin_angle)

        # 2. DIBUJAR CHASIS — Posicionado con offsets configurables
        if self.car_body_img:
            wheels_mid_x = (fw_px + rw_px) / 2.0
            body_cx = wheels_mid_x + self.body_offset_x
            body_cy = body_y + self.body_offset_y
            
            rotated_car = pygame.transform.rotate(self.car_body_img, angle_deg)
            rect = rotated_car.get_rect(center=(int(body_cx), int(body_cy)))
            surface.blit(rotated_car, rect)
        else:
            body_cx = car_screen_x
            body_cy = body_y
            car_rect = pygame.Rect(car_screen_x - 60, body_y - 20, 120, 40)
            roof_rect = pygame.Rect(car_screen_x - 30, body_y - 50, 60, 30)
            pygame.draw.rect(surface, (200, 50, 50), car_rect, border_radius=10)
            pygame.draw.rect(surface, (200, 50, 50), roof_rect, border_radius=10)
            pygame.draw.rect(surface, (150, 200, 255), (car_screen_x - 25, body_y - 45, 20, 20))
            pygame.draw.rect(surface, (150, 200, 255), (car_screen_x + 5, body_y - 45, 20, 20))
        
        # Guardar posiciones para hit-test de drag
        self._debug_positions = {
            'body': (body_cx, body_cy),
            'fw': (fw_draw_x, fw_draw_y),
            'rw': (rw_draw_x, rw_draw_y)
        }
        
        # HUD de modo debug
        if self.debug_mode:
            # Highlight de elementos arrastrables
            for key, (dx, dy) in self._debug_positions.items():
                color = (255, 255, 0) if key == 'body' else (0, 255, 255)
                if self._drag_target == key:
                    color = (255, 100, 100)
                pygame.draw.circle(surface, color, (int(dx), int(dy)), 28, 2)
                lbl_font = pygame.font.SysFont('Arial', 10)
                label = {'body': 'CHASIS', 'fw': 'DELANTERA', 'rw': 'TRASERA'}[key]
                lbl_surf = lbl_font.render(label, True, color)
                surface.blit(lbl_surf, (int(dx) - lbl_surf.get_width()//2, int(dy) - 42))
            
            debug_font = pygame.font.SysFont('Arial', 14, bold=True)
            lines = [
                '=== MODO DEBUG (F1 salir) ===',
                'Arrastra con mouse | Flechas: chasis | +/-: ruedas',
                'S: Guardar  |  Shift: fino  |  R: Reset offsets',
                '',
                f'body_offset: ({self.body_offset_x:.1f}, {self.body_offset_y:.1f})',
                f'fw_offset:   ({self.fw_offset_x:.1f}, {self.fw_offset_y:.1f})',
                f'rw_offset:   ({self.rw_offset_x:.1f}, {self.rw_offset_y:.1f})',
                f'wheel_dist:  {self.wheel_dist:.4f}',
            ]
            dbg_surface = pygame.Surface((340, len(lines) * 18 + 10), pygame.SRCALPHA)
            dbg_surface.fill((0, 0, 0, 180))
            surface.blit(dbg_surface, (self.rect.x + 5, self.rect.y + 5))
            
            for i, line in enumerate(lines):
                color = (255, 255, 0) if i == 0 else (200, 255, 200)
                txt = debug_font.render(line, True, color)
                surface.blit(txt, (self.rect.x + 10, self.rect.y + 8 + i * 18))
    
    def handle_debug_mouse(self, event):
        """Maneja eventos de mouse para arrastrar elementos en modo debug."""
        if not self.debug_mode:
            return
        
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            # Buscar el elemento más cercano al click
            best = None
            best_dist = 40  # Radio máximo de captura
            for key, (px, py) in self._debug_positions.items():
                dist = math.hypot(mx - px, my - py)
                if dist < best_dist:
                    best = key
                    best_dist = dist
            if best:
                self._drag_target = best
                px, py = self._debug_positions[best]
                self._drag_offset = (mx - px, my - py)
        
        elif event.type == pygame.MOUSEMOTION and self._drag_target:
            mx, my = event.pos
            target_x = mx - self._drag_offset[0]
            target_y = my - self._drag_offset[1]
            
            if self._drag_target == 'body':
                # Calcular nuevo offset relativo al centro de ruedas
                wheels_mid_x = (self._debug_positions['fw'][0] - self.fw_offset_x + 
                               self._debug_positions['rw'][0] - self.rw_offset_x) / 2.0
                base_body_y = self._debug_positions['body'][1] - self.body_offset_y
                self.body_offset_x = target_x - wheels_mid_x
                self.body_offset_y = target_y - base_body_y
            
            elif self._drag_target == 'fw':
                base_x = self._debug_positions['fw'][0] - self.fw_offset_x
                base_y = self._debug_positions['fw'][1] - self.fw_offset_y
                self.fw_offset_x = target_x - base_x
                self.fw_offset_y = target_y - base_y
            
            elif self._drag_target == 'rw':
                base_x = self._debug_positions['rw'][0] - self.rw_offset_x
                base_y = self._debug_positions['rw'][1] - self.rw_offset_y
                self.rw_offset_x = target_x - base_x
                self.rw_offset_y = target_y - base_y
        
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._drag_target = None

class ControlPanel:
    def __init__(self, rect, manager, sim):
        self.rect = pygame.Rect(rect)
        self.manager = manager
        self.sim = sim
        self.sliders = {}
        self.labels = {}
        
        params = [
            {"id": "m", "name": "Masa", "min": 500, "max": 2000, "default": self.sim.m, "unit": "kg"},
            {"id": "k", "name": "Resorte (k)", "min": 5000, "max": 50000, "default": self.sim.k, "unit": "N/m"},
            {"id": "c", "name": "Amortig. (c)", "min": 100, "max": 25000, "default": self.sim.c, "unit": "N·s/m"},
            {"id": "h", "name": "Prof. Bache", "min": 2, "max": 20, "default": self.sim.h * 100, "unit": "cm"},
            {"id": "v", "name": "Velocidad", "min": 10, "max": 100, "default": self.sim.v_kmh, "unit": "km/h"}
        ]
        
        y = self.rect.y + 20
        x = self.rect.x + 10
        w = self.rect.width - 20
        
        for p in params:
            lbl = pygame_gui.elements.UILabel(
                relative_rect=pygame.Rect(x, y, w, 25),
                text=f"{p['name']}: {int(p['default'])} {p['unit']}",
                manager=self.manager
            )
            self.labels[p['id']] = (lbl, p)
            
            slider = pygame_gui.elements.UIHorizontalSlider(
                relative_rect=pygame.Rect(x, y + 25, w, 20),
                start_value=p['default'],
                value_range=(p['min'], p['max']),
                manager=self.manager
            )
            self.sliders[p['id']] = slider
            y += 45
            
        y += 10
        self.btn_iniciar = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(x, y, w, 35),
            text="Iniciar",
            manager=self.manager
        )
        self.btn_pausar = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(x, y + 40, w, 35),
            text="Pausar",
            manager=self.manager
        )
        self.btn_reiniciar = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(x, y + 80, w, 35),
            text="Reiniciar",
            manager=self.manager
        )
        self.btn_add_bump = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(x, y + 120, w, 35),
            text="+ Añadir Bache",
            manager=self.manager
        )
        
    def update_labels(self):
        for pid, slider in self.sliders.items():
            lbl, p = self.labels[pid]
            val = round(slider.get_current_value(), 1)
            if p['unit'] in ['kg', 'N/m', 'N·s/m', 'km/h']: val = int(val)
            lbl.set_text(f"{p['name']}: {val} {p['unit']}")
            
    def apply_to_sim(self):
        self.sim.m = self.sliders['m'].get_current_value()
        self.sim.k = self.sliders['k'].get_current_value()
        self.sim.c = self.sliders['c'].get_current_value()
        self.sim.h = self.sliders['h'].get_current_value() / 100.0  # de cm a metros
        self.sim.v_kmh = self.sliders['v'].get_current_value()
        
    def sync_sliders(self):
        """Sincroniza los sliders con los valores actuales de la simulación."""
        self.sliders['m'].set_current_value(self.sim.m)
        self.sliders['k'].set_current_value(self.sim.k)
        self.sliders['c'].set_current_value(self.sim.c)
        self.sliders['h'].set_current_value(self.sim.h * 100)
        self.sliders['v'].set_current_value(self.sim.v_kmh)
        self.update_labels()

    def apply_gesture(self, param, delta):
        """
        Aplica un cambio de gesto al parámetro indicado.
        delta: -1.0 a +1.0 (velocidad relativa de cambio)
        """
        if param is None or abs(delta) < 0.02:
            return

        # Velocidad de cambio: porcentaje del rango por frame
        SPEED = 0.012   # 1.2% del rango por frame a delta=1.0

        ranges = {
            'm': (500,   2000),
            'k': (5000,  50000),
            'c': (100,   25000),
            'h': (2,     20),
            'v': (10,    100),
        }
        if param not in ranges:
            return

        lo, hi = ranges[param]
        span   = hi - lo

        # Obtener valor actual
        current_vals = {
            'm': self.sim.m,
            'k': self.sim.k,
            'c': self.sim.c,
            'h': self.sim.h * 100,
            'v': self.sim.v_kmh,
        }
        cur = current_vals[param]
        new = cur + delta * span * SPEED
        new = max(lo, min(hi, new))

        # Aplicar al sim
        if param == 'm':   self.sim.m     = new
        elif param == 'k': self.sim.k     = new
        elif param == 'c': self.sim.c     = new
        elif param == 'h': self.sim.h     = new / 100.0
        elif param == 'v': self.sim.v_kmh = new

        # Sincronizar slider visualmente
        self.sliders[param].set_current_value(new)
        self.update_labels()

    def process_event(self, event):
        if event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
            self.update_labels()
            self.apply_to_sim()
            # Sin reset: parámetros en caliente
            
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.btn_iniciar:
                self.sim.running = True
            elif event.ui_element == self.btn_pausar:
                self.sim.running = False
            elif event.ui_element == self.btn_reiniciar:
                self.sim.reset()
                self.apply_to_sim()
                self.sim.running = True
            elif event.ui_element == self.btn_add_bump:
                self.sim.add_bump()
                
    def draw(self, surface):
        pygame.draw.rect(surface, (40, 40, 45), self.rect)
        pygame.draw.line(surface, (100, 100, 100), (self.rect.x, self.rect.y), (self.rect.x, self.rect.bottom), 2)


def draw_gesture_hud(surface, gesture_ctrl, panel_rect, font_small, font_tiny, active_param):
    """
    Dibuja en la esquina inferior izquierda:
      - Miniatura de la cámara con el esqueleto de manos
      - Estado del gesto actual
      - Guía rápida de gestos
    """
    if gesture_ctrl is None:
        return

    CAM_W, CAM_H = 330, 227
    HUD_X = 950
    HUD_Y = 490

    # ── Miniatura de cámara ──────────────────────────────────────
    frame = gesture_ctrl.get_frame()
    if frame is not None:
        try:
            cam_surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            surface.blit(cam_surf, (HUD_X, HUD_Y))
            pygame.draw.rect(surface, (80, 200, 140),
                             (HUD_X, HUD_Y, CAM_W, CAM_H), 2)
        except Exception:
            pass
    else:
        pygame.draw.rect(surface, (30, 35, 45), (HUD_X, HUD_Y, CAM_W, CAM_H))
        txt = font_small.render("Sin cámara", True, (50, 50, 50))
        surface.blit(txt, (HUD_X + 70, HUD_Y + 55))

    # ── Estado del gesto ──────────────────────────────────────────
    status, lf, rf = gesture_ctrl.get_status()
    base_y = HUD_Y + CAM_H + 4

    PARAM_COLORS = {
        'c': (255, 200,  60),
        'k': (0,   200, 160),
        'm': (100, 180, 255),
        'v': (255, 120,  80),
        'h': (180,  80, 255),
        None:(120, 120, 130),
    }
    col = PARAM_COLORS.get(active_param, (120, 120, 130))

    # Fondo semitransparente
    """bg = pygame.Surface((CAM_W, 100), pygame.SRCALPHA)
    bg.fill((15, 18, 28, 200))
    surface.blit(bg, (HUD_X, base_y))

    PARAM_NAMES = {
        'c': 'Amortiguador (c)',
        'k': 'Resorte (k)',
        'm': 'Masa (m)',
        'v': 'Velocidad',
        'h': 'Prof. Bache',
        None: '— sin selección —',
    }
    pname = PARAM_NAMES.get(active_param, '—')
    lbl = font_small.render(f"▶  {pname}", True, col)
    surface.blit(lbl, (HUD_X + 6, base_y + 4))

    # Indicador de dedos
    lf_txt = font_tiny.render(f"IZQ: {lf} dedos", True, (160, 200, 160))
    rf_txt = font_tiny.render(f"DER: {rf} dedos", True, (160, 180, 220))
    surface.blit(lf_txt, (HUD_X + 6,       base_y + 22))
    surface.blit(rf_txt, (HUD_X + CAM_W//2, base_y + 22))

    # Mini guía
    guide = [
        "IZQ 1→c  2→k  3→m  4→v  5→h",
        "DER arriba/abajo = cambiar valor",
        "DER puño cerrado = 🚧 BACHE",
    ]
    for i, line in enumerate(guide):
        g = font_tiny.render(line, True, (90, 95, 115))
        surface.blit(g, (HUD_X + 4, base_y + 38 + i * 16))"""


def main():
    pygame.init()
    pygame.display.set_caption("Simulación de Suspensión (Masa-Resorte-Amortiguador)")
    window_surface = pygame.display.set_mode((1280, 720))
    

    manager = pygame_gui.UIManager((1280, 720))
    clock   = pygame.time.Clock()

    graph = OscillationGraph((0, 500, 950, 220))
    sim   = CarSimulation((0, 0, 950, 600), graph)
    panel = ControlPanel((950, 0, 330, 720), manager, sim)

    panel.update_labels()
    panel.apply_to_sim()

    # ── Fuentes para el HUD de gestos ──
    pygame.font.init()
    font_small = pygame.font.SysFont("Arial", 12, bold=True)
    font_tiny  = pygame.font.SysFont("Arial", 10)

    # ── Iniciar controlador de gestos ──
    gesture_ctrl = None
    gesture_enabled = False

    if MEDIAPIPE_OK and GestureController is not None:
        try:
            gesture_ctrl    = GestureController(camera_index=1, show_window=False)
            gesture_enabled = True
            print("[GESTOS] Controlador iniciado correctamente.")
        except Exception as e:
            print(f"[GESTOS] No se pudo iniciar: {e}")
    else:
        print("[GESTOS] mediapipe/opencv no disponible. Instala con:")
        print("         pip install mediapipe opencv-python")

    active_param = None   # parámetro actualmente seleccionado por gesto

    running = True
    while running:
        time_delta = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            sim.handle_debug_mouse(event)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F1:
                    sim.debug_mode = not sim.debug_mode
                    if sim.debug_mode:
                        sim.running = False
                        print('[DEBUG] Modo debug ACTIVADO')
                    else:
                        print('[DEBUG] Modo debug DESACTIVADO')

                # G = toggle gestos on/off
                if event.key == pygame.K_g:
                    gesture_enabled = not gesture_enabled
                    estado = "ACTIVADO" if gesture_enabled else "PAUSADO"
                    print(f"[GESTOS] {estado}")

                if sim.debug_mode:
                    shift = pygame.key.get_mods() & pygame.KMOD_SHIFT
                    step  = 0.5 if shift else 2.0
                    wstep = 0.005 if shift else 0.02
                    if event.key == pygame.K_LEFT:
                        sim.body_offset_x -= step
                    elif event.key == pygame.K_RIGHT:
                        sim.body_offset_x += step
                    elif event.key == pygame.K_UP:
                        sim.body_offset_y -= step
                    elif event.key == pygame.K_DOWN:
                        sim.body_offset_y += step
                    elif event.key in (pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS):
                        sim.wheel_dist += wstep
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        sim.wheel_dist = max(0.1, sim.wheel_dist - wstep)
                    elif event.key == pygame.K_s:
                        sim._save_config()
                    elif event.key == pygame.K_r:
                        sim.fw_offset_x = sim.fw_offset_y = 0.0
                        sim.rw_offset_x = sim.rw_offset_y = 0.0
                        sim.body_offset_x = -5.25
                        sim.body_offset_y = 30
                        print('[DEBUG] Offsets reseteados')

            manager.process_events(event)
            panel.process_event(event)

        manager.update(time_delta)

        # ── Aplicar gestos al sim ──────────────────────────────────────────
        if gesture_enabled and gesture_ctrl is not None:
            param, delta, bump = gesture_ctrl.get_state()
            active_param = param

            if param is not None and abs(delta) > 0.02:
                panel.apply_gesture(param, delta)

            if bump:
                sim.add_bump()
                print(f"[GESTOS] ¡Bache lanzado! t={sim.time:.2f}s")

        # ── Física ──────────────────────────────────────────────────────────
        if not sim.debug_mode:
            sim.update_physics(0.016)

        # ── Dibujo ──────────────────────────────────────────────────────────
        window_surface.fill((0, 0, 0))
        sim.draw(window_surface)
        graph.draw(window_surface)
        panel.draw(window_surface)
        manager.draw_ui(window_surface)

        # HUD de gestos (solo si está habilitado)
        if gesture_ctrl is not None:
            draw_gesture_hud(
                window_surface, gesture_ctrl,
                pygame.Rect(0, 450, 650, 250),
                font_small, font_tiny, active_param
            )

            # Indicador G = gestos ON/OFF
            g_col = (0, 220, 140) if gesture_enabled else (180, 60, 60)
            g_lbl = font_small.render(
                f"[G] Gestos: {'ON ✓' if gesture_enabled else 'OFF'}",
                True, g_col)
            window_surface.blit(g_lbl, (8, 8))

        pygame.display.update()

    # ── Limpieza ────────────────────────────────────────────────────────────
    if gesture_ctrl is not None:
        gesture_ctrl.stop()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
