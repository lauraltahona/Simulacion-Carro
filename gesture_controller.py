# gesture_controller.py
"""
Controlador de gestos con MediaPipe.
Compatible con mediapipe >= 0.9 y >= 0.10 (detecta la versión automáticamente).

INSTALAR:
    pip install mediapipe==0.10.21 opencv-python
    (La versión 0.10.21 restaura mp.solutions en Windows)

GESTOS:
────────────────────────────────────────────────────────
  MANO IZQUIERDA  →  selecciona qué parámetro cambiar:
    ☝  1 dedo  →  Amortiguador (c)
    ✌  2 dedos →  Resorte (k)
       3 dedos →  Masa (m)
       4 dedos →  Velocidad (v)
    ✋  5 dedos →  Prof. Bache (h)

  MANO DERECHA  →  cambia el valor:
    Sube la mano  ▲  →  sube el parámetro
    Baja la mano  ▼  →  baja el parámetro
    Puño cerrado  👊 →  lanza un BACHE

  TECLADO  G  →  activa / pausa gestos
────────────────────────────────────────────────────────
"""

import threading
import time

# ── Importaciones opcionales ─────────────────────────────────────────────────
try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

MEDIAPIPE_OK  = False
MP_USE_LEGACY = False   # True = mp.solutions  |  False = mp.tasks

try:
    import mediapipe as mp

    # Intentar API antigua (0.9.x / algunos builds de 0.10.x)
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands'):
        MP_USE_LEGACY = True
        MEDIAPIPE_OK  = True
    else:
        # API nueva (0.10+): necesita el modelo .task descargado
        from mediapipe.tasks.python import vision as _mp_vision
        from mediapipe.tasks.python.core import base_options as _mp_base
        MEDIAPIPE_OK = True

except ImportError:
    pass

# ── Índices de landmarks ─────────────────────────────────────────────────────
FINGER_TIPS = [8, 12, 16, 20]
FINGER_PIPS = [6, 10, 14, 18]
THUMB_TIP   = 4
THUMB_IP    = 3


def _count_fingers_legacy(hand_landmarks, handedness_label):
    lm    = hand_landmarks.landmark
    count = 0
    if handedness_label == "Right":
        if lm[THUMB_TIP].x < lm[THUMB_IP].x:
            count += 1
    else:
        if lm[THUMB_TIP].x > lm[THUMB_IP].x:
            count += 1
    for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):
        if lm[tip].y < lm[pip].y:
            count += 1
    return count


def _count_fingers_new(hand_landmarks):
    """Para la nueva API: hand_landmarks es una lista de NormalizedLandmark."""
    lm    = hand_landmarks
    count = 0
    # Pulgar: comparar X
    if lm[THUMB_TIP].x < lm[THUMB_IP].x:
        count += 1
    for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):
        if lm[tip].y < lm[pip].y:
            count += 1
    return count


def _hand_center_y_legacy(hand_landmarks):
    lm = hand_landmarks.landmark
    return sum(lm[i].y for i in [0, 5, 9, 13, 17]) / 5


def _hand_center_y_new(hand_landmarks):
    lm = hand_landmarks
    return sum(lm[i].y for i in [0, 5, 9, 13, 17]) / 5


# ── Clase principal ──────────────────────────────────────────────────────────
class GestureController:
    PARAM_MAP   = {1: 'c', 2: 'k', 3: 'm', 4: 'v', 5: 'h'}
    PARAM_NAMES = {
        'c': 'Amortiguador (c)',
        'k': 'Resorte (k)',
        'm': 'Masa (m)',
        'v': 'Velocidad',
        'h': 'Prof. Bache',
    }

    def __init__(self, camera_index=0, show_window=False):
        self.camera_index = camera_index
        self.show_window  = show_window

        self.selected_param = None
        self.delta          = 0.0
        self.bump_triggered = False
        self.active         = False
        self.frame_rgb      = None
        self.status_text    = "Iniciando cámara..."
        self.left_fingers   = 0
        self.right_fingers  = 0

        self._lock        = threading.Lock()
        self._stop_event  = threading.Event()
        self._fist_last   = False

        if MEDIAPIPE_OK and CV2_OK:
            target = self._run_legacy if MP_USE_LEGACY else self._run_new
            self._thread = threading.Thread(target=target, daemon=True)
            self._thread.start()
        else:
            if not CV2_OK:
                self.status_text = "⚠ opencv-python no instalado.  pip install opencv-python"
            else:
                self.status_text = "⚠ mediapipe no instalado.  pip install mediapipe==0.10.21 opencv-python"

    def stop(self):
        self._stop_event.set()

    # ── Getters thread-safe ──────────────────────────────────────────────────
    def get_state(self):
        with self._lock:
            bump = self.bump_triggered
            self.bump_triggered = False
            return self.selected_param, self.delta, bump

    def get_frame(self):
        with self._lock:
            return self.frame_rgb

    def get_status(self):
        with self._lock:
            return self.status_text, self.left_fingers, self.right_fingers

    # ── Utilidades internas ──────────────────────────────────────────────────
    def _process_hands(self, left_lm, right_lm, left_label, right_lm_list,
                       count_fn_left, count_fn_right,
                       center_y_fn_left, center_y_fn_right,
                       right_y_history):
        """Lógica común independiente de la API."""
        # Mano izquierda → seleccionar parámetro
        selected = None
        lf = 0
        if left_lm is not None:
            lf = count_fn_left(left_lm, left_label) if left_label else count_fn_left(left_lm)
            selected = self.PARAM_MAP.get(lf)

        # Mano derecha → delta o puño
        delta   = 0.0
        rf      = 0
        is_fist = False
        if right_lm is not None:
            rf      = count_fn_right(right_lm)
            is_fist = (rf == 0)
            if not is_fist:
                cy = center_y_fn_right(right_lm)
                right_y_history.append(cy)
                if len(right_y_history) > 6:
                    right_y_history.pop(0)
                if len(right_y_history) >= 2:
                    recent = sum(right_y_history[-2:]) / 2
                    old    = sum(right_y_history[:2])  / 2
                    delta  = max(-1.0, min(1.0, (old - recent) * 18.0))
            else:
                right_y_history.clear()

        # Borde de puño
        bump = is_fist and not self._fist_last
        self._fist_last = is_fist

        return selected, lf, rf, delta, bump

    def _save_frame(self, frame, selected, lf, rf, delta, is_fist):
        """Guarda frame redimensionado y actualiza estado compartido."""
        import cv2 as _cv2
        h_fr, w_fr = frame.shape[:2]

        # Overlay
        _cv2.rectangle(frame, (0, 0), (w_fr, 58), (15, 18, 28), -1)
        pname = self.PARAM_NAMES.get(selected, "— sin seleccion —")
        action = "BACHE! 👊" if is_fist else (
            f"subiendo ▲ {delta:.2f}" if delta > 0.05 else (
            f"bajando  ▼ {abs(delta):.2f}" if delta < -0.05 else ""))

        _cv2.putText(frame, f"IZQ {lf} -> {pname}",
                     (8, 20), _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 150), 1)
        _cv2.putText(frame, f"DER {rf}  {action}",
                     (8, 44), _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1)

        if is_fist:
            _cv2.putText(frame, "BACHE!",
                         (w_fr//2 - 50, h_fr//2),
                         _cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 60, 255), 3)

        small = _cv2.resize(frame, (400, 225))
        return _cv2.cvtColor(small, _cv2.COLOR_BGR2RGB)

    # ── Runner API LEGACY (mp.solutions) ────────────────────────────────────
    def _run_legacy(self):
        mp_hands   = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils

        cap = cv2.VideoCapture(self.camera_index)
        
        if not cap.isOpened():
            with self._lock:
                self.status_text = "⚠ No se pudo abrir la cámara (índice 0)"
            return

        with self._lock:
            self.active = True
            self.status_text = "Cámara activa (API legacy)"

        right_y_history = []

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.55,
        ) as hands:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                frame   = cv2.flip(frame, 1)
                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                left_lm = right_lm = None
                left_label = right_label = None

                if results.multi_hand_landmarks:
                    for lm, info in zip(results.multi_hand_landmarks,
                                        results.multi_handedness):
                        label = info.classification[0].label
                        mp_drawing.draw_landmarks(
                            frame, lm, mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec((0,200,150), 2, 3),
                            mp_drawing.DrawingSpec((0,120,80), 1))
                        if label == "Left":
                            left_lm, left_label = lm, label
                        else:
                            right_lm, right_label = lm, label

                selected, lf, rf, delta, bump = self._process_hands(
                    left_lm, right_lm, left_label, right_lm,
                    _count_fingers_legacy, lambda lm: _count_fingers_legacy(lm, "Right"),
                    _hand_center_y_legacy, _hand_center_y_legacy,
                    right_y_history,
                )

                frame_rgb = self._save_frame(frame, selected, lf, rf, delta, rf == 0 and right_lm is not None)

                with self._lock:
                    self.selected_param = selected
                    self.delta          = delta
                    if bump:
                        self.bump_triggered = True
                    self.left_fingers   = lf
                    self.right_fingers  = rf
                    self.frame_rgb      = frame_rgb
                    self.status_text    = f"IZQ={lf} → {self.PARAM_NAMES.get(selected,'—')}  |  DER={rf}"

                if self.show_window:
                    cv2.imshow("Gestos", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                time.sleep(0.016)

        cap.release()
        if self.show_window:
            cv2.destroyAllWindows()
        with self._lock:
            self.active = False
            self.status_text = "Cámara cerrada"

    # ── Runner API NUEVA (mp.tasks — requiere modelo .task) ──────────────────
    def _run_new(self):
        """
        La nueva API requiere descargar el modelo hand_landmarker.task.
        Se descarga automáticamente la primera vez.
        """
        import urllib.request, os, tempfile

        MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
                      "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        model_path = os.path.join(tempfile.gettempdir(), "hand_landmarker.task")

        if not os.path.exists(model_path):
            with self._lock:
                self.status_text = "Descargando modelo de manos... (solo la primera vez)"
            try:
                urllib.request.urlretrieve(MODEL_URL, model_path)
            except Exception as e:
                with self._lock:
                    self.status_text = f"⚠ Error descargando modelo: {e}"
                return

        from mediapipe.tasks.python       import vision as mp_vision
        from mediapipe.tasks.python.core  import base_options as mp_base
        from mediapipe                    import Image as MpImage, ImageFormat

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            with self._lock:
                self.status_text = "⚠ No se pudo abrir la cámara (índice 0)"
            return

        with self._lock:
            self.active = True
            self.status_text = "Cámara activa (API nueva)"

        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_base.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        right_y_history = []

        with mp_vision.HandLandmarker.create_from_options(opts) as landmarker:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                frame = cv2.flip(frame, 1)
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = MpImage(image_format=ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(mp_img)

                left_lm = right_lm = None

                if result.hand_landmarks:
                    for lm_list, handedness in zip(result.hand_landmarks,
                                                   result.handedness):
                        label = handedness[0].display_name  # "Left" / "Right"
                        # Dibujar puntos manualmente
                        h_fr, w_fr = frame.shape[:2]
                        for conn in mp_vision.HandLandmarksConnections.HAND_CONNECTIONS:
                            p1 = lm_list[conn.start]
                            p2 = lm_list[conn.end]
                            cv2.line(frame,
                                     (int(p1.x*w_fr), int(p1.y*h_fr)),
                                     (int(p2.x*w_fr), int(p2.y*h_fr)),
                                     (0, 180, 120), 2)
                        for lm in lm_list:
                            cv2.circle(frame, (int(lm.x*w_fr), int(lm.y*h_fr)),
                                       4, (0, 220, 150), -1)

                        if label == "Left":
                            left_lm = lm_list
                        else:
                            right_lm = lm_list

                # Contar dedos
                lf = _count_fingers_new(left_lm)  if left_lm  else 0
                rf = _count_fingers_new(right_lm) if right_lm else 0
                selected = self.PARAM_MAP.get(lf) if left_lm else None

                is_fist = (rf == 0 and right_lm is not None)
                delta   = 0.0
                if right_lm and not is_fist:
                    cy = _hand_center_y_new(right_lm)
                    right_y_history.append(cy)
                    if len(right_y_history) > 6:
                        right_y_history.pop(0)
                    if len(right_y_history) >= 2:
                        delta = max(-1.0, min(1.0,
                            (sum(right_y_history[:2])/2 - sum(right_y_history[-2:])/2) * 18.0))
                else:
                    right_y_history.clear()

                bump = is_fist and not self._fist_last
                self._fist_last = is_fist

                frame_rgb = self._save_frame(frame, selected, lf, rf, delta, is_fist)

                with self._lock:
                    self.selected_param = selected
                    self.delta          = delta
                    if bump:
                        self.bump_triggered = True
                    self.left_fingers   = lf
                    self.right_fingers  = rf
                    self.frame_rgb      = frame_rgb
                    self.status_text    = f"IZQ={lf} → {self.PARAM_NAMES.get(selected,'—')}  |  DER={rf}"

                if self.show_window:
                    cv2.imshow("Gestos", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                time.sleep(0.016)

        cap.release()
        if self.show_window:
            cv2.destroyAllWindows()
        with self._lock:
            self.active = False
            self.status_text = "Cámara cerrada"
