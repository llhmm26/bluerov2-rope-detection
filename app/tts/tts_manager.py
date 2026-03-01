import pyttsx3
import time
import threading
import queue

ANNOUNCE_INTERVAL = 1.0  # seconds


class TTSManager:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 165)
        self.engine.setProperty("volume", 1.0)

        self.rope_active = False
        self.obstacle_active = False

        self.last_announce = {
            "rope": 0.0,
            "obstacle": 0.0
        }

        self.lock = threading.Lock()

        # ---- NEW: speech queue + single worker thread ----
        self.queue = queue.Queue()
        self.worker = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        """
        Dedicated single thread that handles all TTS.
        Prevents pyttsx3 concurrency crashes.
        """
        while True:
            text = self.queue.get()
            if text is None:
                break

            try:
                self.engine.say(text)
                self.engine.runAndWait()
                print("[TTS] Speaking:", text)
            except Exception as e:
                print("[TTS] Error:", e)

    def _can_announce(self, event_type):
        now = time.time()
        last_time = self.last_announce.get(event_type, 0)

        if now - last_time <  ANNOUNCE_INTERVAL:
            return False
        
        self.last_announce[event_type] = now
        return True

    # ---------------- PUBLIC API ----------------

    def announce(self, event_type, text):
        print("[TTS] DIRECT SPEAK:", text)
        self.engine.say(text)
        self.engine.runAndWait()



    def update(self, summary: dict):

        rope_detected = summary.get("rope_detected", False)
        obstacle_detected = summary.get("obstacle_count", 0) > 0

        # ---- Rope state transition ----
        if rope_detected and not self.rope_active:
            print("[TTS] Rope transition → speaking")
            self.engine.say("Rope detected ahead.")
            self.engine.runAndWait()

        self.rope_active = rope_detected

        # ---- Obstacle state transition ----
        if obstacle_detected and not self.obstacle_active:
            print("[TTS] Obstacle transition → speaking")
            self.engine.say("Obstacle ahead. Proceed with caution.")
            self.engine.runAndWait()

        self.obstacle_active = obstacle_detected