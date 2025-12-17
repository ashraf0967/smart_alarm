import flet as ft
import datetime
import time
import threading
import cv2
import base64
import numpy as np
from PIL import Image
import io
import asyncio
from face_detection import SmileDetector, CameraManager
from alarm_sounds import AlarmSoundManager
import os
import json

class SmartAlarmApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "👋 Smart Smile Alarm"
        
        # Window Settings
        self.page.window.width = 400
        self.page.window.height = 800
        self.page.theme_mode = ft.ThemeMode.DARK
        self.page.bgcolor = ft.colors.BLACK
        self.page.padding = 10
        self.page.scroll = ft.ScrollMode.AUTO
        
        # Components
        self.smile_detector = SmileDetector()
        self.camera_manager = CameraManager()
        self.alarm_manager = AlarmSoundManager()
        
        # Audio Setup
        default_sound = self.alarm_manager.get_current_sound_path()
        if not default_sound or not os.path.exists(default_sound):
             default_sound = "assets/sounds/urgent.wav"
        
        # Determine strict path (Flet assets logic)
        sound_filename = os.path.basename(default_sound)
        initial_src = f"/sounds/{sound_filename}"
        
        self.audio_player = ft.Audio(src=initial_src, autoplay=False, volume=self.alarm_manager.volume)
        self.page.overlay.append(self.audio_player)
        
        # State
        self.alarms = [] 
        self.active_alarm_id = None 
        self.is_alarm_ringing = False
        self.bedside_mode_active = False
        self.stopping_alarm_lock = False
        
        self.smile_detected = False
        self.smile_progress = 0.0
        self.current_smile_status = False
        
        # State for Threads
        self.monitor_thread_running = True
        
        # Load Data
        self.load_alarms()
        self.load_assets()
        
        # UI Setup
        self.setup_ui()
        
        # Check First Run (Permission Request)
        self.check_first_run()

        # Start Background Threads
        self.start_clock_thread()
        self.start_alarm_monitor_thread()
    
    def load_alarms(self):
        try:
            if os.path.exists("alarms.json"):
                with open("alarms.json", "r") as f:
                    data = json.load(f)
                    self.alarms = []
                    now = datetime.datetime.now()
                    for item in data:
                        h, m = map(int, item["time_str"].split(":"))
                        alarm_dt = now.replace(hour=h, minute=m, second=0, microsecond=0)
                        if alarm_dt < now:
                             alarm_dt = alarm_dt.replace(day=now.day + 1)
                        
                        self.alarms.append({
                            "id": item["id"],
                            "time": alarm_dt,
                            "active": item["active"]
                        })
        except Exception as e:
            print(f"Error loading alarms: {e}")
            self.alarms = []

    def save_alarms(self):
        data = []
        for alarm in self.alarms:
            data.append({
                "id": alarm["id"],
                "time_str": alarm["time"].strftime("%H:%M"),
                "active": alarm["active"]
            })
        try:
            with open("alarms.json", "w") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Error saving alarms: {e}")

    def load_assets(self):
        self.icons = {
            "alarm": ft.Icon(ft.icons.ALARM, color=ft.colors.ORANGE_400),
            "camera": ft.Icon(ft.icons.CAMERA_ALT, color=ft.colors.GREEN_400),
            "delete": ft.Icon(ft.icons.DELETE, color=ft.colors.RED_400),
            "add": ft.Icon(ft.icons.ADD, color=ft.colors.GREEN_400),
        }

    def check_first_run(self):
        if not os.path.exists("setup_complete.flag"):
            def grant_permission(e):
                self.page.close_dialog()
                self.show_snackbar("Requesting Camera Permission...")
                # Try to open camera to force permission prompt
                try:
                    success = self.camera_manager.start_camera()
                    time.sleep(1)
                    self.camera_manager.stop_camera()
                except:
                    success = False
                
                if success:
                    self.show_snackbar("Camera Access Granted!")
                    with open("setup_complete.flag", "w") as f:
                        f.write("done")
                else:
                    self.show_snackbar("Camera access checked.")

            dlg = ft.AlertDialog(
                title=ft.Text("Welcome to Smart Smile Alarm"),
                content=ft.Column([
                    ft.Text("To wake you up, we need access to your camera!"),
                    ft.Text("We check for your smile to turn off the alarm.", size=12, color=ft.colors.GREY_400),
                    ft.Text("Please click 'Allow' when the system asks.", weight=ft.FontWeight.BOLD),
                ], tight=True),
                actions=[
                    ft.TextButton("Grant Permission", on_click=grant_permission),
                ],
                modal=True,
            )
            self.page.dialog = dlg
            dlg.open = True
            try: self.page.update()
            except: pass

    def setup_ui(self):
        # Header
        self.header = ft.Container(
            content=ft.Row([
                ft.Icon(ft.icons.ALARM, size=30, color=ft.colors.ORANGE_400),
                ft.Text("Smart Smile Alarm", size=24, weight=ft.FontWeight.BOLD),
            ], alignment=ft.MainAxisAlignment.CENTER),
            padding=10,
            bgcolor=ft.colors.BLUE_GREY_900,
            border_radius=10,
        )
        
        # Current Time
        self.current_time_display = ft.Text("00:00:00", size=40, weight=ft.FontWeight.BOLD, color=ft.colors.CYAN_300)
        
        # Alarms List
        self.alarms_list_view = ft.Column(spacing=10)
        self.update_alarms_list()
        
        # Time Picker
        self.time_picker = ft.TimePicker(
            confirm_text="Add Alarm",
            on_change=self.add_new_alarm
        )
        self.page.overlay.append(self.time_picker)
        
        # Add Button
        self.add_alarm_btn = ft.ElevatedButton(
            "Add Alarm",
            icon=ft.icons.ADD,
            on_click=lambda e: self.page.open(self.time_picker),
            bgcolor=ft.colors.BLUE_700,
            color=ft.colors.WHITE
        )
        
        # Bedside Toggle
        self.bedside_btn = ft.FloatingActionButton(
            text="Start Bedside Mode",
            icon=ft.icons.NIGHTLIGHT_ROUND,
            bgcolor=ft.colors.PURPLE_700,
            on_click=lambda e: self.activate_bedside_mode(),
            width=200
        )
        
        # Camera Preview Box
        self.camera_display = ft.Image(
            src_base64=self.get_placeholder_image(),
            width=300,
            height=225,
            fit=ft.ImageFit.CONTAIN,
            border_radius=10,
        )
        self.smile_status = ft.Text("No smile detected", color=ft.colors.GREY_400)
        self.smile_progress_bar = ft.ProgressBar(value=0, color=ft.colors.GREEN, bgcolor=ft.colors.GREY_800)

        # Sound Selection
        self.setup_sound_selection()

        # Main Layout
        self.main_column = ft.Column([
            self.header,
            ft.Container(height=10),
            ft.Container(
                content=ft.Column([
                    ft.Text("Current Time", color=ft.colors.GREY_400),
                    self.current_time_display,
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                alignment=ft.alignment.center
            ),
            ft.Divider(),
            ft.Text("Your Alarms (Max 3)", size=16, weight=ft.FontWeight.BOLD),
            self.alarms_list_view,
            ft.Container(height=10),
            self.add_alarm_btn,
            ft.Divider(),
            self.sound_selection_container,
            ft.Divider(),
            ft.Text("Test Camera:", color=ft.colors.GREY_400),
            ft.Container(content=self.camera_display, alignment=ft.alignment.center),
            self.smile_status,
            ft.Container(height=20),
            ft.Container(content=self.bedside_btn, alignment=ft.alignment.center, padding=20)
        ], scroll=ft.ScrollMode.AUTO, expand=True)
        
        self.page.add(self.main_column)

    def update_alarms_list(self):
        self.alarms_list_view.controls.clear()
        
        for alarm in self.alarms:
            time_str = alarm["time"].strftime("%H:%M")
            switch = ft.Switch(
                value=alarm["active"],
                on_change=lambda e, a=alarm: self.toggle_alarm_active(a, e.control.value)
            )
            
            card = ft.Container(
                content=ft.Row([
                    ft.Column([
                        ft.Text(time_str, size=24, weight=ft.FontWeight.BOLD),
                        ft.Text("Once" if alarm["active"] else "Off", color=ft.colors.GREY_400),
                    ]),
                    ft.Row([
                        switch,
                        ft.IconButton(ft.icons.DELETE, icon_color=ft.colors.RED_400, 
                                    on_click=lambda e, a=alarm: self.delete_alarm(a))
                    ])
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                padding=15,
                bgcolor=ft.colors.BLUE_GREY_800,
                border_radius=10
            )
            self.alarms_list_view.controls.append(card)
        
        if hasattr(self, 'add_alarm_btn'):
            self.add_alarm_btn.visible = len(self.alarms) < 3
            
        try: self.page.update()
        except: pass

    def add_new_alarm(self, e):
        if self.time_picker.value and len(self.alarms) < 3:
            now = datetime.datetime.now()
            alarm_time = now.replace(
                hour=self.time_picker.value.hour, 
                minute=self.time_picker.value.minute,
                second=0, microsecond=0
            )
            if alarm_time < now:
                alarm_time = alarm_time.replace(day=now.day + 1)
            
            new_alarm = {
                "id": str(time.time()),
                "time": alarm_time,
                "active": True
            }
            self.alarms.append(new_alarm)
            self.save_alarms()
            self.update_alarms_list()
            self.show_snackbar(f"Alarm set for {alarm_time.strftime('%H:%M')}")

    def delete_alarm(self, alarm):
        self.alarms.remove(alarm)
        self.save_alarms()
        self.update_alarms_list()

    def toggle_alarm_active(self, alarm, active):
        alarm["active"] = active
        if active:
            now = datetime.datetime.now()
            if alarm["time"] < now:
                alarm["time"] = alarm["time"].replace(day=now.day + 1)
        
        self.save_alarms()
        self.update_alarms_list()

    def activate_bedside_mode(self):
        active_alarms = [a for a in self.alarms if a["active"]]
        if not active_alarms:
            self.show_snackbar("No active alarms! Please enable an alarm first.")
            return

        self.bedside_mode_active = True
        
        active_alarms.sort(key=lambda x: x["time"])
        next_alarm = active_alarms[0]["time"].strftime("%H:%M")
        
        self.bedside_time = ft.Text(
             datetime.datetime.now().strftime("%H:%M"),
             size=100, weight=ft.FontWeight.BOLD, color=ft.colors.GREY_800
        )
        
        view = ft.Container(
            content=ft.Column([
                ft.Text("Bedside Mode", color=ft.colors.GREY_700),
                ft.Container(height=40),
                self.bedside_time,
                ft.Text(f"Next Alarm: {next_alarm}", size=20, color=ft.colors.ORANGE_400),
                ft.Container(height=80),
                ft.ElevatedButton("Exit", on_click=lambda e: self.exit_bedside_mode(), bgcolor=ft.colors.GREY_900)
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, alignment=ft.MainAxisAlignment.CENTER),
            bgcolor=ft.colors.BLACK,
            alignment=ft.alignment.center,
            expand=True
        )
        
        self.page.clean()
        self.page.add(view)
        self.page.update()

    def get_placeholder_image(self):
        try:
            img = Image.new('RGB', (400, 300), color=(40, 40, 40))
            # Draw text
            # from PIL import ImageDraw
            # d = ImageDraw.Draw(img)
            # d.text((10,10), "Camera Off", fill=(255,255,255))
            
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode()
        except Exception as e:
            print(f"Error generating placeholder: {e}")
            # Return a 1x1 black pixel base64
            return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="

    def exit_bedside_mode(self):
        self.bedside_mode_active = False
        self.page.clean()
        
        # Reset camera placeholder explicitly
        self.camera_display.src_base64 = self.get_placeholder_image()
        self.camera_display.update()
        
        self.setup_ui()
        self.page.add(self.main_column)
        self.page.update()

    def trigger_alarm(self, alarm_id):
        self.is_alarm_ringing = True
        self.active_alarm_id = alarm_id
        self.bedside_mode_active = False
        
        sound_path = self.alarm_manager.get_current_sound_path()
        if not sound_path or not os.path.exists(sound_path):
             sound_path = "assets/sounds/urgent.wav"
        
        relative_path = f"/sounds/{os.path.basename(sound_path)}"
        self.audio_player.src = relative_path
        self.audio_player.update()
        self.audio_player.play()
        
        self.camera_manager.start_camera()
        self.show_alarm_ui()
        
        threading.Thread(target=self.camera_feed_loop, daemon=True).start()
        threading.Thread(target=self.smile_monitor_loop, daemon=True).start()

    def show_alarm_ui(self):
        self.alarm_ui = ft.Container(
            content=ft.Column([
                ft.Text("⏰ WAKE UP! ⏰", size=40, color=ft.colors.RED_400, weight=ft.FontWeight.BOLD),
                ft.Text("Smile to stop!", size=24, color=ft.colors.YELLOW_300),
                ft.Container(content=self.camera_display, padding=10, border=ft.border.all(2, ft.colors.RED_400), border_radius=10),
                self.smile_progress_bar,
                self.smile_status
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, alignment=ft.MainAxisAlignment.CENTER),
            bgcolor=ft.colors.BLACK,
            expand=True,
            alignment=ft.alignment.center
        )
        self.page.clean()
        self.page.add(self.alarm_ui)
        self.page.update()

    def camera_feed_loop(self):
        last_id = None
        while self.is_alarm_ringing:
            frame = self.camera_manager.get_frame()
            if frame is not None and id(frame) != last_id:
                last_id = id(frame)
                try:
                    small = cv2.resize(frame, (320, 240))
                    is_smiling, ratio, annotated = self.smile_detector.detect_smile(small)
                    
                    if annotated is not None:
                        _, buf = cv2.imencode('.jpg', annotated)
                        b64 = base64.b64encode(buf).decode()
                        self.camera_display.src_base64 = b64
                        
                        self.smile_progress = min(ratio / self.smile_detector.smile_threshold, 1.0)
                        self.smile_progress_bar.value = self.smile_progress
                        self.current_smile_status = is_smiling
                        
                        if is_smiling:
                            self.smile_status.value = "SMILING! 😊"
                            self.smile_status.color = ft.colors.GREEN_400
                        else:
                             self.smile_status.value = f"Ratio: {ratio:.2f}"
                             self.smile_status.color = ft.colors.YELLOW_300
                        
                        try: self.page.update()
                        except: pass
                except: pass
            time.sleep(0.05)

    def smile_monitor_loop(self):
        start_smile = None
        last_play = time.time()
        
        while self.is_alarm_ringing:
            # Loop Audio
            if time.time() - last_play > 2.0:
                 self.audio_player.seek(0)
                 self.audio_player.play()
                 last_play = time.time()
            
            if self.current_smile_status:
                if start_smile is None: start_smile = time.time()
                elif time.time() - start_smile > 2.0:
                    self.stop_alarm()
                    break
            else:
                start_smile = None
            time.sleep(0.1)

    def stop_alarm(self):
        if self.stopping_alarm_lock: return
        self.stopping_alarm_lock = True
        
        try:
            self.is_alarm_ringing = False
            self.audio_player.pause()
            self.camera_manager.stop_camera()
            
            # Deactivate
            for alarm in self.alarms:
                if alarm["id"] == self.active_alarm_id:
                    alarm["active"] = False
            self.save_alarms()
            
            try:
                self.page.clean()
                self.page.add(ft.Container(
                    content=ft.Column([
                        ft.Icon(ft.icons.CELEBRATION, size=80, color=ft.colors.GREEN),
                        ft.Text("Good Morning!", size=30),
                    ], alignment=ft.MainAxisAlignment.CENTER),
                    expand=True, alignment=ft.alignment.center
                ))
                self.page.update()
            except: pass

            time.sleep(3)
            self.exit_bedside_mode() 
            
        except Exception as e:
            print(f"Error in stop_alarm: {e}")
        finally:
            self.stopping_alarm_lock = False

    def start_clock_thread(self):
        def loop():
            while self.monitor_thread_running:
                now = datetime.datetime.now()
                try:
                    t_str = now.strftime("%H:%M:%S")
                    if self.bedside_mode_active and hasattr(self, 'bedside_time'):
                        self.bedside_time.value = t_str
                    self.current_time_display.value = t_str
                    self.page.update()
                except: pass
                time.sleep(1)
        threading.Thread(target=loop, daemon=True).start()

    def start_alarm_monitor_thread(self):
        def loop():
            while self.monitor_thread_running:
                if not self.is_alarm_ringing:
                     now = datetime.datetime.now()
                     for alarm in self.alarms:
                         if alarm["active"] and now >= alarm["time"] and now < alarm["time"] + datetime.timedelta(minutes=1):
                             self.trigger_alarm(alarm["id"])
                             time.sleep(60) # Prevent multiple triggers
                time.sleep(1)
        threading.Thread(target=loop, daemon=True).start()

    def show_snackbar(self, message):
        self.page.overlay.append(ft.SnackBar(ft.Text(message), open=True))
        try: self.page.update()
        except: pass

    def setup_sound_selection(self):
        self.sound_options = []
        available_sounds = self.alarm_manager.get_available_sounds()
        for sound in available_sounds:
            self.sound_options.append(ft.Container(
                content=ft.Column([
                    ft.Icon(sound["icon"], size=20, color=ft.colors.BLUE_400),
                    ft.Text(sound["name"], size=10, overflow=ft.TextOverflow.ELLIPSIS),
                    ft.ElevatedButton("Test", height=20, style=ft.ButtonStyle(padding=0), 
                                    on_click=lambda e, s=sound["id"]: self.test_sound(s))
                ], alignment=ft.MainAxisAlignment.CENTER, spacing=2),
                padding=5,
                width=80, height=90,
                bgcolor=ft.colors.BLUE_GREY_800,
                border=ft.border.all(2, ft.colors.BLUE if sound["id"] == self.alarm_manager.current_sound else ft.colors.TRANSPARENT),
                border_radius=5,
                on_click=lambda e, s=sound["id"]: self.select_sound(s),
                data=sound["id"]
            ))
        self.sound_selection_container = ft.Column([
            ft.Text("Alarm Sound", weight=ft.FontWeight.BOLD),
            ft.Row(self.sound_options, scroll=ft.ScrollMode.HIDDEN)
        ])

    def select_sound(self, sound_id):
        self.alarm_manager.set_sound(sound_id)
        for op in self.sound_options:
            op.border = ft.border.all(2, ft.colors.BLUE if op.data == sound_id else ft.colors.TRANSPARENT)
        try: self.page.update()
        except: pass

    def test_sound(self, sound_id):
        self.audio_player.pause()
        sound_path = self.alarm_manager.generated_files[sound_id]
        relative_path = f"/sounds/{os.path.basename(sound_path)}"
        self.audio_player.src = relative_path
        self.audio_player.update()
        self.audio_player.play()

def main(page: ft.Page):
    app = SmartAlarmApp(page)

if __name__ == "__main__":
    ft.app(target=main, assets_dir="assets", view=ft.AppView.FLET_APP)