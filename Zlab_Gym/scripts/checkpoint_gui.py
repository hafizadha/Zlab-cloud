"""Launch saved Z01 policies in the native MuJoCo viewer."""

import os
from pathlib import Path
import subprocess
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import yaml


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT.parent / ".venv-soccer/bin/python"
TASK_PREFIX = "Z01-StdAddHead-"
EXPERIMENT_TASK = {
    "z01_walking": "Velocity-Walk",
    "z01_fast_walking": "Velocity-FastWalk",
    "z01_stride_walking": "Velocity-StrideWalk",
    "z01_human_run": "Velocity-Run",
    "z01_human_run_balanced": "Velocity-Run",
    "z01_soccer_kick": "Soccer-Kick",
    "z01_soccer_approach": "Soccer-Approach",
    "z01_soccer_sprint": "Soccer-Sprint",
    "z01_std_add_head_tracking": "Tracking",
    "z01_std_add_head_tracking_residual": "Tracking-Residual",
    "z01_lafan1_stand_run_stop": "Tracking-Residual",
}
TASKS = tuple(sorted(set(EXPERIMENT_TASK.values()) | {"Velocity-Flat", "Velocity-Rough"}))


def find_checkpoints(include_archive):
    log_root = Path(os.environ.get("ZLAB_LOG_ROOT", "logs")).expanduser()
    roots = [ROOT / log_root / "rsl_rl"]
    if include_archive:
        roots.extend((ROOT / "logs").glob("archive-*/rsl_rl"))
    checkpoints = [
        path for directory in roots for path in directory.glob("*/*/model_*.pt")
        if path.stem.removeprefix("model_").isdigit()
    ]
    return sorted(
        checkpoints,
        key=lambda path: (path.parent.name, int(path.stem.removeprefix("model_"))),
        reverse=True,
    )


def saved_motion_file(checkpoint):
    try:
        with (checkpoint.parent / "params/env.yaml").open() as stream:
            config = yaml.load(stream, Loader=yaml.BaseLoader)
        return config["commands"]["motion"]["motion_file"] or ""
    except (OSError, yaml.YAMLError, KeyError, TypeError):
        return ""


class CheckpointLauncher:
    def __init__(self, window):
        self.window = window
        self.process = None
        self.log_path = None
        self.checkpoints = {}
        self.archive = tk.BooleanVar(value=False)
        self.fast = tk.BooleanVar(value=True)
        self.num_envs = tk.StringVar(value="1")
        self.device = tk.StringVar(value="Auto")
        self.seed = tk.StringVar(value="2026")
        self.task = tk.StringVar()
        self.motion = tk.StringVar()
        self.status = tk.StringVar()
        window.title("Z01 MuJoCo Checkpoint Player")
        window.geometry("840x510")
        window.minsize(670, 400)

        panel = ttk.Frame(window, padding=16)
        panel.pack(fill="both", expand=True)
        ttk.Label(panel, text="Z01 Checkpoint Player", font=("TkDefaultFont", 16, "bold")).pack(anchor="w")
        ttk.Label(panel, text="Choose a saved policy to open it in MuJoCo.").pack(anchor="w", pady=(4, 12))
        toolbar = ttk.Frame(panel)
        toolbar.pack(fill="x", pady=(0, 8))
        ttk.Checkbutton(
            toolbar, text="Include archived runs", variable=self.archive, command=self.refresh
        ).pack(side="left")
        ttk.Button(toolbar, text="Refresh", command=self.refresh).pack(side="right")

        listing = ttk.Frame(panel)
        listing.pack(fill="both", expand=True)
        self.table = ttk.Treeview(
            listing, columns=("policy", "run", "checkpoint"), show="headings", selectmode="browse"
        )
        for column, title, width in (
            ("policy", "Policy", 210), ("run", "Training run", 290), ("checkpoint", "Checkpoint", 110)
        ):
            self.table.heading(column, text=title)
            self.table.column(column, width=width, minwidth=90, stretch=column != "checkpoint")
        scrollbar = ttk.Scrollbar(listing, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.table.pack(side="left", fill="both", expand=True)
        self.table.bind("<<TreeviewSelect>>", self.select)
        self.table.bind("<Double-1>", self.double_click)

        options = ttk.Frame(panel)
        options.pack(fill="x", pady=(12, 0))
        options.columnconfigure(1, weight=1)
        ttk.Label(options, text="Task").grid(row=0, column=0, sticky="w", padx=(0, 10))
        task_box = ttk.Combobox(options, textvariable=self.task, values=TASKS, state="readonly", width=22)
        task_box.grid(row=0, column=1, sticky="w")
        task_box.bind("<<ComboboxSelected>>", self.update_motion_fields)
        ttk.Checkbutton(
            options, text="Faster playback (bypass motor delay)", variable=self.fast
        ).grid(row=0, column=2, sticky="e")
        self.motion_label = ttk.Label(options, text="Motion file")
        self.motion_entry = ttk.Entry(options, textvariable=self.motion)
        self.motion_button = ttk.Button(options, text="Browse…", command=self.browse_motion)
        self.motion_label.grid(row=1, column=0, sticky="w", pady=8)
        self.motion_entry.grid(row=1, column=1, sticky="ew", padx=(0, 8))
        self.motion_button.grid(row=1, column=2, sticky="w")

        settings = ttk.Frame(panel)
        settings.pack(fill="x", pady=(10, 0))
        ttk.Label(settings, text="Environments").pack(side="left")
        ttk.Spinbox(settings, from_=1, to=1024, textvariable=self.num_envs, width=6).pack(
            side="left", padx=(6, 18)
        )
        ttk.Label(settings, text="Device").pack(side="left")
        ttk.Combobox(
            settings, textvariable=self.device, values=("Auto", "cuda:0", "cpu"),
            state="readonly", width=9,
        ).pack(side="left", padx=(6, 18))
        ttk.Label(settings, text="Seed").pack(side="left")
        ttk.Entry(settings, textvariable=self.seed, width=11).pack(side="left", padx=(6, 0))

        actions = ttk.Frame(panel)
        actions.pack(fill="x", pady=(12, 0))
        self.play_button = ttk.Button(actions, text="Play in MuJoCo", command=self.play)
        self.play_button.pack(side="left")
        self.stop_button = ttk.Button(actions, text="Stop playback", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=(8, 0))
        ttk.Label(panel, textvariable=self.status, wraplength=700).pack(anchor="w", pady=(8, 0))
        self.refresh()

    def refresh(self):
        selection = self.table.selection()
        selected_path = self.checkpoints.get(selection[0]) if selection else None
        self.table.delete(*self.table.get_children())
        self.checkpoints.clear()
        selected_row = None
        for path in find_checkpoints(self.archive.get()):
            experiment = path.parent.parent.name
            row = self.table.insert("", "end", values=(
                EXPERIMENT_TASK.get(experiment, experiment), path.parent.name, path.name
            ))
            self.checkpoints[row] = path
            if path == selected_path:
                selected_row = row
        if self.checkpoints:
            row = selected_row or next(iter(self.checkpoints))
            self.table.selection_set(row)
            self.table.see(row)
            self.select()
        else:
            self.task.set("")
            self.motion.set("")
            self.update_motion_fields()
        if self.process is None:
            self.play_button.configure(state="normal" if self.checkpoints else "disabled")
            self.status.set(f"{len(self.checkpoints)} checkpoints available." if self.checkpoints
                            else "No saved checkpoints found. Train a policy and click Refresh.")

    def select(self, _event=None):
        selection = self.table.selection()
        if not selection:
            return
        checkpoint = self.checkpoints[selection[0]]
        self.task.set(EXPERIMENT_TASK.get(checkpoint.parent.parent.name, ""))
        self.motion.set(saved_motion_file(checkpoint) if self.task.get().startswith("Tracking") else "")
        self.update_motion_fields()

    def update_motion_fields(self, _event=None):
        for widget in (self.motion_label, self.motion_entry, self.motion_button):
            if self.task.get().startswith("Tracking"):
                widget.grid()
            else:
                widget.grid_remove()

    def browse_motion(self):
        path = filedialog.askopenfilename(filetypes=[("Motion reference", "*.npz")])
        if path:
            self.motion.set(path)

    def double_click(self, event):
        if self.table.identify_row(event.y):
            self.play()

    def play(self):
        selection = self.table.selection()
        if self.process is not None or not selection:
            return
        checkpoint = self.checkpoints[selection[0]]
        task = self.task.get()
        if task not in TASKS:
            messagebox.showerror("Choose a task", "Select the task that trained this checkpoint.")
            return
        if not checkpoint.is_file():
            messagebox.showerror("Missing checkpoint", "The checkpoint moved or was removed. Click Refresh.")
            return
        try:
            num_envs = int(self.num_envs.get())
            if num_envs < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid environment count", "Enter a positive whole number.")
            return
        try:
            seed = int(self.seed.get())
            if not 0 <= seed < 2**32:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid seed", "Enter a whole number from 0 to 4294967295.")
            return
        command = [
            str(PYTHON), "-u", str(ROOT / "scripts/play.py"), TASK_PREFIX + task,
            f"--checkpoint-file={checkpoint}", f"--num-envs={num_envs}", "--viewer=native", f"--seed={seed}",
        ]
        if self.device.get() != "Auto":
            command.append(f"--device={self.device.get()}")
        if task.startswith("Tracking"):
            if not self.motion.get() or not (ROOT / Path(self.motion.get()).expanduser()).is_file():
                messagebox.showerror("Motion file required", "Choose the .npz reference used for this checkpoint.")
                return
            command.append(f"--motion-file={ROOT / Path(self.motion.get()).expanduser()}")
        if self.fast.get():
            command.extend(("--no-motor-delay", "True"))
        environment = os.environ.copy()
        environment.setdefault("WARP_CACHE_PATH", "/tmp/zlab-warp-cache")
        environment.setdefault("MJLAB_WARP_QUIET", "1")
        environment.setdefault("OMP_NUM_THREADS", "2")
        try:
            with tempfile.NamedTemporaryFile(prefix="z01-play-", suffix=".log", delete=False) as log:
                self.log_path = Path(log.name)
                self.process = subprocess.Popen(
                    command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        except OSError as error:
            messagebox.showerror("Could not launch MuJoCo", str(error))
            return
        self.status.set(f"Opening {task} · {checkpoint.name}. First launch can take a few minutes.")
        self.play_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.window.after(500, self.check_process)

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            self.status.set("Stopping playback…")
        self.stop_button.configure(state="disabled")

    def check_process(self):
        code = self.process.poll()
        if code is None:
            self.window.after(500, self.check_process)
            return
        self.process = None
        self.play_button.configure(state="normal" if self.checkpoints else "disabled")
        self.stop_button.configure(state="disabled")
        self.status.set("Playback closed." if code in (0, -15) else "Playback failed.")
        if code not in (0, -15):
            details = "\n".join(self.log_path.read_text(errors="replace").splitlines()[-10:])
            messagebox.showerror("MuJoCo playback failed", f"{details}\n\nLog: {self.log_path}")


if __name__ == "__main__":
    root = tk.Tk()
    CheckpointLauncher(root)
    root.mainloop()
