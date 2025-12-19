import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import cv2

# I keep the dataset in a local folder with two class subfolders:
# dataset/Drowsy and dataset/Non Drowsy
DATASET_DIR = os.path.join(os.getcwd(), "dataset")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# I load the images using a simple 80/20 split.
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

class_names = train_ds.class_names
print("Class names:", class_names)

# I visualize a few samples to confirm the dataset is correct.
plt.figure(figsize=(10, 8))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[int(labels[i])])
        plt.axis("off")
plt.show()

# I add light augmentation so the model generalizes better.
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
], name="augmentation")

# I build a transfer learning model using MobileNetV2.
base_model = keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, outputs, name="vision_agent_mobilenetv2")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()
# I use early stopping so training stops if validation performance stops improving.
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks
)

# I plot accuracy and loss so I can include them as evidence in the report.
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.figure(figsize=(8, 5))
plt.plot(acc, label="train_accuracy")
plt.plot(val_acc, label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(loss, label="train_loss")
plt.plot(val_loss, label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# I save the model locally so I can load it later for agent workflow simulations.
model.save("vision_agent_model.keras")
print("Saved model as vision_agent_model.keras")

# I create one OpenCV-style visualization (overlay prediction on a sample image).
# This gives me strong screenshot evidence for the Vision Agent output.
sample_path = None
for root, dirs, files in os.walk(DATASET_DIR):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            sample_path = os.path.join(root, f)
            break
    if sample_path:
        break

if sample_path:
    image_bgr = cv2.imread(sample_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(image_rgb, IMG_SIZE).astype(np.float32) / 255.0
    pred = float(model.predict(resized.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 3))[0][0])

    pred_label = class_names[int(pred > 0.5)]
    text = f"Pred: {pred_label} ({pred:.2f})"

    display_img = image_rgb.copy()
    cv2.putText(display_img, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    plt.figure(figsize=(7, 5))
    plt.imshow(display_img)
    plt.axis("off")
    plt.title("Vision Agent Prediction Overlay")
    plt.show()
else:
    print("No image found for OpenCV overlay demo. Check dataset folder.")

import time
import csv
from datetime import datetime

# -----------------------------
# State Agent (Temporal reasoning)
# -----------------------------
# I keep this agent simple and transparent for coursework:
# it aggregates recent Vision Agent probabilities over a sliding window
# and adds context (time of day + driving duration) to compute risk.

class StateAgent:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.prob_history = []

    def update(self, drowsy_prob: float, time_of_day: str, driving_duration_min: int):
        # I store only the latest N probabilities.
        self.prob_history.append(drowsy_prob)
        if len(self.prob_history) > self.window_size:
            self.prob_history.pop(0)

        avg_prob = float(np.mean(self.prob_history))

        # Context weighting: night driving and longer driving increases risk slightly.
        context_bonus = 0.0
        if time_of_day.lower() in ["night", "late_night"]:
            context_bonus += 0.05
        if driving_duration_min >= 120:
            context_bonus += 0.05
        if driving_duration_min >= 240:
            context_bonus += 0.05

        # Final risk score is capped to 1.0
        risk_score = min(1.0, avg_prob + context_bonus)

        # Risk level thresholds (easy to justify in report)
        if risk_score < 0.40:
            fatigue_level = "Low"
        elif risk_score < 0.70:
            fatigue_level = "Medium"
        else:
            fatigue_level = "High"

        return {
            "avg_drowsy_prob_window": avg_prob,
            "context_bonus": context_bonus,
            "risk_score": risk_score,
            "fatigue_level": fatigue_level
        }


# -----------------------------
# Action Agent (Decision + intervention)
# -----------------------------
# I define an adaptive intervention profile as a simple dictionary.
# In real deployment this could be learned; for coursework this is enough to show adaptivity.

class ActionAgent:
    def __init__(self, profile=None):
        if profile is None:
            profile = {
                "preferred_alert": "seat_vibration",   # audio / seat_vibration / visual
                "tolerance_level": "medium",          # low / medium / high
                "min_seconds_between_alerts": 30
            }
        self.profile = profile
        self.last_alert_time = 0.0

    def decide(self, fatigue_level: str, risk_score: float):
        now = time.time()

        # I avoid spamming the driver with alerts.
        if (now - self.last_alert_time) < self.profile["min_seconds_between_alerts"]:
            return {
                "action": "no_action",
                "intensity": "none",
                "message": "No alert triggered (cooldown active)."
            }

        if fatigue_level == "Low":
            return {
                "action": "no_action",
                "intensity": "none",
                "message": "Driver appears alert."
            }

        if fatigue_level == "Medium":
            # For medium risk I choose a gentle alert.
            self.last_alert_time = now
            return {
                "action": "audio_chime",
                "intensity": "low",
                "message": "Fatigue signs detected. Please stay alert."
            }

        # High risk: I use the preferred alert with stronger intensity.
        self.last_alert_time = now

        preferred = self.profile.get("preferred_alert", "seat_vibration")
        intensity = "high" if risk_score >= 0.85 else "medium"

        if preferred == "visual":
            msg = "High fatigue detected. Please take a break soon."
        elif preferred == "audio":
            msg = "High fatigue detected. Please stop and rest."
        else:
            msg = "High fatigue detected. Seat vibration activated. Take a break."

        return {
            "action": preferred,
            "intensity": intensity,
            "message": msg
        }


# -----------------------------
# CSV Logging (Evidence)
# -----------------------------
# I log key outputs so I can show evidence of the full workflow in Section 4.

LOG_PATH = "agent_run_log.csv"

def init_log_file(path: str):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "frame_path",
                "vision_pred_prob",
                "vision_pred_label",
                "state_avg_prob_window",
                "state_context_bonus",
                "state_risk_score",
                "state_fatigue_level",
                "action",
                "action_intensity",
                "action_message"
            ])

def append_log(path: str, row: dict):
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            row["timestamp"],
            row["frame_path"],
            row["vision_pred_prob"],
            row["vision_pred_label"],
            row["state_avg_prob_window"],
            row["state_context_bonus"],
            row["state_risk_score"],
            row["state_fatigue_level"],
            row["action"],
            row["action_intensity"],
            row["action_message"]
        ])


# -----------------------------
# Helper: run the Vision Agent on a single image path
# -----------------------------
def vision_agent_predict(image_path: str):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(image_rgb, IMG_SIZE).astype(np.float32) / 255.0
    prob = float(model.predict(resized.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 3), verbose=0)[0][0])

    pred_label = class_names[int(prob > 0.5)]

    return prob, pred_label, image_rgb


# -----------------------------
# Simulated full workflow execution (Section 4 evidence)
# -----------------------------
# I simulate a short driving session by processing multiple frames/images in sequence.
# This produces clear step-by-step outputs for screenshots.

def collect_sample_frames(dataset_dir: str, limit=12):
    frames = []
    for root, dirs, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                frames.append(os.path.join(root, f))
                if len(frames) >= limit:
                    return frames
    return frames

# I define context for a representative use case: night driving + long duration.
time_of_day = "night"
driving_duration_min = 150  # 2.5 hours

state_agent = StateAgent(window_size=8)
action_agent = ActionAgent(profile={
    "preferred_alert": "seat_vibration",
    "tolerance_level": "medium",
    "min_seconds_between_alerts": 0  # set to 0 for demo so alerts show in evidence
})

init_log_file(LOG_PATH)

frames = collect_sample_frames(DATASET_DIR, limit=12)
print(f"Collected {len(frames)} sample frames for simulated run.")

# I run the full pipeline and print outputs so I can capture screenshots.
for idx, frame_path in enumerate(frames, start=1):
    prob, pred_label, image_rgb = vision_agent_predict(frame_path)

    state_out = state_agent.update(
        drowsy_prob=prob,
        time_of_day=time_of_day,
        driving_duration_min=driving_duration_min
    )

    action_out = action_agent.decide(
        fatigue_level=state_out["fatigue_level"],
        risk_score=state_out["risk_score"]
    )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_row = {
        "timestamp": timestamp,
        "frame_path": frame_path,
        "vision_pred_prob": round(prob, 4),
        "vision_pred_label": pred_label,
        "state_avg_prob_window": round(state_out["avg_drowsy_prob_window"], 4),
        "state_context_bonus": round(state_out["context_bonus"], 4),
        "state_risk_score": round(state_out["risk_score"], 4),
        "state_fatigue_level": state_out["fatigue_level"],
        "action": action_out["action"],
        "action_intensity": action_out["intensity"],
        "action_message": action_out["message"]
    }

    append_log(LOG_PATH, log_row)

    # I print a clean, screenshot-friendly summary for each step.
    print("\n--- STEP", idx, "---")
    print("Input frame:", frame_path)
    print("Vision Agent output:", {"drowsy_prob": round(prob, 4), "pred_label": pred_label})
    print("State Agent output:", state_out)
    print("Action Agent output:", action_out)

    # I also generate a visual evidence frame with overlay.
    display_img = image_rgb.copy()
    overlay_text = f"{pred_label} ({prob:.2f}) | Risk: {state_out['fatigue_level']} ({state_out['risk_score']:.2f})"
    cv2.putText(display_img, overlay_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    plt.figure(figsize=(8, 5))
    plt.imshow(display_img)
    plt.axis("off")
    plt.title(f"Workflow Evidence - Step {idx}")
    plt.show()

print(f"\nWorkflow log saved to: {LOG_PATH}")
