import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile

st.title("Ski Technique Analyzer (Prototype)")

uploaded_file = st.file_uploader("Загрузите видео спуска (mp4)", type=["mp4"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    bad_posture_frames = 0
    angles = []

    stframe = st.empty()
    progress = st.progress(0)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        angle_deg = None

        if results.pose_landmarks:
            # Анализ обеих сторон
            for side in ['RIGHT', 'LEFT']:
                shoulder = results.pose_landmarks.landmark[getattr(mp_pose.PoseLandmark, f"{side}_SHOULDER")]
                hip = results.pose_landmarks.landmark[getattr(mp_pose.PoseLandmark, f"{side}_HIP")]
                knee = results.pose_landmarks.landmark[getattr(mp_pose.PoseLandmark, f"{side}_KNEE")]

                v1 = np.array([shoulder.x - hip.x, shoulder.y - hip.y])
                v2 = np.array([knee.x - hip.x, knee.y - hip.y])

                angle = np.arccos(
                    np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                )
                angle_deg_side = np.degrees(angle)
                if angle_deg is None:
                    angle_deg = angle_deg_side
                else:
                    angle_deg = (angle_deg + angle_deg_side) / 2

            angles.append(angle_deg)

            if angle_deg < 120:
                bad_posture_frames += 1
                cv2.putText(frame, f"Warning: Too much forward lean! ({int(angle_deg)} deg)", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, f"Good posture ({int(angle_deg)} deg)", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        stframe.image(frame, channels="BGR")
        progress.progress(min(frame_count / total_frames, 1.0))

    cap.release()
    pose.close()

    st.write(f"Всего кадров: {frame_count}")
    st.write(f"Кадров с ошибкой (слишком сильный наклон): {bad_posture_frames}")
    if bad_posture_frames > 0:
        st.error("Обратите внимание: часто слишком сильно наклоняетесь вперёд!")
    else:
        st.success("Поздравляем! Сильных наклонов корпуса не обнаружено.")

    st.line_chart(angles, height=200, width=700)
