"""
Salidas visuales: gráfico para calibrar umbrales, exportación de jugadas (clips o
compilado) y video de diagnóstico con las cajas de pelota y jugadores.
"""
import os

import cv2
import numpy as np

from .common import meta_to_dict


def plot_rallies(traj, rallies, meta, t_from_sec=0, t_to_sec=180):
    """
    Altura y velocidad vertical de la pelota con las jugadas marcadas (para calibrar).

    Args:
        traj (pd.DataFrame): trayectoria de la pelota.
        rallies (pd.DataFrame): jugadas.
        meta (dict | pd.DataFrame): metadatos del video.
        t_from_sec (float): inicio del tramo a graficar (s).
        t_to_sec (float): fin del tramo a graficar (s).

    Returns:
        None: muestra el gráfico.
    """
    meta = meta_to_dict(meta)
    import matplotlib.pyplot as plt

    fps = meta["fps"]
    a, b = int(t_from_sec * fps), int(t_to_sec * fps)
    seg = traj.iloc[a:b]
    t = seg["frame"] / fps

    fig, ax = plt.subplots(2, 1, figsize=(18, 6), sharex=True)
    ax[0].plot(t, seg["cy"], lw=0.8)
    ax[0].scatter(t[seg["detected"]], seg["cy"][seg["detected"]], s=2, c="k")
    ax[0].invert_yaxis()
    ax[0].set_ylabel("cy (px, arriba = alto)")
    ax[1].plot(t, seg["vy"], lw=0.8)
    ax[1].axhline(0, c="gray", lw=0.5)
    ax[1].set_ylabel("vy (px/s, + = cae)")
    ax[1].set_xlabel("tiempo (s)")

    for _, r in rallies.iterrows():
        s0, s1 = r["serve_frame"] / fps, r["end_frame"] / fps
        if s1 < t_from_sec or s0 > t_to_sec:
            continue
        for axis in ax:
            axis.axvspan(s0, s1, color="green", alpha=0.15)
            axis.axvline(s1, color="red", lw=1)
        ax[0].text(s1, ax[0].get_ylim()[1], r["end_reason"], fontsize=7, color="red", rotation=90, va="top")
    plt.tight_layout()
    plt.show()


def export_rallies(video_path, rallies, out_dir=None, compilation_path=None, draw_info=False):
    """
    Lee el video secuencialmente (sin cap.set, que en mp4 salta a keyframes y
    desfasa los cortes) y escribe un clip por jugada y/o un compilado.

    Args:
        video_path (str): ruta del video original.
        rallies (pd.DataFrame): jugadas (clip_start, clip_end, serve_frame, end_frame, ...).
        out_dir (str | None): carpeta para un clip por jugada (None = no se escriben).
        compilation_path (str | None): ruta del video compilado (None = no se escribe).
        draw_info (bool): dibuja el número de jugada y marca SAQUE / FIN (útil para revisar).

    Returns:
        None: escribe los videos en disco.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Chequeo: ¿es el mismo video con el que se calcularon las detecciones?
    last_needed = int(rallies["clip_end"].max()) if len(rallies) else 0
    if last_needed >= n_video:
        outside = rallies[rallies["clip_start"] >= n_video]["rally_id"].tolist()
        print(f"⚠️  ATENCIÓN: '{video_path}' tiene {n_video} frames, pero las jugadas llegan al frame "
              f"{last_needed}. ¿Es el mismo video que se usó en track_play_events?\n"
              f"    Jugadas que quedan fuera de este video: {outside}")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    comp = cv2.VideoWriter(compilation_path, fourcc, fps, (w, h)) if compilation_path else None

    rows = rallies.to_dict("records")
    i, f, clip = 0, 0, None
    mark = int(0.5 * fps)
    bad_frames, bad_in_row, exported = 0, 0, []
    last_frame = None

    def skip_bad_frame():
        """
        Frame que no se pudo leer: saltarlo en vez de terminar la exportación.

        Returns:
            None
        """
        nonlocal bad_frames, bad_in_row
        bad_frames += 1
        bad_in_row += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, f + 1)

    while i < len(rows):
        r = rows[i]
        if f >= n_video or bad_in_row > int(5 * fps):
            break  # fin del video (o video dañado más de 5 s seguidos)
        if f < r["clip_start"]:
            if cap.grab():
                bad_in_row = 0
            else:
                skip_bad_frame()
            f += 1
            continue

        ret, frame = cap.read()
        if not ret:
            skip_bad_frame()
            if last_frame is None:
                f += 1
                continue
            frame = last_frame.copy()  # repetir el anterior para no desfasar el clip
        else:
            bad_in_row = 0
            last_frame = frame

        if clip is None and out_dir:
            clip = cv2.VideoWriter(os.path.join(out_dir, f"rally_{r['rally_id']:03d}.mp4"),
                                   fourcc, fps, (w, h))

        if draw_info:
            cv2.putText(frame, f"Jugada {r['rally_id']}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            # Las etiquetas aparecen EN el frame del evento (no antes) y duran `mark`
            if 0 <= f - r["serve_frame"] <= mark:
                cv2.putText(frame, "SAQUE", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            if 0 <= f - r["end_frame"] <= mark:
                cv2.putText(frame, f"FIN: {r['end_reason']}", (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        if clip is not None:
            clip.write(frame)
        if comp is not None:
            comp.write(frame)

        if f >= r["clip_end"]:
            if clip is not None:
                clip.release()
                clip = None
            print(f"Jugada {r['rally_id']} exportada ({r['serve_frame']} a {r['end_frame']}, {r['end_reason']})")
            exported.append(r["rally_id"])
            i += 1
        f += 1

    if clip is not None:
        clip.release()
    if comp is not None:
        comp.release()
    cap.release()
    missing_frames = [r["rally_id"] for r in rows if r["rally_id"] not in exported]
    if bad_frames:
        print(f"⚠️  {bad_frames} frames no se pudieron leer y se saltaron.")
    if missing_frames:
        print(f"⚠️  NO se exportaron {len(missing_frames)} jugadas: {missing_frames}")
    print(f"¡Exportación terminada! {len(exported)} de {len(rows)} jugadas.")


def draw_detections(video_path, df_det, out_path, rallies=None, frames=None, traj=None,
                        net_x=None, margin_sec=1.0, out_scale=0.5, verbose=True):
    """
    Genera un video con las detecciones dibujadas para ver dónde se pierde la pelota.
      - Jugadores: caja celeste (roja si es primer plano, alto > 40% del cuadro).
      - Pelota detectada: caja amarilla con su confianza.
      - traj: punto verde = posición usada (detectada), punto naranja = interpolada;
        texto rojo "SIN PELOTA" si no hay ninguna.
      - net_x: línea magenta de la red.
      - Arriba: nº de frame, tiempo, jugada y "PELOTA PERDIDA hace X s".
    Qué frames se dibujan: `frames` = (inicio, fin), o todas las jugadas de `rallies`
    (± margin_sec), o todo el video si no se pasa nada (¡largo!).

    Args:
        video_path (str): ruta del video original.
        df_det (pd.DataFrame): detecciones a dibujar.
        out_path (str): ruta del video de salida.
        rallies (pd.DataFrame | None): jugadas a dibujar.
        frames (tuple | None): (inicio, fin) a dibujar; tiene prioridad sobre rallies.
        traj (pd.DataFrame | None): salida de build_ball_trajectory.
        net_x (np.ndarray | None): x de la red por frame, p. ej. net_from_pan(...)[0].
        margin_sec (float): margen antes/después de cada jugada.
        out_scale (float): escala del video de salida (0.5 = 960x540, más liviano).
        verbose (bool): imprime el progreso.

    Returns:
        None: escribe el video en out_path.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frames is not None:
        segments = [(max(0, int(frames[0])), min(n - 1, int(frames[1])))]
    elif rallies is not None:
        m = int(margin_sec * fps)
        segments = [(max(0, int(a) - m), min(n - 1, int(b) + m))
                  for a, b in zip(rallies["serve_frame"], rallies["end_frame"])]
    else:
        segments = [(0, n - 1)]
    rid_of = {}
    if rallies is not None:
        for _, r in rallies.iterrows():
            for f in range(int(r["serve_frame"]), int(r["end_frame"]) + 1):
                rid_of[f] = int(r["rally_id"])
    per_frame = {f: g for f, g in df_det.groupby("frame")}
    if traj is not None:
        tcx, tcy, tdet = traj["cx"].to_numpy(), traj["cy"].to_numpy(), traj["detected"].to_numpy()
    ow, oh = int(W * out_scale), int(H * out_scale)
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
    s = 1 / out_scale
    # Lectura secuencial (como export_rallies): cap.set en mp4 salta a keyframes y desfasa
    # las cajas respecto de la imagen
    in_segment = np.zeros(n + 1, dtype=bool)
    for a, b in segments:
        in_segment[a:b + 1] = True
    f_act = 0
    for a, b in segments:
        while f_act < a:
            if not cap.grab():
                break
            f_act += 1
        last_ball = None
        for f in range(a, b + 1):
            if f < f_act:
                continue                       # tramos solapados: ya dibujado
            ok, im = cap.read()
            f_act += 1
            if not ok:
                break
            im = cv2.resize(im, (ow, oh))
            g = per_frame.get(f)
            has_ball = False
            if g is not None:
                for _, d in g.iterrows():
                    p1 = (int(d["x1"] / s), int(d["y1"] / s))
                    p2 = (int(d["x2"] / s), int(d["y2"] / s))
                    if d["type"] == "ball":
                        has_ball = True
                        cv2.rectangle(im, p1, p2, (0, 255, 255), 2)
                        cv2.putText(im, f"{d['conf']:.2f}", (p1[0], p1[1] - 4), 0, 0.45, (0, 255, 255), 1)
                    else:
                        is_big = (d["y2"] - d["y1"]) / H > 0.40
                        cv2.rectangle(im, p1, p2, (0, 0, 255) if is_big else (255, 200, 0), 1)
            if has_ball:
                last_ball = f
            if traj is not None and f < len(tcx):
                if np.isnan(tcx[f]):
                    cv2.putText(im, "SIN PELOTA", (ow - 190, 60), 0, 0.8, (0, 0, 255), 2)
                else:
                    color = (0, 255, 0) if tdet[f] else (0, 140, 255)
                    cv2.circle(im, (int(tcx[f] / s), int(tcy[f] / s)), 6, color, -1)
            if net_x is not None and f < len(net_x) and not np.isnan(net_x[f]):
                x = int(net_x[f] / s)
                cv2.line(im, (x, 0), (x, oh), (255, 0, 255), 1)
            txt = f"frame {f}  t={f / fps:7.2f}s"
            if f in rid_of:
                txt += f"  jugada {rid_of[f]}"
            cv2.putText(im, txt, (10, 25), 0, 0.6, (255, 255, 255), 2)
            if not has_ball:
                lost_s = (f - last_ball) / fps if last_ball is not None else None
                cv2.putText(im, "PELOTA PERDIDA" + (f" hace {lost_s:.2f}s" if lost_s is not None else ""),
                            (10, 50), 0, 0.6, (0, 0, 255), 2)
            out.write(im)
        if verbose:
            print(f"Dibujado {a}-{b}")
    cap.release()
    out.release()
    if verbose:
        print(f"Video de diagnóstico: {out_path}")
