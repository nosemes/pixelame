from pathlib import Path
from uuid import uuid4
from zipfile import ZipFile, ZIP_DEFLATED
import shutil

import cv2
import face_recognition
from flask import Flask, render_template, request, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename

APP_DIR = Path(__file__).parent.resolve()
WORK_DIR = APP_DIR / "treball"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

app = Flask(__name__)
app.secret_key = "canvia-aquesta-clau-si-ho-poses-en-produccio"


def is_image(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def pixelate_face(image_bgr, top, right, bottom, left, pixel_size=5, margin=20):
    height, width = image_bgr.shape[:2]

    top = max(0, top - margin)
    bottom = min(height, bottom + margin)
    left = max(0, left - margin)
    right = min(width, right + margin)

    face_roi = image_bgr[top:bottom, left:right]
    if face_roi.size == 0:
        return image_bgr

    small = cv2.resize(face_roi, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (right - left, bottom - top), interpolation=cv2.INTER_NEAREST)
    image_bgr[top:bottom, left:right] = pixelated
    return image_bgr


def process_images(reference_path: Path, input_dir: Path, output_dir: Path, tolerance=0.55, pixel_size=5, margin=20):
    reference_image = face_recognition.load_image_file(str(reference_path))
    reference_encodings = face_recognition.face_encodings(reference_image)

    if not reference_encodings:
        raise ValueError("No s'ha detectat cap cara a la foto de referència.")

    reference_face = reference_encodings[0]
    results = []

    for source_path in sorted(input_dir.rglob("*")):
        if not source_path.is_file() or not is_image(source_path.name):
            continue

        relative_path = source_path.relative_to(input_dir)
        destination_path = output_dir / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            image_rgb = face_recognition.load_image_file(str(source_path))
            locations = face_recognition.face_locations(image_rgb)
            encodings = face_recognition.face_encodings(image_rgb, locations)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            matches_count = 0
            for encoding, (top, right, bottom, left) in zip(encodings, locations):
                is_match = face_recognition.compare_faces(
                    [reference_face],
                    encoding,
                    tolerance=float(tolerance),
                )[0]

                if is_match:
                    matches_count += 1
                    image_bgr = pixelate_face(
                        image_bgr,
                        top,
                        right,
                        bottom,
                        left,
                        pixel_size=int(pixel_size),
                        margin=int(margin),
                    )

            cv2.imwrite(str(destination_path), image_bgr)
            results.append((str(relative_path), matches_count, None))
        except Exception as exc:
            results.append((str(relative_path), 0, str(exc)))

    return results


def zip_folder(folder: Path, zip_path: Path):
    with ZipFile(zip_path, "w", ZIP_DEFLATED) as zf:
        for file_path in folder.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(folder))


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    reference = request.files.get("reference")
    photos = request.files.getlist("photos")

    if not reference or reference.filename == "":
        flash("Has de seleccionar una foto de referència.")
        return redirect(url_for("index"))

    valid_photos = [f for f in photos if f.filename and is_image(f.filename)]
    if not valid_photos:
        flash("Has de seleccionar almenys una foto JPG, PNG o WEBP per processar.")
        return redirect(url_for("index"))

    tolerance = request.form.get("tolerance", "0.55")
    pixel_size = request.form.get("pixel_size", "5")
    margin = request.form.get("margin", "20")

    job_id = uuid4().hex
    job_dir = WORK_DIR / job_id
    input_dir = job_dir / "originals"
    output_dir = job_dir / "pixelades"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_path = job_dir / secure_filename(reference.filename or "referencia.jpg")
    reference.save(reference_path)

    for photo in valid_photos:
        # Manté subcarpetes si el navegador les envia amb webkitdirectory.
        safe_parts = [secure_filename(part) for part in Path(photo.filename).parts if part not in ("", ".")]
        relative = Path(*safe_parts) if safe_parts else Path(secure_filename(photo.filename))
        target = input_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        photo.save(target)

    try:
        results = process_images(
            reference_path=reference_path,
            input_dir=input_dir,
            output_dir=output_dir,
            tolerance=tolerance,
            pixel_size=pixel_size,
            margin=margin,
        )
    except Exception as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        flash(str(exc))
        return redirect(url_for("index"))

    zip_path = job_dir / "fotos_pixelades.zip"
    zip_folder(output_dir, zip_path)

    total = len(results)
    matched = sum(1 for _, count, error in results if count > 0 and error is None)
    errors = sum(1 for _, _, error in results if error is not None)

    return render_template(
        "result.html",
        job_id=job_id,
        total=total,
        matched=matched,
        errors=errors,
        results=results,
    )


@app.route("/descarrega/<job_id>")
def download(job_id):
    zip_path = WORK_DIR / job_id / "fotos_pixelades.zip"
    if not zip_path.exists():
        flash("No s'ha trobat el ZIP de resultats.")
        return redirect(url_for("index"))
    return send_file(zip_path, as_attachment=True, download_name="fotos_pixelades.zip")


if __name__ == "__main__":
    WORK_DIR.mkdir(exist_ok=True)
    app.run(debug=True, host="127.0.0.1", port=5000)
