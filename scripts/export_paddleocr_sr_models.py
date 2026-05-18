#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_paddleocr_sr_models.py
============================================================
Downloader + exporter for PaddleOCR text super-resolution models.

Why this script exists
----------------------
`pip install paddleocr` is enough for many runtime OCR tasks, but it normally
does not give a convenient local checkout with:

    tools/export_model.py
    tools/infer/predict_sr.py
    configs/sr/*.yml
    ppocr/* training/export modules

PaddleOCR SR weights are published as TRAINING checkpoints.  Our OCR pipeline
expects exported inference models:

    inference.pdmodel
    inference.pdiparams

This script downloads the PaddleOCR source tree into `third_party/PaddleOCR`,
downloads the SR training checkpoints, extracts them, and runs PaddleOCR's
`tools/export_model.py` using the current Python environment.

Supported SR models
-------------------
1. Text Telescope / TBSRN
   Config: configs/sr/sr_telescope.yml
   Output: models/paddle_sr/sr_telescope

2. Text Gestalt / TSRN
   Config: configs/sr/sr_tsrn_transformer_strock.yml
   Output: models/paddle_sr/sr_gestalt

Usage examples
--------------
Export both models with the current Python:

    python scripts/export_paddleocr_sr_models.py --models both

Export only Telescope:

    python scripts/export_paddleocr_sr_models.py --models telescope

Use an already cloned PaddleOCR repository:

    python scripts/export_paddleocr_sr_models.py ^
      --models both ^
      --repo-dir D:/dev/PaddleOCR

Use a specific branch/ref:

    python scripts/export_paddleocr_sr_models.py --repo-ref main
    python scripts/export_paddleocr_sr_models.py --repo-ref release/2.7

After export, set YAML model_dir to:

    models/paddle_sr/sr_telescope
    models/paddle_sr/sr_gestalt
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class SRModelSpec:
    key: str
    display_name: str
    config_rel: str
    archive_name: str
    urls: Tuple[str, ...]
    export_subdir: str
    image_shape: str = "3,32,128"


SR_MODELS = {
    "telescope": SRModelSpec(
        key="telescope",
        display_name="Text Telescope / TBSRN",
        config_rel="configs/sr/sr_telescope.yml",
        archive_name="sr_telescope_train.tar",
        urls=(
            # Main model-zoo link from PaddleOCR docs.
            "https://paddleocr.bj.bcebos.com/contribution/sr_telescope_train.tar",
            # Export section in the docs also mentions this older filename.
            "https://paddleocr.bj.bcebos.com/contribution/Telescope_train.tar.gz",
        ),
        export_subdir="sr_telescope",
    ),
    "gestalt": SRModelSpec(
        key="gestalt",
        display_name="Text Gestalt / TSRN",
        config_rel="configs/sr/sr_tsrn_transformer_strock.yml",
        archive_name="sr_tsrn_transformer_strock_train.tar",
        urls=(
            "https://paddleocr.bj.bcebos.com/sr_tsrn_transformer_strock_train.tar",
        ),
        export_subdir="sr_gestalt",
    ),
}


class ExportError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(msg, flush=True)


def project_root_from_script() -> Path:
    # scripts/export_paddleocr_sr_models.py -> project root
    return Path(__file__).resolve().parents[1]


def safe_mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def which_python(value: str) -> str:
    return value if value else sys.executable


def parse_models(value: str) -> List[SRModelSpec]:
    v = value.lower().strip()
    if v == "both":
        return [SR_MODELS["telescope"], SR_MODELS["gestalt"]]
    if v not in SR_MODELS:
        raise ExportError(f"Unknown model '{value}'. Use: telescope, gestalt, both")
    return [SR_MODELS[v]]


def verify_paddle_runtime(python_exe: str) -> None:
    code = "import paddle; import paddle.inference; print('paddle', paddle.__version__)"
    cmd = [python_exe, "-c", code]
    log("[check] Paddle runtime: " + " ".join(cmd))
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise ExportError(
            "PaddlePaddle is not importable in this Python environment.\n"
            "Install one of:\n"
            "  pip install paddlepaddle\n"
            "  pip install paddlepaddle-gpu\n\n"
            f"Command output:\n{proc.stdout}"
        )
    log(proc.stdout.strip())


def ensure_paddleocr_repo(
    *,
    repo_dir: Path,
    repo_url: str,
    repo_ref: str,
    force_repo_download: bool,
    dry_run: bool,
) -> Path:
    export_py = repo_dir / "tools" / "export_model.py"
    if export_py.exists() and not force_repo_download:
        log(f"[repo] using existing PaddleOCR source: {repo_dir}")
        return repo_dir

    if dry_run:
        log(f"[repo] dry-run: would download PaddleOCR source to {repo_dir}")
        return repo_dir

    if repo_dir.exists() and force_repo_download:
        log(f"[repo] removing existing repo dir: {repo_dir}")
        shutil.rmtree(repo_dir)

    safe_mkdir(repo_dir.parent)

    # Prefer git if available.  It handles branches like release/2.7 correctly.
    git = shutil.which("git")
    if git:
        cmd = [git, "clone", "--depth", "1", "--branch", repo_ref, repo_url, str(repo_dir)]
        log("[repo] cloning: " + " ".join(cmd))
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if proc.returncode == 0 and export_py.exists():
            return repo_dir
        log("[repo] git clone failed, falling back to GitHub zip archive")
        log(proc.stdout[-2000:])
        if repo_dir.exists():
            shutil.rmtree(repo_dir, ignore_errors=True)

    # Zip fallback.  Try heads first, then tags.
    zip_tmp = repo_dir.parent / f"PaddleOCR_{sanitize_ref(repo_ref)}.zip"
    zip_urls = [
        f"https://codeload.github.com/PaddlePaddle/PaddleOCR/zip/refs/heads/{repo_ref}",
        f"https://codeload.github.com/PaddlePaddle/PaddleOCR/zip/refs/tags/{repo_ref}",
    ]
    last_error: Optional[BaseException] = None
    for url in zip_urls:
        try:
            log(f"[repo] downloading source zip: {url}")
            download_file(url, zip_tmp, force=True)
            extract_zip_single_root(zip_tmp, repo_dir)
            if export_py.exists():
                return repo_dir
        except BaseException as e:  # noqa: BLE001 - useful diagnostics for CLI script
            last_error = e
            log(f"[repo] zip attempt failed: {type(e).__name__}: {e}")
            if repo_dir.exists():
                shutil.rmtree(repo_dir, ignore_errors=True)

    raise ExportError(
        "Cannot obtain PaddleOCR source tree with tools/export_model.py.\n"
        f"repo_dir={repo_dir}\nrepo_ref={repo_ref}\nlast_error={last_error}"
    )


def sanitize_ref(ref: str) -> str:
    return ref.replace("/", "_").replace("\\", "_").replace(":", "_")


def download_file(url: str, dst: Path, *, force: bool = False, min_bytes: int = 1024) -> Path:
    if dst.exists() and not force and dst.stat().st_size >= min_bytes:
        log(f"[download] exists: {dst} ({dst.stat().st_size / 1024 / 1024:.2f} MB)")
        return dst

    safe_mkdir(dst.parent)
    tmp = dst.with_suffix(dst.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    log(f"[download] {url}")
    log(f"           -> {dst}")
    start = time.time()

    req = urllib.request.Request(url, headers={"User-Agent": "ocr-lenta-paddleocr-sr-export/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        last_print = 0.0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            now = time.time()
            if now - last_print > 1.0:
                if total > 0:
                    pct = done * 100.0 / total
                    log(f"[download] {pct:5.1f}% {done / 1024 / 1024:.1f}/{total / 1024 / 1024:.1f} MB")
                else:
                    log(f"[download] {done / 1024 / 1024:.1f} MB")
                last_print = now

    if tmp.stat().st_size < min_bytes:
        raise ExportError(f"Downloaded file is too small: {tmp} ({tmp.stat().st_size} bytes)")
    tmp.replace(dst)
    log(f"[download] done in {time.time() - start:.1f}s: {dst}")
    return dst


def extract_zip_single_root(zip_path: Path, dst_dir: Path) -> None:
    tmp_dir = dst_dir.parent / (dst_dir.name + "_zip_extract_tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    safe_mkdir(tmp_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_dir)
    roots = [p for p in tmp_dir.iterdir() if p.is_dir()]
    if len(roots) != 1:
        raise ExportError(f"Expected one root directory inside {zip_path}, got {roots}")
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.move(str(roots[0]), str(dst_dir))
    shutil.rmtree(tmp_dir, ignore_errors=True)


def download_model_archive(spec: SRModelSpec, archives_dir: Path, *, force: bool, dry_run: bool) -> Path:
    out = archives_dir / spec.archive_name
    if dry_run:
        log(f"[model:{spec.key}] dry-run: would download archive to {out}")
        return out

    if out.exists() and not force and out.stat().st_size > 1024:
        log(f"[model:{spec.key}] archive exists: {out}")
        return out

    errors: List[str] = []
    for i, url in enumerate(spec.urls):
        # Preserve filename for fallback Telescope_train.tar.gz.
        candidate = out if i == 0 else archives_dir / Path(url).name
        try:
            return download_file(url, candidate, force=force)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ExportError) as e:
            errors.append(f"{url}: {type(e).__name__}: {e}")
            log(f"[model:{spec.key}] download failed: {errors[-1]}")

    raise ExportError("All downloads failed for " + spec.display_name + "\n" + "\n".join(errors))


def extract_archive(archive: Path, extract_dir: Path, *, force: bool, dry_run: bool) -> Path:
    marker = extract_dir / ".extract_ok"
    if marker.exists() and not force:
        log(f"[extract] using existing extraction: {extract_dir}")
        return extract_dir

    if dry_run:
        log(f"[extract] dry-run: would extract {archive} to {extract_dir}")
        return extract_dir

    if extract_dir.exists() and force:
        shutil.rmtree(extract_dir)
    safe_mkdir(extract_dir)

    log(f"[extract] {archive} -> {extract_dir}")
    mode = "r:gz" if archive.name.endswith((".tar.gz", ".tgz")) else "r"
    try:
        with tarfile.open(archive, mode) as tf:
            safe_extract_tar(tf, extract_dir)
    except tarfile.ReadError:
        # Some .tar files may still be gzipped or vice versa.
        with tarfile.open(archive, "r:*") as tf:
            safe_extract_tar(tf, extract_dir)

    marker.write_text("ok\n", encoding="utf-8")
    return extract_dir


def safe_extract_tar(tf: tarfile.TarFile, dst: Path) -> None:
    dst_resolved = dst.resolve()
    for member in tf.getmembers():
        member_path = (dst / member.name).resolve()
        if not str(member_path).startswith(str(dst_resolved)):
            raise ExportError(f"Unsafe path inside tar: {member.name}")
    tf.extractall(dst)


def find_checkpoint_prefix(extract_dir: Path, explicit_prefix: str = "") -> Path:
    if explicit_prefix:
        p = Path(explicit_prefix).expanduser()
        return strip_pdparams_suffix(p)

    # PaddleOCR export_model expects prefix without .pdparams:
    #   /path/to/best_accuracy
    preferred = sorted(extract_dir.rglob("best_accuracy.pdparams"))
    if preferred:
        return strip_pdparams_suffix(preferred[0])

    candidates = sorted(extract_dir.rglob("*.pdparams"))
    if not candidates:
        raise ExportError(
            f"Cannot find *.pdparams under {extract_dir}. "
            "The model archive may be corrupted or the layout changed."
        )

    # Prefer shorter/nicer names over optimizer/state leftovers.
    def score(p: Path) -> Tuple[int, int, str]:
        name = p.name.lower()
        penalty = 0
        for bad in ("opt", "optimizer", "state", "states"):
            if bad in name:
                penalty += 10
        if "best" in name:
            penalty -= 5
        return penalty, len(str(p)), str(p)

    return strip_pdparams_suffix(sorted(candidates, key=score)[0])


def strip_pdparams_suffix(path: Path) -> Path:
    text = str(path)
    if text.endswith(".pdparams"):
        return Path(text[: -len(".pdparams")])
    return path


def export_model(
    *,
    python_exe: str,
    repo_dir: Path,
    spec: SRModelSpec,
    checkpoint_prefix: Path,
    export_dir: Path,
    force: bool,
    dry_run: bool,
) -> None:
    model_file = export_dir / "inference.pdmodel"
    params_file = export_dir / "inference.pdiparams"
    if model_file.exists() and params_file.exists() and not force:
        log(f"[export:{spec.key}] inference model already exists: {export_dir}")
        return

    config_path = repo_dir / spec.config_rel
    export_py = repo_dir / "tools" / "export_model.py"
    if not export_py.exists():
        raise ExportError(f"export_model.py not found: {export_py}")
    if not config_path.exists():
        raise ExportError(f"SR config not found in PaddleOCR source: {config_path}")

    safe_mkdir(export_dir)
    cmd = [
        python_exe,
        str(export_py),
        "-c",
        str(config_path),
        "-o",
        f"Global.pretrained_model={checkpoint_prefix}",
        f"Global.save_inference_dir={export_dir}",
    ]

    env = os.environ.copy()
    repo_pythonpath = str(repo_dir)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = repo_pythonpath + (os.pathsep + existing_pythonpath if existing_pythonpath else "")

    log(f"[export:{spec.key}] command:")
    log("  " + " ".join(quote_arg(x) for x in cmd))
    if dry_run:
        return

    proc = subprocess.run(
        cmd,
        cwd=str(repo_dir),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log(proc.stdout)
    if proc.returncode != 0:
        raise ExportError(f"PaddleOCR export failed for {spec.display_name}, returncode={proc.returncode}")

    verify_export_dir(export_dir)


def quote_arg(s: object) -> str:
    text = str(s)
    if " " in text or "\t" in text:
        return '"' + text.replace('"', '\\"') + '"'
    return text


def verify_export_dir(export_dir: Path) -> None:
    expected = [export_dir / "inference.pdmodel", export_dir / "inference.pdiparams"]
    missing = [p for p in expected if not p.exists()]
    if missing:
        raise ExportError(f"Export completed but files are missing: {missing}")
    log(f"[verify] OK: {export_dir}")
    for p in expected:
        log(f"         {p.name}: {p.stat().st_size / 1024:.1f} KB")


def run_smoke_predict(
    *,
    python_exe: str,
    repo_dir: Path,
    spec: SRModelSpec,
    export_dir: Path,
    image_path: Path,
    dry_run: bool,
) -> None:
    predict_py = repo_dir / "tools" / "infer" / "predict_sr.py"
    if not predict_py.exists():
        log(f"[smoke:{spec.key}] skipped: predict_sr.py not found: {predict_py}")
        return
    if not image_path.exists():
        log(f"[smoke:{spec.key}] skipped: image not found: {image_path}")
        return

    cmd = [
        python_exe,
        str(predict_py),
        "--sr_model_dir",
        str(export_dir),
        "--image_dir",
        str(image_path),
        "--sr_image_shape",
        spec.image_shape,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_dir) + (os.pathsep + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")
    log(f"[smoke:{spec.key}] command:")
    log("  " + " ".join(quote_arg(x) for x in cmd))
    if dry_run:
        return
    proc = subprocess.run(cmd, cwd=str(repo_dir), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log(proc.stdout)
    if proc.returncode != 0:
        log(f"[smoke:{spec.key}] failed but export is still usable. returncode={proc.returncode}")


def print_yaml_snippet(exports: Sequence[Tuple[SRModelSpec, Path]]) -> None:
    log("\n[YAML] Add/update these backend entries in detected_tracks_dataset_sr_*.yaml:")
    log("super_resolution:")
    log("  profiles:")
    log("    heavy:")
    log("      backends:")
    for spec, export_dir in exports:
        if spec.key == "telescope":
            name = "paddle_telescope_tbsrn"
        else:
            name = "paddle_gestalt_tsrn"
        log(f"        - name: {name}")
        log("          backend: paddle_sr")
        log(f"          model_dir: \"{export_dir.as_posix()}\"")
        log(f"          image_shape: \"{spec.image_shape}\"")
        log("          use_gpu: false")
        log("          mkldnn: true")
        log("          cpu_threads: 2")
        log("          output_scale_hint: 2.0")


def build_arg_parser() -> argparse.ArgumentParser:
    root = project_root_from_script()
    p = argparse.ArgumentParser(
        description="Download and export PaddleOCR SR checkpoints to inference.pdmodel / inference.pdiparams",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--models", default="both", choices=["telescope", "gestalt", "both"], help="Which SR model(s) to export")
    p.add_argument("--project-root", default=str(root), help="Project root used for relative paths")
    p.add_argument("--out-root", default="models/paddle_sr", help="Output root for exported inference models")
    p.add_argument("--work-dir", default="models/paddle_sr/_work", help="Download/extraction work directory")
    p.add_argument("--repo-dir", default="third_party/PaddleOCR", help="PaddleOCR source checkout directory")
    p.add_argument("--repo-url", default="https://github.com/PaddlePaddle/PaddleOCR.git", help="PaddleOCR git repository URL")
    p.add_argument("--repo-ref", default="main", help="PaddleOCR branch/tag/ref to download")
    p.add_argument("--python", default=sys.executable, help="Python executable used to run PaddleOCR export_model.py")
    p.add_argument("--telescope-checkpoint-prefix", default="", help="Optional explicit Telescope checkpoint prefix without .pdparams")
    p.add_argument("--gestalt-checkpoint-prefix", default="", help="Optional explicit Gestalt checkpoint prefix without .pdparams")
    p.add_argument("--force-repo-download", action="store_true", help="Redownload PaddleOCR source")
    p.add_argument("--force-download", action="store_true", help="Redownload checkpoint archives")
    p.add_argument("--force-extract", action="store_true", help="Re-extract checkpoint archives")
    p.add_argument("--force-export", action="store_true", help="Re-export even if inference files already exist")
    p.add_argument("--skip-paddle-check", action="store_true", help="Do not check import paddle before export")
    p.add_argument("--smoke-image", default="", help="Optional image/crop for tools/infer/predict_sr.py smoke test")
    p.add_argument("--dry-run", action="store_true", help="Print actions without downloading/exporting")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    project_root = Path(args.project_root).expanduser().resolve()
    out_root = resolve_against(project_root, args.out_root)
    work_dir = resolve_against(project_root, args.work_dir)
    repo_dir = resolve_against(project_root, args.repo_dir)
    python_exe = which_python(args.python)
    selected = parse_models(args.models)

    log("[config]")
    log(f"  project_root: {project_root}")
    log(f"  out_root:     {out_root}")
    log(f"  work_dir:     {work_dir}")
    log(f"  repo_dir:     {repo_dir}")
    log(f"  repo_ref:     {args.repo_ref}")
    log(f"  python:       {python_exe}")
    log(f"  models:       {', '.join(s.key for s in selected)}")

    if not args.skip_paddle_check and not args.dry_run:
        verify_paddle_runtime(python_exe)

    repo = ensure_paddleocr_repo(
        repo_dir=repo_dir,
        repo_url=args.repo_url,
        repo_ref=args.repo_ref,
        force_repo_download=bool(args.force_repo_download),
        dry_run=bool(args.dry_run),
    )

    archives_dir = safe_mkdir(work_dir / "archives")
    extracted_root = safe_mkdir(work_dir / "extracted")
    exported: List[Tuple[SRModelSpec, Path]] = []

    for spec in selected:
        log("\n" + "=" * 80)
        log(f"[model:{spec.key}] {spec.display_name}")
        archive = download_model_archive(spec, archives_dir, force=bool(args.force_download), dry_run=bool(args.dry_run))
        extract_dir = extract_archive(
            archive,
            extracted_root / spec.key,
            force=bool(args.force_extract),
            dry_run=bool(args.dry_run),
        )
        explicit_prefix = args.telescope_checkpoint_prefix if spec.key == "telescope" else args.gestalt_checkpoint_prefix
        checkpoint_prefix = find_checkpoint_prefix(extract_dir, explicit_prefix=explicit_prefix)
        export_dir = out_root / spec.export_subdir
        log(f"[model:{spec.key}] checkpoint prefix: {checkpoint_prefix}")
        log(f"[model:{spec.key}] export dir:        {export_dir}")

        export_model(
            python_exe=python_exe,
            repo_dir=repo,
            spec=spec,
            checkpoint_prefix=checkpoint_prefix,
            export_dir=export_dir,
            force=bool(args.force_export),
            dry_run=bool(args.dry_run),
        )
        exported.append((spec, export_dir))

        if args.smoke_image:
            run_smoke_predict(
                python_exe=python_exe,
                repo_dir=repo,
                spec=spec,
                export_dir=export_dir,
                image_path=resolve_against(project_root, args.smoke_image),
                dry_run=bool(args.dry_run),
            )

    print_yaml_snippet(exported)
    log("\n[done] PaddleOCR SR export finished")
    return 0


def resolve_against(root: Path, value: str) -> Path:
    p = Path(value).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (root / p).resolve()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExportError as e:
        log("\n[ERROR] " + str(e))
        raise SystemExit(2)
