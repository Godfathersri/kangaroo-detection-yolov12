import argparse
from pathlib import Path
import csv
from ultralytics import YOLO

def train(model_name, data_yaml, imgsz, epochs, batch, device, project , run_name):
    model = YOLO(model_name)
    results = model.train(
        data=data_yaml,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        device=device,
        amp=True,
        workers=4,
        project=project,
        name= run_name,
        exist_ok=False,
        cos_lr=True,
        patience=30,
    )
    weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n[TRAIN] Best weights: {weights}")
    return str(weights)

def validate(weights, data_yaml, imgsz, device):
    model = YOLO(weights)
    results = model.val(
        data=data_yaml,
        imgsz=imgsz,
        device=device,
        workers=4,
        split="val"
    )
    print(f"\n[VALIDATE] Results: {results}")

def predict(weights, source, outdir, imgsz, conf, device):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "images").mkdir(parents=True, exist_ok=True)

    model = YOLO(weights)
    results = model.predict(
        source=source,
        imgsz=imgsz,
        conf=conf,
        save=True,
        project=str(outdir),
        name="images",
        exist_ok=False,
        device=device
    )

    csv_path = outdir / "detections.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "class", "confidence", "x1", "y1", "x2", "y2"])
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            names = r.names
            for (cid, conf, (x1, y1, x2, y2)) in zip(
                boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()
            ):
                writer.writerow([
                    Path(r.path).name, names[int(cid)], float(conf),
                    float(x1), float(y1), float(x2), float(y2)
                ])
    print(f"[PREDICT] Images -> {outdir/'images'} | CSV -> {csv_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "val", "predict", "all"], default="all")
    ap.add_argument("--model", default="yolov12n.pt")
    ap.add_argument("--data", default="data_sample/augmented/data.yaml")
    ap.add_argument("--weights", default="")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default=0)
    ap.add_argument("--project", default="runs")
    ap.add_argument("--predict_source", default="")
    ap.add_argument("--name", type=str, default="train", help="Run name")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    weights = None
    if args.mode in ("train", "all"):
        weights = train(
            model_name=args.model,
            data_yaml=args.data,
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch=args.batch,
            device=args.device,
            run_name=args.name,
            project=args.project
        )
    if args.mode in ("val", "all"):
        w = weights if weights else f"{args.project}/{args.name}/weights/best.pt"
        validate(
            weights=w,
            data_yaml=args.data,
            imgsz=args.imgsz,
            device=args.device
        )
    if args.mode in ("predict", "all"):
        w = weights if weights else f"{args.project}/{args.name}/weights/best.pt"
        assert args.predict_source, "--predict_source must be set for predict/both"
        predict(
            weights=w,
            source=args.predict_source,
            outdir=args.outdir,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device
        )

if __name__ == "__main__":
    main()
