"""Point selection and command-line output for part4a.py."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from part4a import (load_image, save_image, computeH, homography_system,
                    transform, warp_image, mosaic)


def collect_points(images, count):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(images), squeeze=False)
    selected = [[] for _ in images]
    for ax, image in zip(axes[0], images):
        ax.imshow(image)
    fig.suptitle('Alternate matching clicks left/right; right-click undoes last click. Close to save.')
    history = []

    def click(event):
        if event.button == 3 and history:
            i, artist, label = history.pop()
            selected[i].pop(); artist.remove(); label.remove()
        elif event.button == 1 and event.inaxes in axes[0]:
            i = list(axes[0]).index(event.inaxes)
            if len(selected[i]) >= count:
                return
            point = [float(event.xdata), float(event.ydata)]
            selected[i].append(point)
            artist, = event.inaxes.plot(*point, 'r+')
            label = event.inaxes.text(*point, str(len(selected[i])), color='red')
            history.append((i, artist, label))
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', click)
    plt.show()
    if any(len(points) != count for points in selected):
        raise ValueError(f'Need exactly {count} points in each image; nothing saved.')
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-side', type=int, default=1200)
    sub = parser.add_subparsers(dest='command', required=True)
    pick = sub.add_parser('pick', help='Save manually clicked corresponding points')
    pick.add_argument('source'); pick.add_argument('reference')
    pick.add_argument('output'); pick.add_argument('--count', type=int, default=12)
    run = sub.add_parser('pair', help='Estimate H, visualize points, and compare warps')
    run.add_argument('points'); run.add_argument('output')
    rect = sub.add_parser('rectify', help='Click TL,TR,BR,BL of a rectangle')
    rect.add_argument('image'); rect.add_argument('output')
    rect.add_argument('--width', type=int, default=600)
    rect.add_argument('--height', type=int, default=400)
    mos = sub.add_parser('mosaic', help='Blend pairs sharing the same reference')
    mos.add_argument('output'); mos.add_argument('points', nargs='+')
    args = parser.parse_args()
    if args.max_side < 2:
        parser.error('--max-side must be at least 2')
    if args.command == 'pick':
        paths = [str(Path(p).resolve()) for p in (args.source, args.reference)]
        images = [load_image(p, args.max_side) for p in paths]
        src, dst = collect_points(images, args.count)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(dict(source=paths[0], reference=paths[1],
            max_side=args.max_side, source_points=src, reference_points=dst), indent=2))
        return
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    if args.command == 'mosaic':
        records = [json.loads(Path(p).read_text()) for p in args.points]
        first = records[0]
        if any((r['reference'], r['max_side']) != (first['reference'], first['max_side']) for r in records):
            raise ValueError('All pairs must share a reference and resize setting.')
        images = [load_image(first['reference'], first['max_side'])]
        matrices = [np.eye(3)]
        for record in records:
            images.append(load_image(record['source'], record['max_side']))
            matrices.append(computeH(record['source_points'], record['reference_points']))
        result, mask = mosaic(images, matrices)
        save_image(out/'mosaic.png', result)
        save_image(out/'mask.png', mask.astype(float))
        np.save(out/'homographies.npy', matrices)
        return
    if args.command == 'rectify':
        if min(args.width, args.height) < 2:
            parser.error('Rectangle dimensions must be at least 2')
        image = load_image(args.image, args.max_side)
        print('Click top-left, top-right, bottom-right, bottom-left; close window.')
        src = collect_points([image], 4)[0]
        dst = [[0, 0], [args.width-1, 0], [args.width-1, args.height-1], [0, args.height-1]]
        reference = None
        bounds = (0, 0, args.width-1, args.height-1)
        (out/'points.json').write_text(json.dumps(dict(source=str(Path(args.image).resolve()),
            max_side=args.max_side, source_points=src, reference_points=dst), indent=2))
    else:
        record = json.loads(Path(args.points).read_text())
        image = load_image(record['source'], record['max_side'])
        reference = load_image(record['reference'], record['max_side'])
        src, dst = record['source_points'], record['reference_points']
        bounds = None
    H = computeH(src, dst)
    A, b = homography_system(src, dst)
    np.savetxt(out/'H.txt', H); np.savez(out/'system.npz', A=A, b=b)
    errors = np.linalg.norm(transform(src, H)-dst, axis=1)
    report = dict(H=H.tolist(), reprojection_rmse_pixels=float(np.sqrt(np.mean(errors**2))),
                  point_errors_pixels=errors.tolist())
    for method in ('nearest', 'bilinear'):
        start = time.perf_counter()
        warped, mask, origin = warp_image(image, H, method, bounds)
        report[method+'_seconds'] = time.perf_counter()-start
        report['origin_xy'] = origin
        save_image(out/(method+'.png'), warped)
        save_image(out/(method+'_mask.png'), mask.astype(float))
    (out/'report.json').write_text(json.dumps(report, indent=2))
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, im, pts, title in zip(axes, [image, reference], [src, dst], ['Source', 'Reference / target']):
        if im is not None:
            ax.imshow(im)
        ax.scatter(*np.asarray(pts).T, c='red', marker='+')
        for i, (x, y) in enumerate(pts, 1):
            ax.text(x, y, str(i), color='red')
        ax.set_title(title)
        ax.set_aspect('equal')
    fig.savefig(out/'correspondences.png', dpi=160, bbox_inches='tight'); plt.close(fig)
    print(json.dumps(report, indent=2))

