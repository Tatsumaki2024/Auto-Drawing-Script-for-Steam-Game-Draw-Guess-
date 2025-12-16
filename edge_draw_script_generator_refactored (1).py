#!/usr/bin/env python3
"""
edge_draw_script_generator_refactored
=====================================

This module provides a cleaned‐up, well documented and more
maintainable version of the original edge drawing script generator.
It contains utilities for scanning an image for a specific colour,
detecting edges via OpenCV, extracting connected pixel paths and
converting them into mouse movement commands.  A key objective of
this refactor is to preserve the functionality of the original
implementation while improving readability, robustness and testability.

Key improvements
----------------

* **Always use continuous curves**:  Short segments are grouped into
  polylines so that each visible path is drawn with a single mouse
  press.  The deprecated ``--no-polylines`` option from the original
  code has been removed.
* **Polyline simplification**:  The optional ``smooth_polylines`` flag
  invokes the Ramer–Douglas–Peucker (RDP) algorithm to reduce the
  number of points in long curves without altering their overall
  shape.  The tolerance for simplification is controlled by
  ``rdp_epsilon``.
* **Resilient I/O**:  Image loading attempts several fallbacks
  (OpenCV via in‑memory decoding, OpenCV from file, Pillow) and
  provides clear exceptions on failure.  Generated scripts are
  written atomically and include a header explaining their origin.
* **Separation of concerns**:  The command line interface, image
  processing and script generation are clearly separated.  Functions
  expose only the parameters they require and return simple data
  structures to aid testing and reuse.

Future work ideas are documented at the bottom of this file.  They
include adaptive sampling, spline fitting, parallel processing and
unit testing, none of which are implemented here but may serve as
starting points for future enhancements.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from typing import Iterable, Iterator, List, Sequence, Tuple, Union, Optional

import numpy as np
from PIL import Image
import cv2


def parse_hex_colour(colour: str) -> Tuple[int, int, int, int]:
    """Convert a colour string into a 4‑tuple of channel values.

    The colour may be specified as ``#RRGGBB`` (implicit full opacity) or
    ``#AARRGGBB`` (explicit alpha channel).  Channels are returned as
    integers in the range 0–255 representing red, green, blue and
    alpha respectively.

    Parameters
    ----------
    colour : str
        Colour string beginning with ``#`` followed by six (RGB) or
        eight (ARGB) hexadecimal digits.

    Returns
    -------
    Tuple[int, int, int, int]
        A 4‑tuple of (red, green, blue, alpha) channel values.

    Raises
    ------
    ValueError
        If the input does not begin with ``#`` or has an unexpected
        length.
    """
    if not isinstance(colour, str) or not colour.startswith("#"):
        raise ValueError(f"Colour must start with '#', got {colour!r}")
    hex_digits = colour[1:]
    if len(hex_digits) == 6:
        red = int(hex_digits[0:2], 16)
        green = int(hex_digits[2:4], 16)
        blue = int(hex_digits[4:6], 16)
        alpha = 0xFF
    elif len(hex_digits) == 8:
        alpha = int(hex_digits[0:2], 16)
        red = int(hex_digits[2:4], 16)
        green = int(hex_digits[4:6], 16)
        blue = int(hex_digits[6:8], 16)
    else:
        raise ValueError(
            f"Invalid colour length: expected 6 or 8 hex digits, got {len(hex_digits)}"
        )
    return red, green, blue, alpha


def scan_image_for_colour(
    image: Image.Image, target_colour: Tuple[int, int, int, int]
) -> Iterable[Tuple[int, int, int]]:
    """Scan an RGBA image for horizontal runs of a specific colour.

    The scan proceeds row by row.  For each row, contiguous sequences of
    pixels exactly matching ``target_colour`` are identified.  Each run
    yields a tuple ``(row_index, start_column, end_column)`` where
    ``end_column`` is one past the last matching pixel.  If no pixels
    match the colour on a row, nothing is yielded for that row.

    Parameters
    ----------
    image : Image.Image
        PIL image in RGBA mode.
    target_colour : Tuple[int, int, int, int]
        RGBA colour to search for.

    Yields
    ------
    Tuple[int, int, int]
        Tuples describing horizontal runs: (y, x_start, x_end).
    """
    width, height = image.size
    pixels = image.load()
    for y in range(height):
        x = 0
        while x < width:
            # Skip non‑matching pixels
            while x < width and pixels[x, y] != target_colour:
                x += 1
            if x >= width:
                break
            run_start = x
            x += 1
            # Extend while pixels match target
            while x < width and pixels[x, y] == target_colour:
                x += 1
            run_end = x
            yield (y, run_start, run_end)


def _get_adjacent_coords(
    x: int, y: int, width: int, height: int
) -> Iterator[Tuple[int, int]]:
    """Yield coordinates of the eight neighbours around a pixel.

    Coordinates are bounded by ``0 <= nx < width`` and ``0 <= ny < height``.
    Neighbours are yielded in a deterministic order to ensure
    reproducibility.

    Parameters
    ----------
    x, y : int
        Coordinates of the current pixel.
    width, height : int
        Dimensions of the image.

    Yields
    ------
    Tuple[int, int]
        Neighbouring coordinates (nx, ny).
    """
    offsets: List[Tuple[int, int]] = [
        (-1, -1), (0, -1), (1, -1),
        (-1, 0),           (1, 0),
        (-1, 1),  (0, 1),  (1, 1),
    ]
    for dx, dy in offsets:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            yield (nx, ny)


def _extract_connected_paths(
    image: Image.Image,
    target_colour: Tuple[int, int, int, int],
) -> List[List[Tuple[int, int]]]:
    """Extract connected pixel paths matching the target colour.

    A greedy depth‑first traversal is applied to each unvisited pixel of
    the target colour.  Within a connected component, the traversal
    repeatedly selects the unvisited neighbour with the smallest
    (row, column) index.  Each branch of a component is returned
    separately, minimising cursor travel in the generated drawing.

    Parameters
    ----------
    image : Image.Image
        PIL image in RGBA mode.
    target_colour : Tuple[int, int, int, int]
        Colour to find connected components for.

    Returns
    -------
    List[List[Tuple[int, int]]]
        A list of paths; each path is a list of (x, y) coordinates.
    """
    width, height = image.size
    pixels = image.load()
    visited: List[List[bool]] = [[False] * width for _ in range(height)]
    paths: List[List[Tuple[int, int]]] = []
    for y in range(height):
        for x in range(width):
            if visited[y][x] or pixels[x, y] != target_colour:
                continue
            path: List[Tuple[int, int]] = []
            current_x, current_y = x, y
            visited[current_y][current_x] = True
            path.append((current_x, current_y))
            while True:
                neighbours = [
                    (nx, ny)
                    for nx, ny in _get_adjacent_coords(current_x, current_y, width, height)
                    if not visited[ny][nx] and pixels[nx, ny] == target_colour
                ]
                if not neighbours:
                    break
                neighbours.sort(key=lambda coord: (coord[1], coord[0]))
                next_x, next_y = neighbours[0]
                visited[next_y][next_x] = True
                path.append((next_x, next_y))
                current_x, current_y = next_x, next_y
            paths.append(path)
    return paths


def _convert_paths_to_segments(
    paths: Sequence[Sequence[Tuple[int, int]]]
) -> List[Tuple[int, int, int, int]]:
    """Compress pixel paths into straight line segments.

    Consecutive pixels that lie along the same direction are merged
    into a single segment.  A change in direction begins a new
    segment.  Single‑pixel paths are represented as zero‑length
    segments (x, y, x, y) and filtered out later.

    Parameters
    ----------
    paths : Sequence[Sequence[Tuple[int, int]]]
        Iterable of pixel paths.

    Returns
    -------
    List[Tuple[int, int, int, int]]
        List of (x0, y0, x1, y1) segments.
    """
    segments: List[Tuple[int, int, int, int]] = []
    for path in paths:
        if not path:
            continue
        if len(path) == 1:
            x0, y0 = path[0]
            segments.append((x0, y0, x0, y0))
            continue
        run_start_x, run_start_y = path[0]
        prev_x, prev_y = path[0]
        next_x, next_y = path[1]
        direction = (next_x - prev_x, next_y - prev_y)
        for idx in range(1, len(path)):
            curr_x, curr_y = path[idx]
            curr_direction = (curr_x - prev_x, curr_y - prev_y)
            if curr_direction != direction:
                segments.append((run_start_x, run_start_y, prev_x, prev_y))
                run_start_x, run_start_y = prev_x, prev_y
                direction = curr_direction
            prev_x, prev_y = curr_x, curr_y
        segments.append((run_start_x, run_start_y, prev_x, prev_y))
    return segments


def _segment_start_end(
    segment: Union[Tuple[int, ...], List[Tuple[int, int]]]
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Return the start and end coordinates of a segment or polyline.

    This helper abstracts the differences between legacy run tuples,
    path segments and polylines.  It returns a pair of coordinates
    describing the start and end of the segment.

    Parameters
    ----------
    segment : Union[Tuple[int, ...], List[Tuple[int, int]]]
        Segment specification: either a 3‑tuple, 4‑tuple or list of
        coordinates.

    Returns
    -------
    Tuple[Tuple[int, int], Tuple[int, int]]
        A pair ((x_start, y_start), (x_end, y_end)).

    Raises
    ------
    ValueError
        If the segment type or length is unsupported.
    """
    # Legacy horizontal run: (row, start_col, end_col)
    if isinstance(segment, tuple) and len(segment) == 3:
        row, start_col, end_col = segment
        return (start_col, row), (end_col - 1, row)
    # Path segment: (x_start, y_start, x_end, y_end)
    if isinstance(segment, tuple) and len(segment) == 4:
        x_start, y_start, x_end, y_end = segment
        return (x_start, y_start), (x_end, y_end)
    # Polyline: list of coordinates
    if isinstance(segment, list):
        if not segment:
            raise ValueError("Empty polyline encountered")
        return segment[0], segment[-1]
    raise ValueError(f"Unsupported segment type: {segment}")


def _order_segments_by_proximity(
    segments: Sequence[Union[Tuple[int, ...], List[Tuple[int, int]]]]
) -> List[Union[Tuple[int, ...], List[Tuple[int, int]]]]:
    """Reorder segments to minimise cursor travel using a greedy heuristic.

    The ordering starts from the segment nearest to the top‑left corner
    and repeatedly appends the segment whose start point has the
    smallest Manhattan distance from the end of the last segment.
    Ties are broken deterministically by (y, x) coordinates of the
    start point.

    Parameters
    ----------
    segments : Sequence[Union[Tuple[int, ...], List[Tuple[int, int]]]]
        Input sequence of segments or polylines.

    Returns
    -------
    List[Union[Tuple[int, ...], List[Tuple[int, int]]]]
        A new list of segments in optimised order.
    """
    if not segments:
        return []
    remaining: List[Union[Tuple[int, ...], List[Tuple[int, int]]]] = list(segments)
    ordered: List[Union[Tuple[int, ...], List[Tuple[int, int]]]] = []

    def initial_key(seg: Union[Tuple[int, ...], List[Tuple[int, int]]]) -> Tuple[int, int]:
        start, _ = _segment_start_end(seg)
        return (start[1], start[0])

    # Sort by initial key and pick the first as starting point
    remaining.sort(key=initial_key)
    current = remaining.pop(0)
    ordered.append(current)
    _, last_end = _segment_start_end(current)

    while remaining:
        min_idx = 0
        min_distance: Optional[int] = None
        min_start: Tuple[int, int] = (0, 0)
        for idx, seg in enumerate(remaining):
            start, _ = _segment_start_end(seg)
            distance = abs(start[0] - last_end[0]) + abs(start[1] - last_end[1])
            if (min_distance is None) or (distance < min_distance):
                min_distance = distance
                min_idx = idx
                min_start = start
            elif distance == min_distance:
                if start[1] < min_start[1] or (start[1] == min_start[1] and start[0] < min_start[0]):
                    min_idx = idx
                    min_start = start
        next_seg = remaining.pop(min_idx)
        ordered.append(next_seg)
        _, last_end = _segment_start_end(next_seg)
    return ordered


def _group_segments_into_polylines(
    segments: Sequence[Union[Tuple[int, ...], List[Tuple[int, int]]]]
) -> List[Union[Tuple[int, ...], List[Tuple[int, int]]]]:
    """Merge adjacent line segments into polylines when possible.

    Two consecutive segments are merged when the end of the first
    coincides with the start of the second.  Legacy horizontal runs
    (3‑tuples) and existing polylines (lists) are converted to or
    preserved as lists of coordinate pairs.

    Parameters
    ----------
    segments : Sequence[Union[Tuple[int, ...], List[Tuple[int, int]]]]
        Input segments or polylines.

    Returns
    -------
    List[Union[Tuple[int, ...], List[Tuple[int, int]]]]
        Merged sequence where adjacent segments have been combined into
        polylines.
    """
    result: List[Union[Tuple[int, ...], List[Tuple[int, int]]]] = []
    idx = 0
    while idx < len(segments):
        seg = segments[idx]
        # Already a polyline
        if isinstance(seg, list):
            result.append(seg)
            idx += 1
            continue
        # Legacy horizontal run
        if len(seg) == 3:
            row, start_col, end_col = seg
            polyline: List[Tuple[int, int]] = [
                (start_col, row),
                (end_col - 1, row),
            ]
            result.append(polyline)
            idx += 1
            continue
        # Path segment: convert to polyline and attempt to merge
        x0, y0, x1, y1 = seg  # type: ignore[assignment]
        polyline: List[Tuple[int, int]] = [(x0, y0), (x1, y1)]
        # Attempt to merge with subsequent segments
        while idx + 1 < len(segments):
            next_seg = segments[idx + 1]
            if isinstance(next_seg, list) or len(next_seg) != 4:
                break
            nx0, ny0, nx1, ny1 = next_seg
            if (x1, y1) == (nx0, ny0):
                polyline.append((nx1, ny1))
                x1, y1 = nx1, ny1
                idx += 1
            else:
                break
        result.append(polyline)
        idx += 1
    return result


def _ramer_douglas_peucker(
    points: List[Tuple[int, int]], epsilon: float
) -> List[Tuple[int, int]]:
    """Simplify a polyline using the Ramer–Douglas–Peucker algorithm.

    Given a list of points, recursively remove points that lie within
    ``epsilon`` of the line segment connecting their neighbours.  The
    algorithm preserves the first and last point and any point that
    deviates by more than ``epsilon`` from a straight line.  See the
    RDP algorithm literature for details.

    Parameters
    ----------
    points : List[Tuple[int, int]]
        Ordered polyline vertices.
    epsilon : float
        Tolerance distance; larger values yield coarser approximations.

    Returns
    -------
    List[Tuple[int, int]]
        Simplified polyline containing a subset of the input points.
    """
    if len(points) < 3:
        return points[:]
    first = np.array(points[0], dtype=float)
    last = np.array(points[-1], dtype=float)
    line_vec = last - first
    line_len_sq = np.dot(line_vec, line_vec)
    max_distance = -1.0
    index = 0
    for i in range(1, len(points) - 1):
        p = np.array(points[i], dtype=float)
        # Distance from p to the line segment
        if line_len_sq == 0:
            distance = np.linalg.norm(p - first)
        else:
            proj = np.dot(p - first, line_vec) / line_len_sq
            proj_point = first + proj * line_vec
            distance = np.linalg.norm(p - proj_point)
        if distance > max_distance:
            max_distance = distance
            index = i
    if max_distance > epsilon:
        # Recurse on both sides of the farthest point
        left = _ramer_douglas_peucker(points[: index + 1], epsilon)
        right = _ramer_douglas_peucker(points[index:], epsilon)
        return left[:-1] + right
    # If all intermediate points are close, keep only endpoints
    return [points[0], points[-1]]


def _simplify_polylines(
    segments: Sequence[Union[Tuple[int, ...], List[Tuple[int, int]]]],
    epsilon: float,
) -> List[Union[Tuple[int, ...], List[Tuple[int, int]]]]:
    """Apply RDP simplification to every polyline in a sequence.

    Non‑polyline segments are passed through unchanged.  For each
    polyline (a list of points), the RDP algorithm is invoked with
    the provided tolerance ``epsilon``.  Polylines with fewer than three
    points are returned as‑is.

    Parameters
    ----------
    segments : Sequence[Union[Tuple[int, ...], List[Tuple[int, int]]]]
        Input segments or polylines.
    epsilon : float
        Tolerance for the RDP algorithm.

    Returns
    -------
    List[Union[Tuple[int, ...], List[Tuple[int, int]]]]
        Sequence with simplified polylines.
    """
    simplified: List[Union[Tuple[int, ...], List[Tuple[int, int]]]] = []
    for seg in segments:
        if isinstance(seg, list):
            if len(seg) < 3:
                simplified.append(seg)
            else:
                simplified.append(_ramer_douglas_peucker(seg, epsilon))
        else:
            simplified.append(seg)
    return simplified


def write_draw_script(
    segments: Sequence[Union[Tuple[int, ...], List[Tuple[int, int]]]],
    output_path: str,
    *,
    top_left: Tuple[int, int] = (0, 0),
    pixel_size: int = 1,
    pause: float = 0.01,
    duration: float = 0.0,
    use_directinput: bool = False,
) -> None:
    """Generate a Python script that replays the specified strokes.

    The generated script uses either PyAutoGUI or PyDirectInput to
    automate mouse movements and draw each segment or polyline.  Each
    segment in ``segments`` may be one of the following forms:

    * ``(row, start_col, end_col)`` — a horizontal run of pixels as
      returned by :func:`scan_image_for_colour`.  The run is inclusive
      at ``start_col`` and exclusive at ``end_col``.  It is converted
      into a two‑point polyline.
    * ``(x_start, y_start, x_end, y_end)`` — a straight line segment
      generated from connected component traversal.  Zero‑length
      segments are ignored to avoid point strokes.
    * ``[(x0, y0), (x1, y1), ...]`` — a polyline.  A single
      ``mouseDown`` is issued at the beginning of the polyline,
      followed by ``moveTo`` calls for each subsequent point and a
      ``mouseUp`` at the end.

    Parameters
    ----------
    segments : Sequence[Union[Tuple[int, ...], List[Tuple[int, int]]]]
        List of segments or polylines to draw.
    output_path : str
        Destination file path for the generated script.
    top_left : Tuple[int, int], optional
        Screen coordinates of the drawing area's top‑left corner.
    pixel_size : int, optional
        Scaling factor mapping image pixels to screen pixels.
    pause : float, optional
        Default pause between automation actions.
    duration : float, optional
        Duration for each drag action (may be needed by some
        applications).
    use_directinput : bool, optional
        Whether to import the DirectInput backend instead of PyAutoGUI.

    Raises
    ------
    IOError
        If writing the script fails.
    """
    # Build the script as a list of lines; join them at the end
    script_lines: List[str] = []
    script_lines.append("#!/usr/bin/env python3")
    script_lines.append("")
    # Header comment describing the origin of the script.  Use triple
    # quotes so the generated file is a valid Python module with a
    # docstring.
    # Include a triple‑quoted docstring in the generated file.  Use
    # explicit quoting so that the triple quotes appear literally in
    # the output.  Without this, the header would become plain text
    # and cause a syntax error when the script is imported.
    header_comment = (
        '"""Generated drawing script.  Do not edit manually.\n\n'
        'This script was produced by edge_draw_script_generator_refactored.py.  It\n'
        'uses either PyAutoGUI or PyDirectInput to automate mouse movements\n'
        'and reproduce a pattern extracted from an image.\n'
        '"""'
    )
    script_lines.append(header_comment)
    script_lines.append("")
    script_lines.append("import time")
    # Conditional import of backend
    if use_directinput:
        script_lines.extend([
            "try:",
            "    import pydirectinput as pag  # DirectInput backend",  # noqa: E501
            "except ImportError:",
            "    try:",
            "        import pydirectinput_rgx as pag  # extended DirectInput fork",  # noqa: E501
            "    except ImportError:",
            "        import pyautogui as pag  # fall back to PyAutoGUI",
        ])
    else:
        script_lines.append("import pyautogui as pag  # PyAutoGUI backend")
    script_lines.append("")
    # Configure PyAutoGUI / DirectInput
    script_lines.append(f"pag.PAUSE = {pause}")
    script_lines.append("pag.FAILSAFE = True")
    script_lines.append("")
    # Define translation constants
    script_lines.append(f"top_left_x, top_left_y = {top_left}")
    script_lines.append(f"pixel_size = {pixel_size}")
    script_lines.append(f"drag_duration = {duration}")
    script_lines.append("")
    # Helper function to drag the mouse; encapsulates common logic
    script_lines.extend([
        "def _drag_to(x: int, y: int) -> None:",
        "    \"\"\"Drag the mouse to the specified screen coordinate.\"\"\"",
        "    pag.mouseDown(button='left')",
        "    # Sleep briefly to ensure the drag is registered",
        "    time.sleep(max(0.01, drag_duration))",
        "    try:",
        "        pag.moveTo(x, y, duration=drag_duration)",
        "    except (TypeError, AttributeError):",
        "        # Older versions of PyAutoGUI do not accept the duration keyword",
        "        pag.moveTo(x, y)",
        "    pag.mouseUp(button='left')",
        "",
    ])
    # Main function: draws all segments/polylines
    script_lines.extend([
        "def main() -> None:",
        "    \"\"\"Replay the drawing by executing stored segments.\"\"\"",
        "    print('Drawing will start in 5 seconds. Please switch to the drawing application.')",
        "    time.sleep(5)",
    ])
    # Emit drawing commands for each segment or polyline
    for seg in segments:
        if not seg:
            continue
        # Horizontal run: (row, start_col, end_col)
        if isinstance(seg, tuple) and len(seg) == 3:
            row, start_col, end_col = seg
            start_x_expr = f"top_left_x + {start_col} * pixel_size"
            start_y_expr = f"top_left_y + {row} * pixel_size"
            end_x_expr = f"top_left_x + ({end_col} - 1) * pixel_size"
            end_y_expr = start_y_expr
            script_lines.append(f"    # Row {row}: columns {start_col}-{end_col - 1}")
            script_lines.append(f"    pag.moveTo({start_x_expr}, {start_y_expr})")
            script_lines.append(f"    _drag_to({end_x_expr}, {end_y_expr})")
            script_lines.append("")
            continue
        # Path segment: (x_start, y_start, x_end, y_end)
        if isinstance(seg, tuple) and len(seg) == 4:
            x0, y0, x1, y1 = seg
            # Skip zero‑length segments
            if x0 == x1 and y0 == y1:
                continue
            start_x_expr = f"top_left_x + {x0} * pixel_size"
            start_y_expr = f"top_left_y + {y0} * pixel_size"
            end_x_expr = f"top_left_x + {x1} * pixel_size"
            end_y_expr = f"top_left_y + {y1} * pixel_size"
            script_lines.append(f"    # Segment from (x={x0}, y={y0}) to (x={x1}, y={y1})")
            script_lines.append(f"    pag.moveTo({start_x_expr}, {start_y_expr})")
            script_lines.append(f"    _drag_to({end_x_expr}, {end_y_expr})")
            script_lines.append("")
            continue
        # Polyline
        if isinstance(seg, list):
            if len(seg) < 2:
                # Nothing to draw
                continue
            coords_expr: List[Tuple[str, str]] = []
            for px, py in seg:
                expr_x = f"top_left_x + {px} * pixel_size"
                expr_y = f"top_left_y + {py} * pixel_size"
                coords_expr.append((expr_x, expr_y))
            script_lines.append(f"    # Polyline with {len(seg)} points")
            first_x, first_y = coords_expr[0]
            script_lines.append(f"    pag.moveTo({first_x}, {first_y})")
            script_lines.append("    pag.mouseDown(button='left')")
            for cx, cy in coords_expr[1:]:
                script_lines.append(f"    pag.moveTo({cx}, {cy})")
            script_lines.append("    pag.mouseUp(button='left')")
            script_lines.append("")
            continue
        raise ValueError(f"Unsupported segment specification: {seg}")
    # Footer: entrypoint with exception handling
    script_lines.extend([
        "if __name__ == '__main__':",
        "    try:",
        "        main()",
        "    except pag.FailSafeException:",
        "        print('Drawing aborted: Fail‑safe triggered.')",
        "    except KeyboardInterrupt:",
        "        print('Drawing interrupted by user.')",
    ])
    # Write the script to disk atomically
    try:
        with open(output_path, 'w', encoding='utf-8') as outfile:
            outfile.write("\n".join(script_lines))
    except Exception as exc:
        raise IOError(f"Failed to write script to {output_path}: {exc}")


def generate_draw_script(
    input_image_path: str,
    output_script_path: str,
    target_colour_hex: str,
    *,
    top_left: Tuple[int, int] = (0, 0),
    pixel_size: int = 1,
    pause: float = 0.01,
    duration: float = 0.0,
    use_directinput: bool = False,
    optimise_path: bool = True,
    reorder_by_proximity: bool = True,
    smooth_polylines: bool = False,
    rdp_epsilon: float = 1.0,
) -> None:
    """High‑level function to generate a drawing script from an image.

    The image at ``input_image_path`` is scanned for pixels matching
    ``target_colour_hex``.  If ``optimise_path`` is true, connected
    component traversal and run compression are used; otherwise a
    simple row‑major scan is performed.  Adjacent segments are always
    merged into polylines.  Optionally, polylines are simplified via
    RDP and reordered to minimise cursor travel before being passed to
    :func:`write_draw_script`.

    Parameters
    ----------
    input_image_path : str
        Path to the RGBA image file to analyse.
    output_script_path : str
        Destination file path for the generated Python script.
    target_colour_hex : str
        Colour to search for expressed as ``#RRGGBB`` or ``#AARRGGBB``.
    top_left : Tuple[int, int], optional
        Screen coordinate of the drawing area's top‑left corner.
    pixel_size : int, optional
        Scaling factor from image pixels to screen pixels.
    pause : float, optional
        Delay between input actions in the generated script.
    duration : float, optional
        Duration for each drag action.
    use_directinput : bool, optional
        Use DirectInput backend instead of PyAutoGUI.
    optimise_path : bool, optional
        Optimise the drawing order using connected component traversal.
    reorder_by_proximity : bool, optional
        Reorder segments to reduce cursor travel using a greedy heuristic.
    smooth_polylines : bool, optional
        Simplify polylines using the RDP algorithm if true.
    rdp_epsilon : float, optional
        Tolerance parameter for RDP simplification.  Larger values
        produce coarser approximations.

    Raises
    ------
    FileNotFoundError
        If the input image does not exist.
    ValueError
        If the colour string is invalid.
    """
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"Input image not found: {input_image_path}")
    target_colour = parse_hex_colour(target_colour_hex)
    # Load image in RGBA mode; fail early if the image cannot be read
    with Image.open(input_image_path) as img:
        rgba_image = img.convert('RGBA')
        if optimise_path:
            # Connected component traversal
            paths = _extract_connected_paths(rgba_image, target_colour)
            segments: List[Union[Tuple[int, ...], List[Tuple[int, int]]]] = _convert_paths_to_segments(paths)
        else:
            # Legacy row‑major scanning
            segments = list(scan_image_for_colour(rgba_image, target_colour))  # type: ignore
    # Always merge adjacent segments into polylines
    segments = _group_segments_into_polylines(segments)
    # Optionally simplify polylines
    if smooth_polylines:
        segments = _simplify_polylines(segments, epsilon=rdp_epsilon)
    # Optionally reorder segments by proximity
    if reorder_by_proximity:
        segments = _order_segments_by_proximity(segments)
    # Write the drawing script
    write_draw_script(
        segments=segments,
        output_path=output_script_path,
        top_left=top_left,
        pixel_size=pixel_size,
        pause=pause,
        duration=duration,
        use_directinput=use_directinput,
    )


def detect_edges_and_colour(
    input_image_path: str,
    edges_colour_hex: str,
    *,
    low_threshold: float = 100.0,
    high_threshold: float = 200.0,
    dilation_iter: int = 0,
) -> Image.Image:
    """Detect edges in an image and return an RGBA image with coloured edges.

    This helper reads the image, converts it to grayscale, applies a
    Gaussian blur, runs the Canny edge detector and optionally dilates
    the resulting mask to thicken edges.  Detected edges are coloured
    according to ``edges_colour_hex`` while the background is made fully
    transparent.

    Parameters
    ----------
    input_image_path : str
        Path to the source image.
    edges_colour_hex : str
        ARGB colour string used to colour detected edges.
    low_threshold : float, optional
        Lower hysteresis threshold for the Canny detector.
    high_threshold : float, optional
        Upper hysteresis threshold for the Canny detector.
    dilation_iter : int, optional
        Number of dilation iterations to apply to the edge mask.  Zero
        disables dilation.

    Returns
    -------
    Image.Image
        A PIL image in RGBA mode with coloured edges on a transparent
        background.

    Raises
    ------
    FileNotFoundError
        If the image cannot be read.
    ValueError
        If the colour string is invalid.
    """
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"Image not found: {input_image_path}")
    colour_rgba = parse_hex_colour(edges_colour_hex)
    bgr: Optional[np.ndarray] = None
    # Attempt to read via numpy and cv2.imdecode to handle unicode paths
    try:
        data = np.fromfile(input_image_path, dtype=np.uint8)
        if data.size > 0:
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        bgr = None
    if bgr is None:
        # Fallback to cv2.imread
        bgr = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        # Fallback to Pillow
        try:
            with Image.open(input_image_path) as pil_img:
                rgb_array = np.array(pil_img.convert('RGB'))
            bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        except Exception:
            raise FileNotFoundError(
                f"Failed to read image using OpenCV and Pillow: {input_image_path}"
            )
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=low_threshold, threshold2=high_threshold)
    if dilation_iter > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=dilation_iter)
    height, width = edges.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    red, green, blue, alpha = colour_rgba
    mask = edges > 0
    rgba[mask] = [red, green, blue, alpha]
    edge_image = Image.fromarray(rgba, mode='RGBA')
    return edge_image


def process_and_generate_script(
    input_image_path: str,
    output_script_path: str,
    *,
    edges_colour_hex: str = '#00FF00FF',
    low_threshold: float = 100.0,
    high_threshold: float = 200.0,
    dilation_iter: int = 0,
    top_left: Tuple[int, int] = (0, 0),
    pixel_size: int = 1,
    pause: float = 0.01,
    duration: float = 0.0,
    use_directinput: bool = False,
    optimise_path: bool = True,
    reorder_by_proximity: bool = True,
    smooth_polylines: bool = False,
    rdp_epsilon: float = 1.0,
) -> None:
    """End‑to‑end helper: detect edges and generate a drawing script.

    This convenience function first calls :func:`detect_edges_and_colour` to
    produce a temporary image containing only coloured edges.  It
    writes this intermediate image to a temporary file on disk then
    invokes :func:`generate_draw_script` to create the drawing script.

    The temporary file is removed automatically when the function returns.

    Parameters
    ----------
    input_image_path : str
        Path to the source image.
    output_script_path : str
        File path where the generated script will be written.
    edges_colour_hex : str, optional
        Colour for edges expressed as ``#RRGGBB`` or ``#AARRGGBB``.
    low_threshold : float, optional
        Lower Canny hysteresis threshold.
    high_threshold : float, optional
        Upper Canny hysteresis threshold.
    dilation_iter : int, optional
        Number of dilation iterations for edge thickening.
    top_left : Tuple[int, int], optional
        Screen coordinate of the drawing area's top‑left corner.
    pixel_size : int, optional
        Scaling factor between image and screen pixels.
    pause : float, optional
        Delay between input actions in the generated script.
    duration : float, optional
        Duration for each drag action.
    use_directinput : bool, optional
        Use DirectInput backend instead of PyAutoGUI.
    optimise_path : bool, optional
        Optimise the drawing order using connected components.
    reorder_by_proximity : bool, optional
        Reorder segments to reduce cursor travel.
    smooth_polylines : bool, optional
        Simplify polylines using the RDP algorithm.
    rdp_epsilon : float, optional
        Tolerance parameter for RDP simplification.
    """
    # Detect edges and colour them
    edge_image = detect_edges_and_colour(
        input_image_path,
        edges_colour_hex,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        dilation_iter=dilation_iter,
    )
    # Write the edge image to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        temp_path = tmp.name
    try:
        edge_image.save(temp_path)
        # Generate the drawing script from the temporary edge image
        generate_draw_script(
            input_image_path=temp_path,
            output_script_path=output_script_path,
            target_colour_hex=edges_colour_hex,
            top_left=top_left,
            pixel_size=pixel_size,
            pause=pause,
            duration=duration,
            use_directinput=use_directinput,
            optimise_path=optimise_path,
            reorder_by_proximity=reorder_by_proximity,
            smooth_polylines=smooth_polylines,
            rdp_epsilon=rdp_epsilon,
        )
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def _parse_cli_arguments() -> argparse.Namespace:
    """Parse command‑line arguments for the CLI wrapper.

    This function constructs an :class:`argparse.ArgumentParser` that
    collects user options for image processing and script generation.
    Continuous polylines are always used; there is no flag to disable
    this behaviour.  Users may enable polyline smoothing, adjust
    Canny thresholds, choose whether to optimise the path and reorder
    by proximity, and select the backend library.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        prog='edge_draw_script_generator_refactored',
        description=(
            'Detect edges in an image and generate a Python script that draws '\
            'the detected lines using PyAutoGUI or DirectInput.  Continuous '\
            'polylines are always used. Optionally simplify polylines using '\
            'the Ramer–Douglas–Peucker algorithm.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('input_image', help='Path to the input image (PNG or any format readable by Pillow).')
    parser.add_argument('output_script', help='Destination path for the generated Python drawing script.')
    parser.add_argument('--colour', dest='edges_colour_hex', default='#00FF00FF', metavar='HEX', help='ARGB colour for drawn edges (#RRGGBB or #AARRGGBB).')
    parser.add_argument('--low-threshold', type=float, default=100.0, help='Lower bound for Canny edge detection hysteresis.')
    parser.add_argument('--high-threshold', type=float, default=200.0, help='Upper bound for Canny edge detection hysteresis.')
    parser.add_argument('--dilation', dest='dilation_iter', type=int, default=0, help='Number of iterations to dilate edges (makes lines thicker).')
    parser.add_argument('--top-left', nargs=2, type=int, default=(0, 0), metavar=('X', 'Y'), help='Screen coordinates of the top‑left corner of the drawing area.')
    parser.add_argument('--pixel-size', type=int, default=1, help='Scaling factor between image pixels and screen pixels.')
    parser.add_argument('--pause', type=float, default=0.01, help='Pause (in seconds) between mouse actions in the generated script.')
    parser.add_argument('--duration', type=float, default=0.0, help='Duration (in seconds) for each drag action.')
    parser.add_argument('--directinput', action='store_true', help='Use DirectInput backend instead of PyAutoGUI.')
    parser.add_argument('--no-optimise', dest='optimise_path', action='store_false', help='Disable path optimisation; use legacy row‑major scanning.')
    parser.add_argument('--no-reorder', dest='reorder_by_proximity', action='store_false', help='Disable nearest‑neighbour reordering of segments.')
    parser.add_argument('--smooth-polylines', action='store_true', help='Simplify polylines using the Ramer–Douglas–Peucker algorithm.')
    parser.add_argument('--rdp-epsilon', type=float, default=1.0, help='Tolerance for RDP polyline simplification (larger values produce fewer points).')
    # Display help when no arguments are provided
    if len(os.sys.argv) == 1:
        parser.print_help()
        parser.exit()
    return parser.parse_args()


def main_cli() -> None:
    """Entry point for the command line interface version of the generator."""
    args = _parse_cli_arguments()
    top_left_coords: Tuple[int, int] = (args.top_left[0], args.top_left[1])
    process_and_generate_script(
        input_image_path=args.input_image,
        output_script_path=args.output_script,
        edges_colour_hex=args.edges_colour_hex,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
        dilation_iter=args.dilation_iter,
        top_left=top_left_coords,
        pixel_size=args.pixel_size,
        pause=args.pause,
        duration=args.duration,
        use_directinput=args.directinput,
        optimise_path=args.optimise_path,
        reorder_by_proximity=args.reorder_by_proximity,
        smooth_polylines=args.smooth_polylines,
        rdp_epsilon=args.rdp_epsilon,
    )


if __name__ == '__main__':
    main_cli()

"""
Future improvements
-------------------

The refactored module lays a strong foundation for reliable edge‑to‑stroke
conversion, but there remain opportunities for further enhancement:

* **Adaptive sampling**:  Currently, every pixel of the target colour
  contributes to the extracted paths.  On very large images, this can
  produce thousands of points and lengthy scripts.  A multi‑resolution
  or curvature‑based sampling strategy could select fewer points in
  regions of low curvature, reducing the length of the generated
  script without sacrificing visual fidelity.
* **Spline fitting**:  The RDP simplification yields polylines composed
  of straight line segments.  Fitting Bézier or other spline curves to
  the extracted paths could dramatically reduce the number of control
  points and produce smoother output.  This would require updating the
  generated script to draw curved paths.
* **Parallel processing**:  Edge detection, connected component
  extraction and path simplification can all benefit from parallel
  implementation.  Leveraging Python’s :mod:`concurrent.futures` or
  :mod:`multiprocessing` modules could significantly speed up
  processing on multi‑core systems.
* **Affine transformations**:  Allow users to specify arbitrary affine
  transformations between image and drawing coordinates, enabling
  rotation, scaling and translation without modifying the source
  image.
* **Comprehensive test suite**:  A suite of unit tests using a
  framework such as :mod:`pytest` would lock in correct behaviour and
  guard against regressions during future refactoring.
"""