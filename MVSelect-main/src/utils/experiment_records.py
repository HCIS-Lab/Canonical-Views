"""Write numerical measurements independently of diagnostic plotting."""
import json
from pathlib import Path


def json_value(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
    if hasattr(value, 'tolist'):
        return value.tolist()
    raise TypeError(f'Unsupported measurement type: {type(value).__name__}')


def append_epoch_record(path, record):
    text = json.dumps(record, default=json_value, allow_nan=False)
    with Path(path).open('a') as stream:
        stream.write(text + '\n')


def plot_after_records(plotter, error_path, *args, **kwargs):
    """Preserve completed numerical exports if an optional figure fails."""
    try:
        plotter(*args, **kwargs)
        return True
    except Exception:
        import traceback
        detail = traceback.format_exc()
        Path(error_path).write_text(detail)
        print(f'Numerical results saved; diagnostic plotting failed. Details: {error_path}')
        return False
