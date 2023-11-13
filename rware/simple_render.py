import numpy as np
from .warehouse import Direction
from PIL import Image

PIXEL_SCALE = 3
IMG_SCALE = 12

BASE_COLOUR = (0,0,255)
CHEQUER_V = 230

GOAL_COLOR = (  0,  0, 48)
SHELF_COLOR = (155,  220, 154)
REQUEST_COLOR = (155,  220, 255)

AGENT_COLOR = ( 24, 255, 255)
AGENT_CARRYING_COLOR = (0, 255, 255)

DIRECTION_MAP = {
    Direction.UP: 0,
    Direction.LEFT: 1,
    Direction.DOWN: 2,
    Direction.RIGHT: 3,
}


def _pixel_to_slice(pixel):
    return slice(pixel*PIXEL_SCALE, (pixel+1)*PIXEL_SCALE)


def _color_to_arr(color):
    return np.expand_dims(np.array(color, dtype=np.uint8), (1,2))


def _agent_pixel(direction, carrying_shelf):
    """Builds an agent with given direction and carrying status.

    Returns a pixel array and a mask array which emulates transparency.
    """
    color = AGENT_CARRYING_COLOR if carrying_shelf else AGENT_COLOR
    pixel = np.zeros((3, PIXEL_SCALE, PIXEL_SCALE), dtype=np.uint8)
    changed = np.zeros((3, PIXEL_SCALE, PIXEL_SCALE), dtype=bool)
    mid = PIXEL_SCALE/2
    if PIXEL_SCALE%2 == 0:
        for r in range(int(mid)):
            for c in range(int(mid-(r+1)), int(mid+(r+1))):
                pixel[:,r,c] = _color_to_arr(color).reshape(3)
                changed[:,r,c] = 1
    else:
        for r in range(int(np.ceil(mid))):
            for c in range(int(np.floor(mid-r)), int(np.ceil(mid+r))):
                pixel[:,r,c] = _color_to_arr(color).reshape(3)
                changed[:,r,c] = 1
    pixel = np.rot90(pixel, direction, axes=(1,2))
    changed = np.rot90(changed, direction, axes=(1,2))
    return pixel, changed


def render(env):
    """Renders the environment."""
    base_pixel = np.tile(
        _color_to_arr(BASE_COLOUR),
        (PIXEL_SCALE, PIXEL_SCALE)
        )
    grid_size = env.grid_size
    img = np.tile(base_pixel, grid_size)

    # chequer
    for y in range(grid_size[0]):
        for x in range(grid_size[1]):
            if (x-y)%2 == 0:
                r = _pixel_to_slice(y)
                c = _pixel_to_slice(x)
                img[2,r,c] = CHEQUER_V

    # goals
    for (x,y) in env.goals:
        r = _pixel_to_slice(y)
        c = _pixel_to_slice(x)
        img[:,r,c] = _color_to_arr(GOAL_COLOR)

    # shelves
    for shelf in env.shelfs:
        r = _pixel_to_slice(shelf.y)
        c = _pixel_to_slice(shelf.x)
        img[:,r,c] = _color_to_arr(SHELF_COLOR)

    # requested shelvess
    for shelf in env.request_queue:
        r = _pixel_to_slice(shelf.y)
        c = _pixel_to_slice(shelf.x)
        img[:,r,c] = _color_to_arr(REQUEST_COLOR)

    # agents
    for agent in env.agents_obj.values():
        r = _pixel_to_slice(agent.y)
        c = _pixel_to_slice(agent.x)
        ag,changed = _agent_pixel(DIRECTION_MAP[agent.dir], agent.carrying_shelf)
        img[:,r,c] = (img[:,r,c]*~changed) + ag

    rgb_img = Image.fromarray(
        img.transpose((1,2,0)), mode="HSV"
    ).convert(
        "RGB"
    ).resize(
        (IMG_SCALE*img.shape[2], IMG_SCALE*img.shape[1]),
        resample=Image.Resampling.NEAREST,
    )
    return np.asarray(rgb_img)
