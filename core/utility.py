def require_position_columns(available_columns, body: str):
    """
    Column names holding a body's position
    Raises a ValueError if the requested columns are not in the file
    """
    position_axes = ("X", "Y", "Z")
    tagged = [f"{body}_orig_{ax}_pos" for ax in position_axes]
    if all(c in available_columns for c in tagged):
        return tagged
    raise ValueError(f"The requested body origin is not in the file: {body}")
